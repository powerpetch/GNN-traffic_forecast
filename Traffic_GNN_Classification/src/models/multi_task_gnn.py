"""
Multi-Task GNN Model for Traffic Classification
==============================================

Single model that predicts both:
1. Traffic congestion level (4 classes: gridlock, congested, moderate, free_flow)
2. Rush hour classification (binary: rush_hour vs non_rush_hour)

Uses ST-GCN (Spatio-Temporal Graph Convolutional Network) architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import numpy as np
from typing import Dict, Tuple, Optional
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class TemporalConvBlock(nn.Module):
    """Temporal convolution block for processing time series data"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(TemporalConvBlock, self).__init__()
        
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=kernel_size//2)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_channels, seq_length]
        Returns:
            x: [batch_size, out_channels, seq_length]
        """
        x = self.conv1d(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class SpatialGraphConv(nn.Module):
    """Spatial graph convolution using GCN or GAT"""
    
    def __init__(self, in_features: int, out_features: int, 
                 conv_type: str = 'GCN', heads: int = 4):
        super(SpatialGraphConv, self).__init__()
        
        self.conv_type = conv_type
        
        if conv_type == 'GCN':
            self.conv = GCNConv(in_features, out_features)
        elif conv_type == 'GAT':
            self.conv = GATConv(in_features, out_features // heads, heads=heads, concat=True)
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")
        
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, in_features]
            edge_index: [2, num_edges]
        Returns:
            x: [num_nodes, out_features]
        """
        x = self.conv(x, edge_index)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class STGCNBlock(nn.Module):
    """ST-GCN Block: Temporal -> Spatial -> Temporal"""
    
    def __init__(self, in_channels: int, spatial_channels: int, 
                 out_channels: int, conv_type: str = 'GCN'):
        super(STGCNBlock, self).__init__()
        
        # Temporal convolutions
        self.temporal1 = TemporalConvBlock(in_channels, out_channels)
        self.temporal2 = TemporalConvBlock(out_channels, out_channels)
        
        # Spatial graph convolution
        self.spatial = SpatialGraphConv(out_channels, spatial_channels, conv_type)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_channels, num_nodes, seq_length]
            edge_index: [2, num_edges]
        Returns:
            x: [batch_size, out_channels, num_nodes, seq_length]
        """
        batch_size, in_channels, num_nodes, seq_length = x.shape
        residual = x
        
        # Reshape for temporal convolution: [batch_size * num_nodes, in_channels, seq_length]
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, num_nodes, in_channels, seq_length]
        x = x.view(batch_size * num_nodes, in_channels, seq_length)
        
        # First temporal convolution
        x = self.temporal1(x)  # [batch_size * num_nodes, out_channels, seq_length]
        out_channels = x.shape[1]
        
        # Reshape for spatial convolution: [batch_size * seq_length, num_nodes, out_channels]
        x = x.view(batch_size, num_nodes, out_channels, seq_length)
        x = x.permute(0, 3, 1, 2).contiguous()  # [batch_size, seq_length, num_nodes, out_channels]
        x = x.view(batch_size * seq_length, num_nodes, out_channels)
        
        # Spatial graph convolution
        spatial_outputs = []
        for t in range(batch_size * seq_length):
            if t < x.shape[0]:
                spatial_out = self.spatial(x[t], edge_index)  # [num_nodes, spatial_channels]
                spatial_outputs.append(spatial_out)
        
        if spatial_outputs:
            x = torch.stack(spatial_outputs, dim=0)  # [batch_size * seq_length, num_nodes, spatial_channels]
        
        # Reshape back: [batch_size, seq_length, num_nodes, spatial_channels]
        x = x.view(batch_size, seq_length, num_nodes, -1)
        x = x.permute(0, 3, 2, 1).contiguous()  # [batch_size, spatial_channels, num_nodes, seq_length]
        
        # Reshape for second temporal convolution
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, num_nodes, spatial_channels, seq_length]
        x = x.view(batch_size * num_nodes, x.shape[2], seq_length)
        
        # Second temporal convolution
        x = self.temporal2(x)  # [batch_size * num_nodes, out_channels, seq_length]
        
        # Reshape back to original format
        x = x.view(batch_size, num_nodes, out_channels, seq_length)
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, out_channels, num_nodes, seq_length]
        
        # Residual connection
        residual = self.residual(residual.permute(0, 2, 1, 3).contiguous().view(batch_size * num_nodes, in_channels, seq_length))
        residual = residual.view(batch_size, num_nodes, -1, seq_length).permute(0, 2, 1, 3).contiguous()
        
        return F.relu(x + residual)

class MultiTaskTrafficGNN(pl.LightningModule):
    """
    Multi-task GNN for traffic prediction
    Predicts both congestion level and rush hour classification
    """
    
    def __init__(self, 
                 num_features: int = 9,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 num_classes_congestion: int = 4,
                 num_classes_rush: int = 2,
                 conv_type: str = 'GCN',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4):
        super(MultiTaskTrafficGNN, self).__init__()
        
        self.save_hyperparameters()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes_congestion = num_classes_congestion
        self.num_classes_rush = num_classes_rush
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # ST-GCN layers
        self.stgcn_layers = nn.ModuleList()
        
        # First layer
        self.stgcn_layers.append(
            STGCNBlock(num_features, hidden_dim, hidden_dim, conv_type)
        )
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.stgcn_layers.append(
                STGCNBlock(hidden_dim, hidden_dim, hidden_dim, conv_type)
            )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification heads
        self.congestion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes_congestion)
        )
        
        self.rush_hour_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes_rush)
        )
        
        # Loss functions
        self.congestion_loss = nn.CrossEntropyLoss()
        self.rush_hour_loss = nn.CrossEntropyLoss()
        
        # Metrics storage
        self.train_metrics = {'congestion_acc': [], 'rush_hour_acc': []}
        self.val_metrics = {'congestion_acc': [], 'rush_hour_acc': []}
        
    def forward(self, data) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            data: PyTorch Geometric data object
        Returns:
            Dictionary with congestion and rush hour predictions
        """
        x = data.x  # [num_nodes, num_features]
        edge_index = data.edge_index  # [2, num_edges]
        
        # Reshape for ST-GCN: [batch_size, num_features, num_nodes, seq_length]
        # For simplicity, we'll treat each sample as batch_size=1, seq_length=1
        batch_size = 1
        seq_length = 1
        num_nodes = x.shape[0]
        
        x = x.unsqueeze(0).unsqueeze(-1)  # [1, num_nodes, num_features, 1]
        x = x.permute(0, 2, 1, 3)  # [1, num_features, num_nodes, 1]
        
        # Pass through ST-GCN layers
        for layer in self.stgcn_layers:
            x = layer(x, edge_index)
        
        # Global pooling: [batch_size, hidden_dim, 1, 1]
        x = self.global_pool(x)
        x = x.squeeze(-1).squeeze(-1)  # [batch_size, hidden_dim]
        
        # Classification
        congestion_logits = self.congestion_classifier(x)
        rush_hour_logits = self.rush_hour_classifier(x)
        
        return {
            'congestion_logits': congestion_logits,
            'rush_hour_logits': rush_hour_logits
        }
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        outputs = self(batch)
        
        congestion_logits = outputs['congestion_logits']
        rush_hour_logits = outputs['rush_hour_logits']
        
        # Get targets (take first element of sequence for simplicity)
        congestion_targets = batch.y_congestion[0] if batch.y_congestion.dim() > 0 else batch.y_congestion
        rush_hour_targets = batch.y_rush_hour[0] if batch.y_rush_hour.dim() > 0 else batch.y_rush_hour
        
        # Ensure targets are scalar
        if congestion_targets.dim() > 0:
            congestion_targets = congestion_targets[0]
        if rush_hour_targets.dim() > 0:
            rush_hour_targets = rush_hour_targets[0]
        
        # Calculate losses
        congestion_loss = self.congestion_loss(congestion_logits, congestion_targets.unsqueeze(0))
        rush_hour_loss = self.rush_hour_loss(rush_hour_logits, rush_hour_targets.unsqueeze(0))
        
        # Combined loss
        total_loss = congestion_loss + rush_hour_loss
        
        # Calculate accuracies
        congestion_pred = torch.argmax(congestion_logits, dim=1)
        rush_hour_pred = torch.argmax(rush_hour_logits, dim=1)
        
        congestion_acc = (congestion_pred == congestion_targets.unsqueeze(0)).float().mean()
        rush_hour_acc = (rush_hour_pred == rush_hour_targets.unsqueeze(0)).float().mean()
        
        # Log metrics
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_congestion_loss', congestion_loss)
        self.log('train_rush_hour_loss', rush_hour_loss)
        self.log('train_congestion_acc', congestion_acc)
        self.log('train_rush_hour_acc', rush_hour_acc)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        outputs = self(batch)
        
        congestion_logits = outputs['congestion_logits']
        rush_hour_logits = outputs['rush_hour_logits']
        
        # Get targets
        congestion_targets = batch.y_congestion[0] if batch.y_congestion.dim() > 0 else batch.y_congestion
        rush_hour_targets = batch.y_rush_hour[0] if batch.y_rush_hour.dim() > 0 else batch.y_rush_hour
        
        # Ensure targets are scalar
        if congestion_targets.dim() > 0:
            congestion_targets = congestion_targets[0]
        if rush_hour_targets.dim() > 0:
            rush_hour_targets = rush_hour_targets[0]
        
        # Calculate losses
        congestion_loss = self.congestion_loss(congestion_logits, congestion_targets.unsqueeze(0))
        rush_hour_loss = self.rush_hour_loss(rush_hour_logits, rush_hour_targets.unsqueeze(0))
        
        total_loss = congestion_loss + rush_hour_loss
        
        # Calculate accuracies
        congestion_pred = torch.argmax(congestion_logits, dim=1)
        rush_hour_pred = torch.argmax(rush_hour_logits, dim=1)
        
        congestion_acc = (congestion_pred == congestion_targets.unsqueeze(0)).float().mean()
        rush_hour_acc = (rush_hour_pred == rush_hour_targets.unsqueeze(0)).float().mean()
        
        # Log metrics
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_congestion_loss', congestion_loss)
        self.log('val_rush_hour_loss', rush_hour_loss)
        self.log('val_congestion_acc', congestion_acc)
        self.log('val_rush_hour_acc', rush_hour_acc)
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
    
    def predict(self, data) -> Dict[str, np.ndarray]:
        """Make predictions"""
        self.eval()
        with torch.no_grad():
            outputs = self(data)
            
            congestion_probs = F.softmax(outputs['congestion_logits'], dim=1)
            rush_hour_probs = F.softmax(outputs['rush_hour_logits'], dim=1)
            
            congestion_pred = torch.argmax(congestion_probs, dim=1)
            rush_hour_pred = torch.argmax(rush_hour_probs, dim=1)
            
            return {
                'congestion_prediction': congestion_pred.cpu().numpy(),
                'congestion_probabilities': congestion_probs.cpu().numpy(),
                'rush_hour_prediction': rush_hour_pred.cpu().numpy(),
                'rush_hour_probabilities': rush_hour_probs.cpu().numpy()
            }

class SimpleMultiTaskGNN(nn.Module):
    """
    Simplified version for easier training and debugging
    """
    
    def __init__(self, num_features: int = 9, hidden_dim: int = 64):
        super(SimpleMultiTaskGNN, self).__init__()
        
        # Simple feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Classification heads
        self.congestion_head = nn.Linear(hidden_dim, 4)  # 4 congestion classes
        self.rush_hour_head = nn.Linear(hidden_dim, 2)   # 2 rush hour classes
        
    def forward(self, x):
        """
        Simple forward pass
        
        Args:
            x: [num_nodes, num_features] or [batch_size, num_features]
        """
        if x.dim() == 2 and x.shape[0] > 1:
            # Multiple nodes, aggregate by mean
            x = x.mean(dim=0, keepdim=True)
        elif x.dim() == 2:
            # Single node or already aggregated
            pass
        else:
            # Flatten if needed
            x = x.view(1, -1)
        
        # Feature transformation
        features = self.feature_transform(x)
        
        # Classification
        congestion_logits = self.congestion_head(features)
        rush_hour_logits = self.rush_hour_head(features)
        
        return {
            'congestion_logits': congestion_logits,
            'rush_hour_logits': rush_hour_logits
        }

if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy data
    num_nodes = 10
    num_features = 9
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index)
    
    # Test simple model
    simple_model = SimpleMultiTaskGNN(num_features, hidden_dim=64)
    simple_outputs = simple_model(x)
    
    print("Simple model outputs:")
    print(f"Congestion logits shape: {simple_outputs['congestion_logits'].shape}")
    print(f"Rush hour logits shape: {simple_outputs['rush_hour_logits'].shape}")
    
    # Test full model
    model = MultiTaskTrafficGNN(num_features=num_features, hidden_dim=32, num_layers=1)
    
    # Add dummy targets to data
    data.y_congestion = torch.tensor([1])  # Single target
    data.y_rush_hour = torch.tensor([0])   # Single target
    
    outputs = model(data)
    print("\nFull model outputs:")
    print(f"Congestion logits shape: {outputs['congestion_logits'].shape}")
    print(f"Rush hour logits shape: {outputs['rush_hour_logits'].shape}")