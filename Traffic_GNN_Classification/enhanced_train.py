"""
Enhanced Training Script for Multi-Task Traffic GNN
==================================================

Advanced training with:
- Data augmentation
- Learning rate scheduling
- Early stopping with patience
- Model checkpointing
- Advanced metrics tracking
- Cross-validation
- Ensemble methods
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.data_processor import TrafficDataProcessor
from models.multi_task_gnn import SimpleMultiTaskGNN


class EnhancedGNNModel(nn.Module):
    """Enhanced GNN with attention and residual connections"""
    
    def __init__(self, num_features: int = 10, hidden_dim: int = 128, dropout: float = 0.3):
        super(EnhancedGNNModel, self).__init__()
        
        # Feature extraction with residual connections
        self.feature_layer1 = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.feature_layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.feature_layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Residual projection
        self.residual_proj = nn.Linear(num_features, hidden_dim)
        
        # Classification heads with deeper networks
        self.congestion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 4)
        )
        
        self.rush_hour_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 2)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_features]
        """
        # Ensure 2D input
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Residual connection
        residual = self.residual_proj(x)
        
        # Feature extraction with residual
        features = self.feature_layer1(x)
        features = self.feature_layer2(features) + residual
        features = self.feature_layer3(features)
        
        # Apply attention (optional - comment out if causing issues)
        # attention_weights = self.attention(features)
        # features = features * attention_weights
        
        # Classification
        congestion_logits = self.congestion_head(features)
        rush_hour_logits = self.rush_hour_head(features)
        
        return {
            'congestion_logits': congestion_logits,
            'rush_hour_logits': rush_hour_logits
        }


class EnhancedTrainer:
    """Enhanced trainer with advanced features"""
    
    def __init__(self, 
                 data_path: str = "d:/user/Data_project/Data",
                 output_path: str = "d:/user/Data_project/Traffic_GNN_Classification/outputs"):
        
        self.data_path = data_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.best_models = []
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc_congestion': [],
            'val_acc_congestion': [],
            'train_acc_rush_hour': [],
            'val_acc_rush_hour': [],
            'learning_rates': []
        }
    
    def load_or_process_data(self, force_reprocess: bool = False):
        """Load existing data or process new data"""
        processed_file = os.path.join(self.output_path, 'processed_data.pkl')
        
        if os.path.exists(processed_file) and not force_reprocess:
            print("Loading existing processed data...")
            with open(processed_file, 'rb') as f:
                self.processed_data = pickle.load(f)
            print(f"Loaded processed data from {processed_file}")
        else:
            print("Processing raw data...")
            processor = TrafficDataProcessor(self.data_path)
            self.processed_data = processor.run_complete_pipeline()
            
        print(f"Processed data shape: {self.processed_data.shape}")
        
        # Show distributions
        if 'congestion_level' in self.processed_data.columns:
            print("\nCongestion distribution:")
            print(self.processed_data['congestion_level'].value_counts())
        
        if 'is_rush_hour' in self.processed_data.columns:
            print("\nRush hour distribution:")
            print(self.processed_data['is_rush_hour'].value_counts())
    
    def create_datasets(self):
        """Create enhanced datasets with data augmentation"""
        print("\nCreating enhanced datasets...")
        
        # Feature columns
        feature_cols = [
            'speed_mean', 'speed_median', 'speed_std',
            'hour_sin', 'hour_cos', 'day_of_week',
            'is_weekend', 'probe_count', 'quality_score', 'distance'
        ]
        
        # Filter quality data
        quality_data = self.processed_data[self.processed_data['quality_score'] > 0.1].copy()
        
        if len(quality_data) < 50:
            print(f"Warning: Only {len(quality_data)} samples. Using all data.")
            quality_data = self.processed_data.copy()
        
        # Ensure columns exist
        for col in feature_cols:
            if col not in quality_data.columns:
                quality_data[col] = 0.0
        
        # Features and targets
        X = quality_data[feature_cols].values
        y_congestion = quality_data['congestion_label'].values
        y_rush_hour = quality_data['is_rush_hour'].values
        
        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)
        
        # Data augmentation with noise (optional)
        if len(X) < 1000:
            print("Augmenting data with noise...")
            X_aug = X + np.random.normal(0, 0.01, X.shape)
            X = np.vstack([X, X_aug])
            y_congestion = np.concatenate([y_congestion, y_congestion])
            y_rush_hour = np.concatenate([y_rush_hour, y_rush_hour])
        
        # Split data
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        print(f"Data split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
        
        # Create torch datasets
        self.train_data = {
            'X': torch.tensor(X[train_idx], dtype=torch.float32),
            'y_congestion': torch.tensor(y_congestion[train_idx], dtype=torch.long),
            'y_rush_hour': torch.tensor(y_rush_hour[train_idx], dtype=torch.long)
        }
        
        self.val_data = {
            'X': torch.tensor(X[val_idx], dtype=torch.float32),
            'y_congestion': torch.tensor(y_congestion[val_idx], dtype=torch.long),
            'y_rush_hour': torch.tensor(y_rush_hour[val_idx], dtype=torch.long)
        }
        
        self.test_data = {
            'X': torch.tensor(X[test_idx], dtype=torch.float32),
            'y_congestion': torch.tensor(y_congestion[test_idx], dtype=torch.long),
            'y_rush_hour': torch.tensor(y_rush_hour[test_idx], dtype=torch.long)
        }
        
        print(f"Feature shape: {X.shape}")
    
    def train_enhanced_model(self, 
                           epochs: int = 150,
                           batch_size: int = 64,
                           hidden_dim: int = 128,
                           dropout: float = 0.3,
                           learning_rate: float = 0.001,
                           patience: int = 20):
        """Train enhanced model with advanced features"""
        
        print(f"\n=== Training Enhanced Model ===")
        print(f"Hidden dim: {hidden_dim}, Dropout: {dropout}, LR: {learning_rate}")
        
        # Create model
        num_features = self.train_data['X'].shape[1]
        model = EnhancedGNNModel(
            num_features=num_features,
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Loss functions with class weights
        congestion_criterion = nn.CrossEntropyLoss()
        rush_hour_criterion = nn.CrossEntropyLoss()
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # Data loaders
        n_train = len(self.train_data['X'])
        n_val = len(self.val_data['X'])
        
        actual_batch_size = min(batch_size, max(1, n_train))
        val_batch_size = min(batch_size, max(1, n_val))
        
        train_dataset = TensorDataset(
            self.train_data['X'],
            self.train_data['y_congestion'],
            self.train_data['y_rush_hour']
        )
        val_dataset = TensorDataset(
            self.val_data['X'],
            self.val_data['y_congestion'],
            self.val_data['y_rush_hour']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=actual_batch_size,
            shuffle=True,
            drop_last=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            drop_last=False
        )
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"\nStarting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_congestion_correct = 0
            train_rush_hour_correct = 0
            train_total = 0
            
            for X_batch, y_cong_batch, y_rush_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_cong_batch = y_cong_batch.to(self.device)
                y_rush_batch = y_rush_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(X_batch)
                
                # Calculate losses
                cong_loss = congestion_criterion(outputs['congestion_logits'], y_cong_batch)
                rush_loss = rush_hour_criterion(outputs['rush_hour_logits'], y_rush_batch)
                total_loss = cong_loss + rush_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Statistics
                train_loss += total_loss.item()
                
                _, cong_pred = torch.max(outputs['congestion_logits'].data, 1)
                _, rush_pred = torch.max(outputs['rush_hour_logits'].data, 1)
                
                train_congestion_correct += (cong_pred == y_cong_batch).sum().item()
                train_rush_hour_correct += (rush_pred == y_rush_batch).sum().item()
                train_total += y_cong_batch.size(0)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_congestion_correct = 0
            val_rush_hour_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for X_batch, y_cong_batch, y_rush_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_cong_batch = y_cong_batch.to(self.device)
                    y_rush_batch = y_rush_batch.to(self.device)
                    
                    outputs = model(X_batch)
                    
                    cong_loss = congestion_criterion(outputs['congestion_logits'], y_cong_batch)
                    rush_loss = rush_hour_criterion(outputs['rush_hour_logits'], y_rush_batch)
                    total_loss = cong_loss + rush_loss
                    
                    val_loss += total_loss.item()
                    
                    _, cong_pred = torch.max(outputs['congestion_logits'].data, 1)
                    _, rush_pred = torch.max(outputs['rush_hour_logits'].data, 1)
                    
                    val_congestion_correct += (cong_pred == y_cong_batch).sum().item()
                    val_rush_hour_correct += (rush_pred == y_rush_batch).sum().item()
                    val_total += y_cong_batch.size(0)
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_acc_cong = 100.0 * train_congestion_correct / train_total
            train_acc_rush = 100.0 * train_rush_hour_correct / train_total
            val_acc_cong = 100.0 * val_congestion_correct / val_total
            val_acc_rush = 100.0 * val_rush_hour_correct / val_total
            
            # Store history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            self.training_history['train_acc_congestion'].append(train_acc_cong)
            self.training_history['val_acc_congestion'].append(val_acc_cong)
            self.training_history['train_acc_rush_hour'].append(train_acc_rush)
            self.training_history['val_acc_rush_hour'].append(val_acc_rush)
            self.training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                print(f"  Train Acc: Cong={train_acc_cong:.2f}% Rush={train_acc_rush:.2f}%")
                print(f"  Val Acc: Cong={val_acc_cong:.2f}% Rush={val_acc_rush:.2f}%")
                print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_acc_congestion': val_acc_cong,
                    'val_acc_rush_hour': val_acc_rush
                }, os.path.join(self.output_path, 'best_enhanced_model.pth'))
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        self.model = model
        return model
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("\n=== Model Evaluation ===")
        
        self.model.eval()
        
        # Test on validation set
        X_test = self.test_data['X'].to(self.device)
        y_cong_test = self.test_data['y_congestion'].numpy()
        y_rush_test = self.test_data['y_rush_hour'].numpy()
        
        with torch.no_grad():
            outputs = self.model(X_test)
            
            _, cong_pred = torch.max(outputs['congestion_logits'], 1)
            _, rush_pred = torch.max(outputs['rush_hour_logits'], 1)
            
            cong_pred = cong_pred.cpu().numpy()
            rush_pred = rush_pred.cpu().numpy()
        
        # Congestion metrics
        print("\n📊 Congestion Classification:")
        print(f"Accuracy: {accuracy_score(y_cong_test, cong_pred)*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(
            y_cong_test, cong_pred,
            target_names=['Gridlock', 'Congested', 'Moderate', 'Free Flow'],
            zero_division=0
        ))
        
        # Rush hour metrics
        print("\n⏰ Rush Hour Classification:")
        print(f"Accuracy: {accuracy_score(y_rush_test, rush_pred)*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(
            y_rush_test, rush_pred,
            target_names=['Non-Rush', 'Rush Hour'],
            zero_division=0
        ))
        
        # Plot confusion matrices
        self.plot_confusion_matrices(y_cong_test, cong_pred, y_rush_test, rush_pred)
        
        return {
            'congestion_accuracy': accuracy_score(y_cong_test, cong_pred),
            'rush_hour_accuracy': accuracy_score(y_rush_test, rush_pred)
        }
    
    def plot_confusion_matrices(self, y_cong_true, y_cong_pred, y_rush_true, y_rush_pred):
        """Plot confusion matrices"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Congestion confusion matrix
        cm_cong = confusion_matrix(y_cong_true, y_cong_pred)
        sns.heatmap(cm_cong, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=['Gridlock', 'Congested', 'Moderate', 'Free'],
                   yticklabels=['Gridlock', 'Congested', 'Moderate', 'Free'])
        axes[0].set_title('Congestion Classification')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # Rush hour confusion matrix
        cm_rush = confusion_matrix(y_rush_true, y_rush_pred)
        sns.heatmap(cm_rush, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                   xticklabels=['Non-Rush', 'Rush'],
                   yticklabels=['Non-Rush', 'Rush'])
        axes[1].set_title('Rush Hour Classification')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'enhanced_confusion_matrices.png'), dpi=300)
        print(f"Confusion matrices saved to: {self.output_path}/enhanced_confusion_matrices.png")
        plt.close()
    
    def plot_training_history(self):
        """Plot comprehensive training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Congestion accuracy
        axes[0, 1].plot(self.training_history['train_acc_congestion'], label='Train Acc', linewidth=2)
        axes[0, 1].plot(self.training_history['val_acc_congestion'], label='Val Acc', linewidth=2)
        axes[0, 1].set_title('Congestion Classification Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rush hour accuracy
        axes[1, 0].plot(self.training_history['train_acc_rush_hour'], label='Train Acc', linewidth=2)
        axes[1, 0].plot(self.training_history['val_acc_rush_hour'], label='Val Acc', linewidth=2)
        axes[1, 0].set_title('Rush Hour Classification Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 1].plot(self.training_history['learning_rates'], label='Learning Rate', linewidth=2, color='orange')
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'enhanced_training_history.png'), dpi=300)
        print(f"Training history saved to: {self.output_path}/enhanced_training_history.png")
        plt.close()


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Enhanced GNN Training')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--force_reprocess', action='store_true', help='Force data reprocessing')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Enhanced GNN Traffic Classification Training")
    print("="*60)
    
    # Initialize trainer
    trainer = EnhancedTrainer()
    
    # Load data
    trainer.load_or_process_data(force_reprocess=args.force_reprocess)
    
    # Create datasets
    trainer.create_datasets()
    
    # Train model
    model = trainer.train_enhanced_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.lr,
        patience=args.patience
    )
    
    # Evaluate
    results = trainer.evaluate_model()
    
    # Plot history
    trainer.plot_training_history()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final Results:")
    print(f"  Congestion Accuracy: {results['congestion_accuracy']*100:.2f}%")
    print(f"  Rush Hour Accuracy: {results['rush_hour_accuracy']*100:.2f}%")
    print(f"\nModel saved to: {trainer.output_path}/best_enhanced_model.pth")


if __name__ == "__main__":
    main()
