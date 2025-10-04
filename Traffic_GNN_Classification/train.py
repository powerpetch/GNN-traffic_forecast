"""
Training Script for Multi-Task Traffic GNN
==========================================

Trains the GNN model to predict both:
1. Traffic congestion classification
2. Rush hour classification

Includes evaluation metrics and model saving
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.data_processor import TrafficDataProcessor
from utils.graph_constructor import GraphConstructor
from models.multi_task_gnn import MultiTaskTrafficGNN, SimpleMultiTaskGNN

class TrafficGNNTrainer:
    """Main trainer class for traffic GNN"""
    
    def __init__(self, 
                 data_path: str = "d:/user/Data_project/GNN_fore/src/data/raw",
                 output_path: str = "d:/user/Data_project/Traffic_GNN_Classification/outputs",
                 use_simple_model: bool = True):
        
        self.data_path = data_path
        self.output_path = output_path
        self.use_simple_model = use_simple_model
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize components
        self.processor = None
        self.graph_constructor = None
        self.model = None
        self.processed_data = None
        
        # Training data
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def prepare_data(self, force_reprocess: bool = False):
        """Prepare and process all data for training"""
        print("=== Preparing Data ===")
        
        processed_data_path = os.path.join(self.output_path, "processed_data.pkl")
        graph_data_path = os.path.join(self.output_path, "graph_data.pkl")
        
        # Check if processed data exists
        if not force_reprocess and os.path.exists(processed_data_path):
            print("Loading existing processed data...")
            self.processor = TrafficDataProcessor(self.data_path)
            self.processed_data = self.processor.load_processed_data(processed_data_path)
        else:
            print("Processing raw data...")
            self.processor = TrafficDataProcessor(self.data_path)
            self.processed_data = self.processor.process_all_data()
            self.processor.save_processed_data(processed_data_path)
        
        if self.processed_data is None or len(self.processed_data) == 0:
            raise ValueError("No processed data available")
        
        print(f"Processed data shape: {self.processed_data.shape}")
        print(f"Congestion distribution:\n{self.processed_data['congestion_level'].value_counts()}")
        print(f"Rush hour distribution:\n{self.processed_data['is_rush_hour'].value_counts()}")
        
        return self.processed_data
    
    def create_simple_datasets(self):
        """Create simple datasets without graph structure (for SimpleMultiTaskGNN)"""
        print("Creating simple datasets...")
        
        # Define feature columns
        feature_cols = [
            'mean_speed', 'median_speed', 'speed_std', 'count_probes', 'quality_score',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend'
        ]
        
        # Filter data with sufficient quality (use lower threshold for small datasets)
        initial_size = len(self.processed_data)
        quality_data = self.processed_data[self.processed_data['quality_score'] > 0.3].copy()
        
        if len(quality_data) < 100:  # If we have very few high-quality samples
            print(f"Warning: Only {len(quality_data)} high-quality samples found. Lowering quality threshold.")
            quality_data = self.processed_data[self.processed_data['quality_score'] > 0.1].copy()
            
        if len(quality_data) < 50:  # Still too few
            print(f"Warning: Only {len(quality_data)} samples found. Using all data.")
            quality_data = self.processed_data.copy()
        
        # Ensure we have all required columns
        missing_cols = [col for col in feature_cols if col not in quality_data.columns]
        for col in missing_cols:
            quality_data[col] = 0.0
        
        # Features and targets
        X = quality_data[feature_cols].values
        y_congestion = quality_data['congestion_label'].values
        y_rush_hour = quality_data['is_rush_hour'].values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Split data (adjust for small datasets)
        n_samples = len(X)
        
        if n_samples < 10:
            # For very small datasets, use simpler split
            print(f"Warning: Small dataset ({n_samples} samples). Using simple split.")
            n_train = max(1, int(0.8 * n_samples))  # At least 1 sample for training
            n_val = max(0, int(0.1 * n_samples))   # Can be 0 for very small datasets
        else:
            # Normal split for larger datasets
            n_train = int(0.7 * n_samples)
            n_val = int(0.15 * n_samples)
        
        # Random shuffle
        indices = np.random.permutation(n_samples)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
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
        
        print(f"Data split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
        print(f"Feature shape: {X.shape}")
        
        return self.train_data, self.val_data, self.test_data
    
    def train_simple_model(self, epochs: int = 100, batch_size: int = 32, patience: int = 20):
        """Train the simple model with efficiency improvements"""
        print("=== Training Simple Model (Enhanced Efficiency) ===")
        
        # Create model
        num_features = self.train_data['X'].shape[1]
        self.model = SimpleMultiTaskGNN(num_features=num_features, hidden_dim=64)
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        print(f"Using device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Use AdamW optimizer (better weight decay)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        congestion_criterion = torch.nn.CrossEntropyLoss()
        rush_hour_criterion = torch.nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        # Adjust batch size for small datasets
        n_train = len(self.train_data['X'])
        n_val = len(self.val_data['X'])
        
        # Use smaller batch size if dataset is small
        actual_batch_size = min(batch_size, max(1, n_train))
        val_batch_size = min(batch_size, max(1, n_val)) if n_val > 0 else 1
        
        print(f"Using batch size: {actual_batch_size} (train), {val_batch_size} (val)")
        
        # Training data loaders
        train_dataset = torch.utils.data.TensorDataset(
            self.train_data['X'], 
            self.train_data['y_congestion'], 
            self.train_data['y_rush_hour']
        )
        val_dataset = torch.utils.data.TensorDataset(
            self.val_data['X'], 
            self.val_data['y_congestion'], 
            self.val_data['y_rush_hour']
        )
        
        train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, drop_last=False) if n_val > 0 else None
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        train_acc_congestion = []
        train_acc_rush_hour = []
        val_acc_congestion = []
        val_acc_rush_hour = []
        learning_rates = []
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Early stopping patience: {patience} epochs")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_congestion_correct = 0
            train_rush_hour_correct = 0
            train_total = 0
            
            for batch_idx, (X_batch, y_congestion_batch, y_rush_hour_batch) in enumerate(train_loader):
                X_batch = X_batch.to(device)
                y_congestion_batch = y_congestion_batch.to(device)
                y_rush_hour_batch = y_rush_hour_batch.to(device)
                
                # Data augmentation: Add small noise during training
                if self.model.training:
                    noise = torch.randn_like(X_batch) * 0.01
                    X_batch = X_batch + noise
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(X_batch)
                
                # Calculate losses
                congestion_loss = congestion_criterion(outputs['congestion_logits'], y_congestion_batch)
                rush_hour_loss = rush_hour_criterion(outputs['rush_hour_logits'], y_rush_hour_batch)
                total_loss = congestion_loss + rush_hour_loss
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Statistics
                train_loss += total_loss.item()
                
                _, congestion_pred = torch.max(outputs['congestion_logits'].data, 1)
                _, rush_hour_pred = torch.max(outputs['rush_hour_logits'].data, 1)
                
                train_total += y_congestion_batch.size(0)
                train_congestion_correct += (congestion_pred == y_congestion_batch).sum().item()
                train_rush_hour_correct += (rush_hour_pred == y_rush_hour_batch).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_congestion_correct = 0
            val_rush_hour_correct = 0
            val_total = 0
            
            if val_loader is not None:
                with torch.no_grad():
                    for X_batch, y_congestion_batch, y_rush_hour_batch in val_loader:
                        X_batch = X_batch.to(device)
                        y_congestion_batch = y_congestion_batch.to(device)
                        y_rush_hour_batch = y_rush_hour_batch.to(device)
                        
                        outputs = self.model(X_batch)
                        
                        congestion_loss = congestion_criterion(outputs['congestion_logits'], y_congestion_batch)
                        rush_hour_loss = rush_hour_criterion(outputs['rush_hour_logits'], y_rush_hour_batch)
                        total_loss = congestion_loss + rush_hour_loss
                        
                        val_loss += total_loss.item()
                        
                        _, congestion_pred = torch.max(outputs['congestion_logits'].data, 1)
                        _, rush_hour_pred = torch.max(outputs['rush_hour_logits'].data, 1)
                        
                        val_total += y_congestion_batch.size(0)
                        val_congestion_correct += (congestion_pred == y_congestion_batch).sum().item()
                        val_rush_hour_correct += (rush_hour_pred == y_rush_hour_batch).sum().item()
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader) if val_loader is not None else 0.0
            
            train_congestion_acc = 100 * train_congestion_correct / train_total
            train_rush_hour_acc = 100 * train_rush_hour_correct / train_total
            val_congestion_acc = 100 * val_congestion_correct / val_total if val_total > 0 else 0.0
            val_rush_hour_acc = 100 * val_rush_hour_correct / val_total if val_total > 0 else 0.0
            
            # Store metrics
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_acc_congestion.append(train_congestion_acc)
            train_acc_rush_hour.append(train_rush_hour_acc)
            val_acc_congestion.append(val_congestion_acc)
            val_acc_rush_hour.append(val_rush_hour_acc)
            learning_rates.append(optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}]')
                print(f'  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
                print(f'  Train - Congestion Acc: {train_congestion_acc:.2f}%, Rush Hour Acc: {train_rush_hour_acc:.2f}%')
                print(f'  Val - Congestion Acc: {val_congestion_acc:.2f}%, Rush Hour Acc: {val_rush_hour_acc:.2f}%')
                print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model and check early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'val_congestion_acc': val_congestion_acc,
                    'val_rush_hour_acc': val_rush_hour_acc
                }, os.path.join(self.output_path, 'best_model.pth'))
                if (epoch + 1) % 10 == 0:
                    print(f'  ✓ New best model saved!')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'\nEarly stopping triggered after {epoch+1} epochs')
                    print(f'No improvement for {patience} epochs')
                    break
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_acc_congestion': train_acc_congestion,
            'train_acc_rush_hour': train_acc_rush_hour,
            'val_acc_congestion': val_acc_congestion,
            'val_acc_rush_hour': val_acc_rush_hour,
            'learning_rates': learning_rates,
            'best_epoch': epoch,
            'total_epochs': len(train_losses)
        }
        
        with open(os.path.join(self.output_path, 'training_history.pkl'), 'wb') as f:
            pickle.dump(history, f)
        
        print(f"\n=== Training Summary ===")
        print(f"Total epochs trained: {len(train_losses)}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best congestion accuracy: {max(val_acc_congestion):.2f}%")
        print(f"Best rush hour accuracy: {max(val_acc_rush_hour):.2f}%")
        print(f"Final learning rate: {learning_rates[-1]:.6f}")
        
        return history
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        print("=== Evaluating Model ===")
        
        if self.model is None:
            # Load best model
            checkpoint = torch.load(os.path.join(self.output_path, 'best_model.pth'))
            num_features = self.test_data['X'].shape[1]
            self.model = SimpleMultiTaskGNN(num_features=num_features, hidden_dim=64)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        
        # Test data
        X_test = self.test_data['X'].to(device)
        y_congestion_test = self.test_data['y_congestion'].numpy()
        y_rush_hour_test = self.test_data['y_rush_hour'].numpy()
        
        # Predictions
        with torch.no_grad():
            outputs = self.model(X_test)
            
            _, congestion_pred = torch.max(outputs['congestion_logits'].data, 1)
            _, rush_hour_pred = torch.max(outputs['rush_hour_logits'].data, 1)
            
            congestion_pred = congestion_pred.cpu().numpy()
            rush_hour_pred = rush_hour_pred.cpu().numpy()
        
        # Calculate metrics
        congestion_acc = accuracy_score(y_congestion_test, congestion_pred)
        rush_hour_acc = accuracy_score(y_rush_hour_test, rush_hour_pred)
        
        print(f"Test Accuracy - Congestion: {congestion_acc:.4f}, Rush Hour: {rush_hour_acc:.4f}")
        
        # Classification reports
        congestion_labels = ['Gridlock', 'Congested', 'Moderate', 'Free Flow']
        rush_hour_labels = ['Non-Rush Hour', 'Rush Hour']
        
        print("\nCongestion Classification Report:")
        print(classification_report(y_congestion_test, congestion_pred, 
                                  target_names=congestion_labels))
        
        print("\nRush Hour Classification Report:")
        print(classification_report(y_rush_hour_test, rush_hour_pred, 
                                  target_names=rush_hour_labels))
        
        # Save evaluation results
        eval_results = {
            'congestion_accuracy': congestion_acc,
            'rush_hour_accuracy': rush_hour_acc,
            'congestion_predictions': congestion_pred,
            'rush_hour_predictions': rush_hour_pred,
            'congestion_targets': y_congestion_test,
            'rush_hour_targets': y_rush_hour_test
        }
        
        with open(os.path.join(self.output_path, 'evaluation_results.pkl'), 'wb') as f:
            pickle.dump(eval_results, f)
        
        return eval_results
    
    def plot_training_history(self):
        """Plot training history with learning rate"""
        with open(os.path.join(self.output_path, 'training_history.pkl'), 'rb') as f:
            history = pickle.load(f)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss plots
        axes[0, 0].plot(history['train_losses'], label='Train Loss')
        axes[0, 0].plot(history['val_losses'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Congestion accuracy
        axes[0, 1].plot(history['train_acc_congestion'], label='Train Acc')
        axes[0, 1].plot(history['val_acc_congestion'], label='Val Acc')
        axes[0, 1].set_title('Congestion Classification Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Rush hour accuracy
        axes[1, 0].plot(history['train_acc_rush_hour'], label='Train Acc')
        axes[1, 0].plot(history['val_acc_rush_hour'], label='Val Acc')
        axes[1, 0].set_title('Rush Hour Classification Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')  
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate over time
        if 'learning_rates' in history:
            axes[1, 1].plot(history['learning_rates'], color='orange')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        # Final metrics comparison
        final_metrics = ['Congestion Acc', 'Rush Hour Acc']
        train_final = [history['train_acc_congestion'][-1], history['train_acc_rush_hour'][-1]]
        val_final = [history['val_acc_congestion'][-1], history['val_acc_rush_hour'][-1]]
        
        x = np.arange(len(final_metrics))
        width = 0.35
        
        axes[1, 2].bar(x - width/2, train_final, width, label='Train')
        axes[1, 2].bar(x + width/2, val_final, width, label='Validation')
        axes[1, 2].set_title('Final Model Performance')
        axes[1, 2].set_ylabel('Accuracy (%)')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(final_metrics)
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_path, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {plot_path}")
        plt.close()  # Close the plot instead of showing it
    
    def run_complete_pipeline(self, epochs: int = 100, batch_size: int = 32, force_reprocess: bool = False):
        """Run the complete training pipeline"""
        print("=== Starting Complete Training Pipeline ===")
        
        # Prepare data
        self.prepare_data(force_reprocess=force_reprocess)
        
        # Create datasets
        self.create_simple_datasets()
        
        # Train model with patience parameter
        history = self.train_simple_model(epochs=epochs, batch_size=batch_size, patience=20)
        
        # Evaluate model
        eval_results = self.evaluate_model()
        
        # Plot results
        self.plot_training_history()
        
        print("=== Pipeline Complete ===")
        print(f"Final Results:")
        print(f"  Congestion Classification Accuracy: {eval_results['congestion_accuracy']:.4f}")
        print(f"  Rush Hour Classification Accuracy: {eval_results['rush_hour_accuracy']:.4f}")
        
        return history, eval_results

def main():
    """Main training function with efficiency improvements"""
    parser = argparse.ArgumentParser(description='Train Multi-Task Traffic GNN (Enhanced Efficiency)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--force_reprocess', action='store_true', help='Force reprocessing of data')
    parser.add_argument('--data_path', type=str, 
                       default="d:/user/Data_project/GNN_fore/src/data/raw",
                       help='Path to raw data')
    parser.add_argument('--output_path', type=str,
                       default="d:/user/Data_project/Traffic_GNN_Classification/outputs",
                       help='Path to save outputs')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = TrafficGNNTrainer(
        data_path=args.data_path,
        output_path=args.output_path,
        use_simple_model=True
    )
    
    # Run pipeline
    print(f"\n" + "="*60)
    print(f"Enhanced Training Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Early Stopping Patience: {args.patience}")
    print(f"  Optimizer: AdamW with weight decay")
    print(f"  LR Scheduler: ReduceLROnPlateau")
    print(f"  Data Augmentation: Gaussian noise (std=0.01)")
    print(f"  Gradient Clipping: max_norm=1.0")
    print(f"="*60 + "\n")
    
    history, eval_results = trainer.run_complete_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        force_reprocess=args.force_reprocess
    )

if __name__ == "__main__":
    main()