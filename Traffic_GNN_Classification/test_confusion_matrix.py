"""
Test script to generate confusion matrix only (without full training)
"""
import os
import sys
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from multi_task_gnn import SimpleMultiTaskGNN

def test_confusion_matrix():
    """Generate confusion matrix from existing trained model"""
    print("="*60)
    print("Testing Confusion Matrix Generation")
    print("="*60)
    
    # Use absolute path to correct project location
    output_path = "d:/user/Data_project/Project_data/Traffic_GNN_Classification/outputs"
    
    # Check if required files exist
    model_path = os.path.join(output_path, 'best_model.pth')
    data_path = os.path.join(output_path, 'processed_data.pkl')
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        print("   Please train the model first using: python train.py")
        return
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Processed data not found at {data_path}")
        print("   Please train the model first using: python train.py")
        return
    
    print(f"✓ Found model file: {model_path}")
    print(f"✓ Found data file: {data_path}")
    
    # Load processed data
    print("\nLoading processed data...")
    with open(data_path, 'rb') as f:
        processed_data = pickle.load(f)
    
    # Split data (same as training)
    from sklearn.model_selection import train_test_split
    
    X = processed_data['X']
    y_congestion = processed_data['y_congestion']
    y_rush_hour = processed_data['y_rush_hour']
    
    # Split: 70% train, 15% val, 15% test
    X_temp, X_test, y_cong_temp, y_cong_test, y_rush_temp, y_rush_test = train_test_split(
        X, y_congestion, y_rush_hour, test_size=0.15, random_state=42, stratify=y_congestion
    )
    
    print(f"Test set size: {len(X_test)} samples")
    
    # Load model
    print("\nLoading trained model...")
    checkpoint = torch.load(model_path)
    num_features = X_test.shape[1]
    model = SimpleMultiTaskGNN(num_features=num_features, hidden_dim=64)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    # Convert to tensors
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Make predictions
    print("\nMaking predictions...")
    with torch.no_grad():
        outputs = model(X_test_tensor)
        
        _, congestion_pred = torch.max(outputs['congestion_logits'].data, 1)
        _, rush_hour_pred = torch.max(outputs['rush_hour_logits'].data, 1)
        
        congestion_pred = congestion_pred.cpu().numpy()
        rush_hour_pred = rush_hour_pred.cpu().numpy()
    
    # Calculate accuracies
    congestion_acc = accuracy_score(y_cong_test, congestion_pred)
    rush_hour_acc = accuracy_score(y_rush_test, rush_hour_pred)
    
    print(f"\n✓ Congestion Accuracy: {congestion_acc:.4f}")
    print(f"✓ Rush Hour Accuracy: {rush_hour_acc:.4f}")
    
    # Print classification reports
    congestion_labels = ['Gridlock', 'Congested', 'Moderate', 'Free Flow']
    rush_hour_labels = ['Non-Rush Hour', 'Rush Hour']
    
    print("\n" + "="*60)
    print("Congestion Classification Report:")
    print("="*60)
    print(classification_report(y_cong_test, congestion_pred, 
                              target_names=congestion_labels, zero_division=0))
    
    print("\n" + "="*60)
    print("Rush Hour Classification Report:")
    print("="*60)
    print(classification_report(y_rush_test, rush_hour_pred, 
                              target_names=rush_hour_labels, zero_division=0))
    
    # Create confusion matrices
    print("\n" + "="*60)
    print("Generating Confusion Matrices...")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Congestion Confusion Matrix
    cm_congestion = confusion_matrix(y_cong_test, congestion_pred)
    sns.heatmap(cm_congestion, annot=True, fmt='d', cmap='Blues', 
                xticklabels=congestion_labels, yticklabels=congestion_labels,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Congestion Level - Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    
    # Rush Hour Confusion Matrix
    cm_rush_hour = confusion_matrix(y_rush_test, rush_hour_pred)
    sns.heatmap(cm_rush_hour, annot=True, fmt='d', cmap='Oranges',
                xticklabels=rush_hour_labels, yticklabels=rush_hour_labels,
                ax=axes[1], cbar_kws={'label': 'Count'})
    axes[1].set_title('Rush Hour - Confusion Matrix', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_path, 'confusion_matrices.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrices saved to: {output_file}")
    plt.close()
    
    print("\n" + "="*60)
    print("✅ Success! Confusion matrix generated successfully!")
    print("="*60)
    print(f"\nYou can find the image at: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    try:
        test_confusion_matrix()
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
