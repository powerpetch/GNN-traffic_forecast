"""
Model Comparison Script
======================
Compare Simple vs Enhanced GNN models
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle

def load_models():
    """Load both simple and enhanced models"""
    output_path = "d:/user/Data_project/Traffic_GNN_Classification/outputs"
    
    models = {}
    
    # Load simple model
    simple_path = os.path.join(output_path, 'simple_multi_task_gnn.pth')
    if os.path.exists(simple_path):
        checkpoint = torch.load(simple_path, map_location='cpu')
        models['simple'] = {
            'checkpoint': checkpoint,
            'name': 'Simple GNN'
        }
        print("✓ Loaded Simple GNN model")
    
    # Load enhanced model
    enhanced_path = os.path.join(output_path, 'best_enhanced_model.pth')
    if os.path.exists(enhanced_path):
        checkpoint = torch.load(enhanced_path, map_location='cpu')
        models['enhanced'] = {
            'checkpoint': checkpoint,
            'name': 'Enhanced GNN'
        }
        print("✓ Loaded Enhanced GNN model")
    
    return models

def compare_training_histories():
    """Compare training histories"""
    output_path = "d:/user/Data_project/Traffic_GNN_Classification/outputs"
    
    # Load histories
    simple_hist_path = os.path.join(output_path, 'training_history.pkl')
    enhanced_hist_path = os.path.join(output_path, 'enhanced_training_history.pkl')
    
    histories = {}
    
    if os.path.exists(simple_hist_path):
        with open(simple_hist_path, 'rb') as f:
            histories['simple'] = pickle.load(f)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot comparisons
    metrics = [
        ('train_loss', 'Training Loss'),
        ('val_loss', 'Validation Loss'),
        ('val_acc_congestion', 'Congestion Accuracy (%)'),
        ('val_acc_rush_hour', 'Rush Hour Accuracy (%)')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        if 'simple' in histories and metric in histories['simple']:
            ax.plot(histories['simple'][metric], label='Simple GNN', linewidth=2, alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved: {output_path}/model_comparison.png")
    plt.close()

def print_model_comparison():
    """Print model statistics comparison"""
    models = load_models()
    
    if not models:
        print("No models found to compare!")
        return
    
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    
    data = []
    
    for model_type, model_info in models.items():
        checkpoint = model_info['checkpoint']
        
        row = {
            'Model': model_info['name'],
            'Val Loss': checkpoint.get('val_loss', 'N/A'),
            'Congestion Acc': f"{checkpoint.get('val_acc_congestion', 0):.2f}%",
            'Rush Hour Acc': f"{checkpoint.get('val_acc_rush_hour', 0):.2f}%",
            'Epoch': checkpoint.get('epoch', 'N/A')
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    print("\n" + df.to_string(index=False))
    print("\n" + "="*70)
    
    # Recommendations
    print("\n📊 RECOMMENDATIONS:")
    print("─"*70)
    
    if 'enhanced' in models and 'simple' in models:
        enh_cong = models['enhanced']['checkpoint'].get('val_acc_congestion', 0)
        sim_cong = models['simple']['checkpoint'].get('val_acc_congestion', 0)
        
        if enh_cong > sim_cong:
            improvement = enh_cong - sim_cong
            print(f"✓ Enhanced model performs {improvement:.2f}% better on congestion classification")
            print("  → Use enhanced model for production")
        else:
            print("✓ Simple model performs comparably with fewer parameters")
            print("  → Use simple model for faster inference")
    
    print("─"*70)

if __name__ == "__main__":
    print("\n🔍 Comparing GNN Models...")
    print("="*70)
    
    try:
        print_model_comparison()
        compare_training_histories()
        
        print("\n✅ Comparison complete!")
        
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
