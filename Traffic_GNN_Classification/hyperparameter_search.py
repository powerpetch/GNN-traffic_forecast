"""
Hyperparameter Tuning Script
============================
Automatically find the best hyperparameters
"""

import os
import sys
import numpy as np
import pandas as pd
from itertools import product
import subprocess

def run_training_experiment(params, experiment_id):
    """Run a single training experiment with given parameters"""
    
    cmd = [
        'py', 'enhanced_train.py',
        '--epochs', str(params['epochs']),
        '--batch_size', str(params['batch_size']),
        '--hidden_dim', str(params['hidden_dim']),
        '--dropout', str(params['dropout']),
        '--lr', str(params['lr']),
        '--patience', '15'
    ]
    
    print(f"\n{'='*70}")
    print(f"Experiment {experiment_id}: {params}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        return True
    except Exception as e:
        print(f"Experiment {experiment_id} failed: {e}")
        return False

def hyperparameter_search():
    """Grid search for best hyperparameters"""
    
    # Define search space
    param_grid = {
        'epochs': [50, 100],
        'batch_size': [32, 64],
        'hidden_dim': [64, 128, 256],
        'dropout': [0.2, 0.3, 0.4],
        'lr': [0.001, 0.0005]
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    print(f"\n🔍 Starting Hyperparameter Search")
    print(f"Total combinations to try: {len(combinations)}")
    print(f"Estimated time: {len(combinations) * 10} minutes")
    print(f"{'='*70}\n")
    
    results = []
    
    for idx, params in enumerate(combinations, 1):
        success = run_training_experiment(params, idx)
        
        if success:
            # Try to load results
            try:
                import torch
                checkpoint = torch.load('outputs/best_enhanced_model.pth', map_location='cpu')
                
                result = {
                    **params,
                    'val_acc_congestion': checkpoint.get('val_acc_congestion', 0),
                    'val_acc_rush_hour': checkpoint.get('val_acc_rush_hour', 0),
                    'val_loss': checkpoint.get('val_loss', float('inf'))
                }
                results.append(result)
                
                print(f"\n✓ Experiment {idx} complete:")
                print(f"  Congestion Acc: {result['val_acc_congestion']:.2f}%")
                print(f"  Rush Hour Acc: {result['val_acc_rush_hour']:.2f}%")
                print(f"  Val Loss: {result['val_loss']:.4f}")
                
            except Exception as e:
                print(f"Could not load results for experiment {idx}: {e}")
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('val_acc_congestion', ascending=False)
        
        output_file = 'outputs/hyperparameter_search_results.csv'
        df.to_csv(output_file, index=False)
        
        print(f"\n{'='*70}")
        print("HYPERPARAMETER SEARCH COMPLETE")
        print(f"{'='*70}")
        print(f"\n📊 Top 5 Configurations:\n")
        print(df.head().to_string(index=False))
        print(f"\n✓ Full results saved to: {output_file}")
        
        # Best configuration
        best = df.iloc[0]
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"{'─'*70}")
        for param in ['epochs', 'batch_size', 'hidden_dim', 'dropout', 'lr']:
            print(f"  {param}: {best[param]}")
        print(f"{'─'*70}")
        print(f"  Congestion Accuracy: {best['val_acc_congestion']:.2f}%")
        print(f"  Rush Hour Accuracy: {best['val_acc_rush_hour']:.2f}%")
        print(f"  Validation Loss: {best['val_loss']:.4f}")
        print(f"{'─'*70}")

def quick_search():
    """Quick search with fewer combinations"""
    
    param_grid = {
        'epochs': [50],
        'batch_size': [32, 64],
        'hidden_dim': [64, 128],
        'dropout': [0.3],
        'lr': [0.001]
    }
    
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    print(f"\n🔍 Quick Hyperparameter Search")
    print(f"Combinations to try: {len(combinations)}")
    print(f"{'='*70}\n")
    
    results = []
    
    for idx, params in enumerate(combinations, 1):
        success = run_training_experiment(params, idx)
        
        if success:
            try:
                import torch
                checkpoint = torch.load('outputs/best_enhanced_model.pth', map_location='cpu')
                
                result = {
                    **params,
                    'val_acc_congestion': checkpoint.get('val_acc_congestion', 0),
                    'val_acc_rush_hour': checkpoint.get('val_acc_rush_hour', 0)
                }
                results.append(result)
                
            except:
                pass
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('val_acc_congestion', ascending=False)
        print(f"\n📊 Results:\n")
        print(df.to_string(index=False))
        
        best = df.iloc[0]
        print(f"\n🏆 Best: hidden_dim={int(best['hidden_dim'])}, batch_size={int(best['batch_size'])}")
        print(f"   Accuracy: {best['val_acc_congestion']:.2f}%")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        print("Running quick search...")
        quick_search()
    else:
        print("Running full hyperparameter search...")
        print("TIP: Use --quick for faster search")
        hyperparameter_search()
