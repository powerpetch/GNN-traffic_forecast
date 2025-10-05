"""
Shared Model Management Functions
Used across dashboard and training tabs
"""
import os
import pickle
from pathlib import Path
from datetime import datetime

def scan_available_models():
    """
    Scan outputs folder for available pre-trained models
    Returns list of model info dictionaries
    
    Used by:
    - dashboard.py: Sidebar model selection
    - tab_training.py: Available Pre-trained Models section
    """
    models = []
    
    # Get absolute path to project root
    # This works from both app/ and project root
    if Path(__file__).parent.name == "app":
        project_root = Path(__file__).parent.parent
    else:
        project_root = Path(__file__).parent
    
    base_path = project_root / "outputs"
    
    if not base_path.exists():
        return models
    
    # Define model locations to scan
    model_locations = [
        (base_path / "best_model.pth", "Simple GNN (Base)", base_path / "training_history.pkl"),
        (base_path / "enhanced_training" / "enhanced_model.pth", "Enhanced GNN", base_path / "enhanced_training" / "training_history.pkl"),
        (base_path / "optimized_training" / "optimized_model.pth", "Optimized GNN", base_path / "optimized_training" / "model_config.pkl"),
        (base_path / "quick_training" / "quick_model.pth", "Quick Training GNN", base_path / "quick_training" / "config.pkl"),
    ]
    
    for model_path, model_name, history_path in model_locations:
        if model_path.exists():
            # Get file info
            file_stat = model_path.stat()
            file_size_mb = file_stat.st_size / (1024 * 1024)  # Convert to MB
            modified_time = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d")
            
            # Try to load training history for accuracy
            accuracy = "N/A"
            try:
                if history_path.exists():
                    with open(history_path, 'rb') as f:
                        history = pickle.load(f)
                        if 'val_acc_congestion' in history:
                            best_acc = max(history['val_acc_congestion'])
                            accuracy = f"{best_acc:.1f}%"
                        elif 'congestion_accuracy' in history:
                            accuracy = f"{history['congestion_accuracy']*100:.1f}%"
            except:
                pass
            
            models.append({
                "name": model_name,
                "path": str(model_path),
                "accuracy": accuracy,
                "size": f"{file_size_mb:.1f}MB",
                "date": modified_time
            })
    
    return models

def get_model_list_for_selector():
    """
    Get list of model names for dropdown selector
    Combines trained models with default models
    
    Used by: dashboard.py sidebar
    """
    models = []
    
    # Get trained models
    trained_models = scan_available_models()
    for model in trained_models:
        models.append(model['name'])
    
    # Add default models for compatibility
    default_models = ["Baseline Model", "Deep GNN", "Attention GNN"]
    for default in default_models:
        if default not in models:
            models.append(default)
    
    # If no models found, return defaults
    if not models:
        return ["Enhanced GNN", "Baseline Model", "Deep GNN", "Attention GNN"]
    
    return models
