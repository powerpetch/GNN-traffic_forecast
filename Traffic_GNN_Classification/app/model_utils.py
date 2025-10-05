"""
Utility functions for model performance tracking and cache management
"""

import streamlit as st
from typing import Dict, Any

def clear_model_cache(model_name: str = None):
    """
    Clear cached data for a specific model or all models
    
    Args:
        model_name: Name of model to clear cache for. If None, clear all.
    """
    if model_name:
        # Clear specific model cache
        keys_to_remove = [
            f'analytics_data_{model_name}',
            f'predictions_{model_name}',
            f'performance_{model_name}',
            f'traffic_map_{model_name}',
            f'network_viz_{model_name}'
        ]
    else:
        # Clear all model-related caches
        keys_to_remove = [k for k in st.session_state.keys() 
                         if any(k.startswith(prefix) for prefix in 
                               ['analytics_data_', 'predictions_', 'performance_', 
                                'traffic_map_', 'network_viz_'])]
    
    cleared = 0
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
            cleared += 1
    
    return cleared

def get_model_cache_status() -> Dict[str, Any]:
    """Get status of cached data for all models"""
    cache_status = {}
    
    for key in st.session_state.keys():
        if key.startswith('analytics_data_'):
            model_name = key.replace('analytics_data_', '')
            cache_status[model_name] = {
                'has_analytics': True,
                'has_predictions': f'predictions_{model_name}' in st.session_state,
                'has_performance': f'performance_{model_name}' in st.session_state
            }
    
    return cache_status

def register_trained_model(model_name: str, model_path: str, performance_metrics: Dict):
    """
    Register a newly trained model in session state
    
    Args:
        model_name: Name of the trained model
        model_path: Path where model is saved
        performance_metrics: Dict of performance metrics
    """
    if 'trained_models' not in st.session_state:
        st.session_state['trained_models'] = []
    
    if model_name not in st.session_state['trained_models']:
        st.session_state['trained_models'].append(model_name)
    
    # Store model metadata
    st.session_state[f'model_metadata_{model_name}'] = {
        'path': model_path,
        'metrics': performance_metrics,
        'timestamp': st.session_state.get('training_timestamp', 'Unknown')
    }
    
    # Clear old caches for this model
    clear_model_cache(model_name)

def get_active_model_info(selected_model: str) -> Dict[str, Any]:
    """Get information about the currently selected model"""
    metadata_key = f'model_metadata_{selected_model}'
    
    if metadata_key in st.session_state:
        return st.session_state[metadata_key]
    
    # Default info for built-in models
    default_models = {
        "Enhanced GNN": {
            "path": "outputs/enhanced_model.pth",
            "metrics": {
                "congestion_acc": 0.983,
                "rush_hour_acc": 0.986,
                "avg_accuracy": 0.9845
            },
            "description": "ST-GCN with Multi-Head Attention"
        },
        "Baseline Model": {
            "path": "outputs/baseline_model.pth",
            "metrics": {
                "congestion_acc": 0.698,
                "rush_hour_acc": 0.978,
                "avg_accuracy": 0.838
            },
            "description": "Simple MLP Baseline"
        },
        "Deep GNN": {
            "path": "outputs/deep_gnn.pth",
            "metrics": {
                "congestion_acc": 0.92,
                "rush_hour_acc": 0.982,
                "avg_accuracy": 0.951
            },
            "description": "Deep Graph Network"
        },
        "Attention GNN": {
            "path": "outputs/attention_gnn.pth",
            "metrics": {
                "congestion_acc": 0.95,
                "rush_hour_acc": 0.984,
                "avg_accuracy": 0.967
            },
            "description": "Multi-Head Attention GNN"
        }
    }
    
    return default_models.get(selected_model, {
        "path": "unknown",
        "metrics": {},
        "description": "Custom Model"
    })
