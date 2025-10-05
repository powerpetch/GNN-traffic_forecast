"""
Configuration file for the GNN Traffic Dashboard
Contains constants, colors, styling, and shared settings
"""

import streamlit as st
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Page configuration
PAGE_CONFIG = {
    "page_title": "GNN Traffic Forecasting Dashboard",
    "page_icon": "🚦", 
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Professional color scheme
COLORS = {
    # Primary colors
    'primary_blue': '#3498db',
    'primary_green': '#27ae60',
    'primary_red': '#e74c3c',
    'primary_orange': '#f39c12',
    'primary_purple': '#8e44ad',
    
    # Traffic congestion colors
    'gridlock': '#e74c3c',      # Red
    'congested': '#f39c12',     # Orange  
    'moderate': '#f1c40f',      # Yellow
    'free_flow': '#27ae60',     # Green
    
    # Graph node colors
    'high_traffic': '#ff4757',
    'medium_traffic': '#ffa502', 
    'low_traffic': '#2ed573',
    'normal_traffic': '#5352ed',
    
    # Background colors
    'card_bg': '#ffffff',
    'sidebar_bg': '#f8f9fa',
    'main_bg': '#ffffff'
}

# CSS Styling
CSS_STYLES = """
<style>
    .main-header {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .gradient-bg {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        text-align: center;
        color: white;
    }
    
    .professional-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .status-card {
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid rgba(0,0,0,0.1);
    }
    
    .traffic-level-card {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border: 1px solid rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
    }
</style>
"""

# Traffic level labels and descriptions
TRAFFIC_LEVELS = {
    # Order MUST match: 0=Gridlock, 1=Congested, 2=Moderate, 3=Free-flow
    'labels': ['Gridlock', 'Congested', 'Moderate', 'Free-flow'],
    'colors': [COLORS['gridlock'], COLORS['congested'], COLORS['moderate'], COLORS['free_flow']],
    'descriptions': [
        'Severe congestion, minimal movement',
        'Heavy traffic, slower speeds',
        'Moderate traffic, reduced speeds',
        'Light traffic, normal speeds'
    ]
}

# Model architecture information
MODEL_ARCHITECTURES = {
    "Enhanced GNN": "**Best for beginners** - Balanced performance with dropout and batch normalization for stability",
    "Deep GNN": "**For complex patterns** - Multiple layers with residual connections, good for large datasets", 
    "Attention GNN": "**For temporal data** - Uses attention mechanism to focus on important time patterns",
    "Residual GNN": "**For deep networks** - Skip connections prevent vanishing gradients, very stable training"
}

# Default settings
DEFAULT_SETTINGS = {
    'map_style': 'OpenStreetMap',
    'show_confidence': True,
    'auto_update': False,
    'update_interval': 30,
    'zoom_level': 11,
    'center_lat': 13.7563,
    'center_lon': 100.5018
}

# Bangkok districts and areas
BANGKOK_DISTRICTS = [
    'Phra Nakhon', 'Dusit', 'Nong Chok', 'Bang Rak', 'Bang Kho Laem',
    'Phra Khanong', 'Pom Prap Sattru Phai', 'Thon Buri', 'Bangkok Yai',
    'Huai Khwang', 'Khlong Toei', 'Suan Luang', 'Chatuchak', 'Bang Sue',
    'Phaya Thai', 'Dinde Daeng', 'Samphanthawong', 'Pathum Wan', 'Bang Na',
    'Lak Si', 'Sai Mai', 'Khan Na Yao', 'Saphan Phut', 'Wang Thonglang',
    'Khlong Sam Wa', 'Nong Khaem', 'Rat Burana', 'Bang Phlat', 'Thawi Watthana',
    'Thung Khru', 'Bang Bon', 'Lat Krabang', 'Yan Nawa', 'Bang Kapi',
    'Phasi Charoen', 'Min Buri', 'Lat Phrao', 'Taling Chan', 'Bueng Kum',
    'Sathorn', 'Bang Rak', 'Silom', 'Sukhumvit', 'Ratchada', 'Ekkamai',
    'Thong Lo', 'Ari', 'Saphan Taksin', 'Chong Nonsi'
]

# Training configuration defaults
TRAINING_DEFAULTS = {
    'epochs': 75,
    'learning_rate': 0.001,
    'batch_size': 32,
    'hidden_dim': 64,
    'num_layers': 3,
    'dropout': 0.2,
    'patience': 10
}

# Chart configuration
CHART_CONFIG = {
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'font_family': 'Arial, sans-serif',
    'font_size': 12,
    'margin': dict(l=20, r=20, t=40, b=20)
}

def init_page_config():
    """Initialize Streamlit page configuration (call only once)"""
    if 'page_configured' not in st.session_state:
        st.set_page_config(**PAGE_CONFIG)
        st.session_state.page_configured = True

def apply_css():
    """Apply custom CSS styling"""
    st.markdown(CSS_STYLES, unsafe_allow_html=True)

def get_traffic_color(level):
    """Get color for traffic congestion level (0-3)"""
    return TRAFFIC_LEVELS['colors'][min(max(int(level), 0), 3)]

def get_traffic_label(level):
    """Get label for traffic congestion level (0-3)"""
    return TRAFFIC_LEVELS['labels'][min(max(int(level), 0), 3)]