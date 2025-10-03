"""
Configuration file for Multi-Task Traffic GNN Classification
============================================================
"""

import os

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Base data path
DATA_PATH = os.path.join("..", "Data")

# Road network data
ROAD_NETWORK_PATH = os.path.join(DATA_PATH, "hotosm_tha_roads_lines_gpkg")

# Probe data folders
PROBE_FOLDERS = [
    "PROBE-202401",
    "PROBE-202402", 
    "PROBE-202403",
    "PROBE-202404",
    "PROBE-202405",
    "PROBE-202406",
    "PROBE-202407",
    "PROBE-202408",
    "PROBE-202410",
    "PROBE-202412"
]

# Data processing parameters
AGGREGATION_MINUTES = 5  # Time window for aggregation
MIN_PROBES_PER_BIN = 3   # Minimum probes required per time bin
MAX_SPEED_THRESHOLD = 150  # Maximum reasonable speed (km/h)
MIN_SPEED_THRESHOLD = 0    # Minimum reasonable speed (km/h)

# Quality filtering
MIN_QUALITY_SCORE = 0.3
MAX_DISTANCE_TO_ROAD = 100  # meters

# ============================================================================
# MODEL CONFIGURATION  
# ============================================================================

# Model architecture
NUM_FEATURES = 10
HIDDEN_DIM = 64
NUM_CONGESTION_CLASSES = 4
NUM_RUSH_HOUR_CLASSES = 2

# Congestion classification thresholds (km/h)
CONGESTION_THRESHOLDS = {
    'gridlock': 10,     # < 10 km/h
    'congested': 30,    # 10-30 km/h  
    'moderate': 50,     # 30-50 km/h
    'free_flow': float('inf')  # > 50 km/h
}

# Rush hour time ranges (24-hour format)
RUSH_HOUR_RANGES = [
    (7, 9),   # Morning rush: 7:00-9:00
    (17, 19)  # Evening rush: 17:00-19:00
]

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Training parameters
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Loss weights for multi-task learning
CONGESTION_LOSS_WEIGHT = 1.0
RUSH_HOUR_LOSS_WEIGHT = 1.0

# Train/validation split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

# Early stopping
PATIENCE = 10
MIN_DELTA = 0.001

# ============================================================================
# GRAPH CONFIGURATION
# ============================================================================

# Graph construction parameters
MAX_EDGE_DISTANCE = 500  # meters - max distance for edge connections
MIN_EDGE_DISTANCE = 10   # meters - min distance for edge connections
SPATIAL_THRESHOLD = 0.001  # degrees - spatial resolution for adjacency

# Graph features
INCLUDE_EDGE_FEATURES = True
NORMALIZE_FEATURES = True

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Output paths
OUTPUT_DIR = "outputs"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
RESULTS_SAVE_PATH = os.path.join(OUTPUT_DIR, "evaluation_results.pkl")
HISTORY_SAVE_PATH = os.path.join(OUTPUT_DIR, "training_history.pkl")
DATA_SAVE_PATH = os.path.join(OUTPUT_DIR, "processed_data.pkl")

# Visualization settings
FIGURE_SIZE = (12, 8)
DPI = 300
SAVE_PLOTS = True

# ============================================================================
# DASHBOARD CONFIGURATION
# ============================================================================

# Dashboard settings
DASHBOARD_PORT = 8501
AUTO_REFRESH_SECONDS = 30
MAX_MAP_POINTS = 100

# Map settings
DEFAULT_MAP_CENTER = [13.7563, 100.5018]  # Bangkok coordinates
DEFAULT_ZOOM = 11
MAP_STYLE = 'OpenStreetMap'

# Color schemes
CONGESTION_COLORS = {
    'Gridlock': '#FF4444',
    'Congested': '#FF8800',
    'Moderate': '#FFAA00', 
    'Free Flow': '#44FF44'
}

RUSH_HOUR_COLORS = {
    'Non-Rush Hour': '#4ECDC4',
    'Rush Hour': '#FF6B6B'
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Logging settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(OUTPUT_DIR, 'traffic_gnn.log')

# Progress bar settings
SHOW_PROGRESS = True
PROGRESS_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'

# ============================================================================
# THAI LOCALIZATION (Optional)
# ============================================================================

# Thai language labels
THAI_LABELS = {
    'congestion': {
        'Gridlock': 'ติดขัดรุนแรง',
        'Congested': 'ติดขัด',
        'Moderate': 'ปานกลาง',
        'Free Flow': 'คล่องตัว'
    },
    'rush_hour': {
        'Rush Hour': 'ชั่วโมงเร่งด่วน',
        'Non-Rush Hour': 'เวลาปกติ'
    }
}

# Thai time format
THAI_TIME_FORMAT = '%H:%M น.'
THAI_DATE_FORMAT = '%d/%m/%Y'

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

# Performance settings
USE_GPU = True
NUM_WORKERS = 4
PIN_MEMORY = True

# Memory management
MAX_MEMORY_GB = 8
BATCH_SIZE_AUTO_ADJUST = True

# Random seed for reproducibility
RANDOM_SEED = 42

# Debugging
DEBUG_MODE = False
VERBOSE = True

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration parameters"""
    assert AGGREGATION_MINUTES > 0, "Aggregation minutes must be positive"
    assert MIN_PROBES_PER_BIN > 0, "Minimum probes per bin must be positive" 
    assert 0 < TRAIN_SPLIT < 1, "Train split must be between 0 and 1"
    assert TRAIN_SPLIT + VAL_SPLIT <= 1, "Train + val split cannot exceed 1"
    assert EPOCHS > 0, "Epochs must be positive"
    assert BATCH_SIZE > 0, "Batch size must be positive"
    assert LEARNING_RATE > 0, "Learning rate must be positive"
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("✅ Configuration validation passed!")

if __name__ == "__main__":
    validate_config()