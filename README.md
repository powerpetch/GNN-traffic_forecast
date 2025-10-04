# 🚦 Multi-Task Traffic GNN Classification

## Thai Traffic Analysis using Graph Neural Networks
**ระบบวิเคราะห์การจราจรด้วย Graph Neural Networks สำหรับการจำแนกความหนาแน่นของการจราจรและการตระหนักถึงชั่วโมงเร่งด่วน**

A comprehensive Graph Neural Network system for traffic prediction and smart navigation using Bangkok traffic data.


---

## 📋 Project Overview

This project implements a **Multi-Task Graph Neural Network (GNN)** system for simultaneous prediction of:

1. **Traffic Congestion Classification** (4 classes):
   - 🔴 **Gridlock** (< 10 km/h) - การจราจรติดขัดอย่างรุนแรง
   - 🟠 **Congested** (10-30 km/h) - การจราจรติดขัด
   - 🟡 **Moderate** (30-50 km/h) - การจราจรปานกลาง
   - 🟢 **Free Flow** (> 50 km/h) - การจราจรคล่องตัว

2. **Rush Hour Detection** (Binary classification):
   - ⏰ **Rush Hour** - ชั่วโมงเร่งด่วน
   - 🏙️ **Non-Rush Hour** - เวลาปกติ

### 🎯 Key Features

- **Spatio-Temporal Graph Convolutional Networks (ST-GCN)** for traffic pattern analysis
- **Real-time prediction** using GPS probe data
- **Interactive dashboard** with live traffic visualization
- **Thai road network integration** using OpenStreetMap data
- **Multi-task learning** for efficient dual classification
- **Map-matching algorithms** for GPS-to-road alignment

---

## 🏗️ Project Structure

```
Traffic_GNN_Classification/
├── 📁 src/
│   ├── 📁 data/
│   │   └── 📄 data_processor.py      # Data processing & map-matching
│   ├── 📁 models/
│   │   └── 📄 multi_task_gnn.py      # ST-GCN model architecture
│   ├── 📁 utils/
│   │   └── 📄 graph_constructor.py   # Graph building utilities
│   └── 📁 config/
│       └── 📄 __init__.py
├── 📁 app/
│   └── 📄 dashboard.py               # Streamlit dashboard
├── 📁 outputs/                      # Model outputs & results
├── 📄 train.py                      # Training pipeline
├── 📄 requirements.txt              # Dependencies
└── 📄 README.md                     # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd Traffic_GNN_Classification

# Install dependencies
pip install -r requirements.txt

# Alternative: Install with conda
conda create -n traffic_gnn python=3.9
conda activate traffic_gnn
pip install -r requirements.txt
```

### 2. Data Preparation

Place your data in the following structure:
```
../Data/
├── PROBE-202401/           # GPS probe data (January 2024)
├── PROBE-202402/           # GPS probe data (February 2024)
└── hotosm_tha_roads_lines_gpkg/  # Thai road network
```

### 3. Model Training

```bash
# Train the multi-task GNN model
python train.py

# Training with custom parameters
python train.py --epochs 100 --batch_size 64 --learning_rate 0.001
```

### 4. Launch Dashboard

```bash
# Run the interactive dashboard
streamlit run app/dashboard.py

# Dashboard will be available at: http://localhost:8501
```

---

## 📊 Model Architecture

### ST-GCN (Spatio-Temporal Graph Convolutional Network)

```python
Multi-Task Traffic GNN
├── Input Layer (10 features)
├── ST-GCN Block 1 (64 hidden units)
├── ST-GCN Block 2 (32 hidden units)
├── Dual Classification Heads:
│   ├── Congestion Head (4 classes)
│   └── Rush Hour Head (2 classes)
```

### Features Used:
- **Speed Statistics**: Mean, median, standard deviation
- **Temporal Features**: Hour sine/cosine, day-of-week encoding
- **Spatial Features**: GPS coordinates, road network topology
- **Quality Metrics**: Probe count, data quality scores

---

## 🗺️ Dashboard Features

### 1. Live Traffic Map
- **Real-time traffic visualization** on Bangkok road network
- **Color-coded congestion levels** with interactive markers
- **Confidence scores** for each prediction
- **Auto-refresh capability** for live monitoring

### 2. Analytics Dashboard
- **24-hour traffic pattern analysis**
- **Speed vs congestion correlation plots**
- **Rush hour probability by time**
- **Detailed prediction tables**

### 3. Model Performance
- **Accuracy metrics** for both classification tasks
- **Confusion matrices** visualization
- **Performance gauges** and statistics

### 4. Training Analysis
- **Training history visualization**
- **Loss curves** and accuracy trends
- **Model convergence analysis**

---

## 🔬 Technical Details

### Data Processing Pipeline

1. **GPS Probe Loading**: Load raw GPS trajectories from CSV files
2. **Map Matching**: Align GPS points to road network using spatial indexing
3. **Temporal Aggregation**: Group data into 5-minute intervals
4. **Feature Engineering**: Create temporal and spatial features
5. **Label Creation**: Generate congestion and rush hour labels

### Graph Construction

1. **Road Network Graph**: Build from OpenStreetMap GPKG data
2. **Spatial Adjacency**: Connect nearby road segments
3. **Feature Assignment**: Assign traffic features to graph nodes
4. **PyTorch Geometric**: Convert to PyG Data format

### Training Process

1. **Multi-Task Loss**: Combined loss for both classification tasks
2. **Weighted Loss**: Balance between congestion and rush hour tasks
3. **Adam Optimizer**: With learning rate scheduling
4. **Early Stopping**: Prevent overfitting with validation monitoring

---

## 📈 Performance Metrics

### Expected Performance:
- **Congestion Classification**: ~85-90% accuracy
- **Rush Hour Detection**: ~90-95% accuracy
- **Training Time**: ~10-20 minutes on GPU
- **Inference Speed**: Real-time predictions

### Evaluation Metrics:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance
- **F1-Score**: Balanced metric for each task
- **Confusion Matrix**: Detailed classification analysis

---

## 🔧 Configuration Options

### Training Parameters:
```python
--epochs 50              # Number of training epochs
--batch_size 32          # Batch size for training
--learning_rate 0.001    # Learning rate
--hidden_dim 64          # Hidden layer dimensions
--weight_decay 1e-4      # L2 regularization
--congestion_weight 1.0  # Loss weight for congestion task
--rush_hour_weight 1.0   # Loss weight for rush hour task
```

### Data Parameters:
```python
DATA_PATH = "../Data"           # Path to data directory
ROAD_NETWORK_PATH = "hotosm_tha_roads_lines_gpkg"
PROBE_FOLDERS = ["PROBE-202401", "PROBE-202402"]
AGGREGATION_MINUTES = 5         # Time aggregation window
MIN_PROBES_PER_BIN = 3         # Minimum probes for valid data
```

---

## 🌐 Thai Language Support

The system includes Thai language annotations and is designed for Thai traffic patterns:

- **Thai road network** from OpenStreetMap Thailand
- **Bangkok-specific** traffic patterns and rush hours
- **Thai language labels** in dashboard (optional)
- **Local time zones** and calendar considerations

---

## 🚧 Troubleshooting

### Common Issues:

1. **CUDA/GPU Issues**:
   ```bash
   # Install CPU-only PyTorch if no GPU
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Geospatial Dependencies**:
   ```bash
   # Install GDAL for geopandas
   conda install -c conda-forge gdal
   ```

3. **Memory Issues**:
   - Reduce batch size in training
   - Use data sampling for large datasets
   - Monitor memory usage during processing

4. **Dashboard Issues**:
   ```bash
   # Clear Streamlit cache
   streamlit cache clear
   ```

---

## 📝 Citation & References

```bibtex
@software{traffic_gnn_2024,
  title={Multi-Task Traffic GNN Classification},
  author={GitHub Copilot},
  year={2024},
  description={Spatio-Temporal Graph Neural Networks for Traffic Analysis},
  keywords={GNN, Traffic Analysis, Multi-Task Learning, Bangkok, Thailand}
}
```

### References:
- **ST-GCN**: Spatio-Temporal Graph Convolutional Networks
- **PyTorch Geometric**: Graph neural network library
- **OpenStreetMap**: Road network data source
- **Streamlit**: Dashboard framework

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Advanced GNN Architectures**: GraphSAGE, GAT, etc.
2. **Real-time Data Integration**: Live traffic feeds
3. **Additional Features**: Weather, events, accidents
4. **Mobile Application**: React Native dashboard
5. **Thai Language**: Complete localization

---

## 📄 License

This project is open source and available under the MIT License.

---

## 📞 Support

For questions, issues, or contributions:

1. **GitHub Issues**: Report bugs and feature requests
2. **Documentation**: Check this README and code comments
3. **Community**: Join discussions and share improvements

---

**Built with ❤️ using Graph Neural Networks for Thai Traffic Analysis**

*ระบบวิเคราะห์การจราจรไทยด้วยเทคโนโลยี Graph Neural Networks*
