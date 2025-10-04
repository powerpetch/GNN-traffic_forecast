# 🚀 Quick Start Guide - Traffic GNN Classification

## ✅ Installation Fixed!

The batch size mismatch error has been **FIXED**. The training should now work correctly.

---

## 🎯 Three Ways to Run the Program

### 1️⃣ **EASIEST: Double-Click to Start Dashboard** ⭐ RECOMMENDED

Simply double-click: **`START_DASHBOARD.bat`**

- Dashboard opens at: http://localhost:8501
- No command line needed!
- Press `Ctrl+C` in the terminal to stop

---

### 2️⃣ **Run Dashboard from Command Line**

```powershell
cd d:\user\Data_project\Project_data\Traffic_GNN_Classification
py -m streamlit run app/dashboard.py
```

**Alternative dashboard:**
```powershell
py -m streamlit run app/dashboard_clean.py
```

---

### 3️⃣ **Train New Models**

**Option A: Double-Click**
- Double-click: **`RUN_TRAINING.bat`**

**Option B: Command Line**
```powershell
cd d:\user\Data_project\Project_data\Traffic_GNN_Classification
py train.py --epochs 100 --batch_size 32
```

**Quick training (for testing):**
```powershell
py train.py --epochs 10
```

---

## 📊 What You'll See in the Dashboard

### **5 Interactive Tabs:**

1. **📊 Overview**
   - Project information
   - System status
   - Quick stats

2. **🗺️ Live Traffic Map**
   - Real-time Bangkok traffic visualization
   - Color-coded congestion levels
   - Interactive markers

3. **🚗 Route Optimizer**
   - Smart route planning
   - 217 Bangkok locations
   - Dynamic distance/time/cost calculations
   - Haversine distance formula

4. **🧠 GNN Architecture**
   - Neural network visualization
   - Model structure diagram
   - Layer connections

5. **📈 Model Analytics**
   - Performance metrics
   - Accuracy gauges
   - Prediction confidence
   - Model-specific analytics

---

## 🔧 What Was Fixed

### **Problem:** 
```
ValueError: Expected input batch_size (1) to match target batch_size (32)
```

### **Solution:**
Modified `SimpleMultiTaskGNN.forward()` method in `/src/models/multi_task_gnn.py`:
- Now correctly preserves batch dimension
- Removed incorrect mean aggregation
- Handles batched inputs properly

### **Additional Fix:**
- Changed `plt.show()` to `plt.close()` in training script
- Plots are now saved to files instead of blocking execution
- Training history saved to: `outputs/training_history.png`

---

## 📁 Output Files

After training, check the `outputs/` folder for:

```
outputs/
├── processed_data.pkl          # Processed traffic data
├── simple_multi_task_gnn.pth   # Trained model weights
├── training_history.png        # Training curves
├── training_history.pkl        # Training metrics
└── confusion_matrices.png      # Model performance visualization
```

---

## 💡 Pro Tips

### **On Your System, Always Use:**
```powershell
py -m pip install <package>    # NOT "pip install"
py train.py                     # NOT "python train.py"
py -m streamlit run app.py      # NOT "streamlit run"
```

### **Quick Commands:**
```powershell
# Install missing packages
py -m pip install -r requirements.txt

# Check Python version
py --version

# Upgrade a package
py -m pip install --upgrade streamlit

# List installed packages
py -m pip list
```

---

## 🐛 Troubleshooting

### **Dashboard won't start:**
```powershell
py -m pip install --upgrade streamlit streamlit-folium
```

### **Import errors:**
```powershell
py -m pip install torch torch-geometric geopandas folium
```

### **Training takes too long:**
```powershell
# Use fewer epochs for testing
py train.py --epochs 10
```

### **Out of memory:**
```powershell
# Use smaller batch size
py train.py --batch_size 16
```

---

## 🎓 Model Training Details

### **Default Parameters:**
- Epochs: 100
- Batch Size: 32
- Learning Rate: 0.001
- Hidden Dimension: 64

### **Expected Performance:**
- Congestion Classification: ~85-90% accuracy
- Rush Hour Detection: ~90-95% accuracy
- Training Time: 10-30 minutes (CPU)

### **Model Architecture:**
```
SimpleMultiTaskGNN
├── Feature Transform (10 → 64 → 64)
├── Congestion Head (64 → 4 classes)
└── Rush Hour Head (64 → 2 classes)
```

---

## 🌟 Features

✅ Multi-task learning (congestion + rush hour)
✅ Real-time traffic prediction
✅ Interactive Bangkok map
✅ Smart route optimization  
✅ 217 Bangkok locations
✅ Dynamic calculations
✅ Model performance analytics
✅ Professional UI (no emojis in headers)

---

## 📞 Support

If you encounter any issues:
1. Check this guide first
2. Verify all packages are installed: `py -m pip list`
3. Try running with `--force_reprocess` flag to regenerate data
4. Check the `outputs/` folder for error logs

---

## ✨ Quick Start Checklist

- [ ] All packages installed: `py -m pip install -r requirements.txt`
- [ ] Data files in place: `../Data/PROBE-202401/`, etc.
- [ ] Double-click `START_DASHBOARD.bat` OR
- [ ] Run: `py -m streamlit run app/dashboard.py`
- [ ] Open browser to: http://localhost:8501
- [ ] Enjoy your Traffic GNN Dashboard! 🎉

---

**Last Updated:** October 2025  
**Status:** ✅ All bugs fixed - Ready to run!
