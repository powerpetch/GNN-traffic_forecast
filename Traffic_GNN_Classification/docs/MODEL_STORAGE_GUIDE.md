# 📦 Model Storage Guide - ตำแหน่งเก็บโมเดล

## 📂 โครงสร้าง Folder สำหรับโมเดล

```
D:\user\Data_project\Project_data\Traffic_GNN_Classification\
│
├── outputs/                                      # โฟลเดอร์หลักเก็บโมเดลทั้งหมด
│   │
│   ├── best_model.pth                           # ✨ โมเดลหลัก (Simple GNN)
│   ├── training_history.pkl                     # ประวัติการเทรน
│   ├── training_history.png                     # กราฟการเทรน
│   ├── confusion_matrices.png                   # Confusion Matrix
│   ├── evaluation_results.pkl                   # ผล evaluation
│   ├── processed_data.pkl                       # ข้อมูลที่ประมวลผลแล้ว
│   │
│   ├── enhanced_training/                       # โมเดล Enhanced GNN
│   │   ├── enhanced_model.pth                   # ✨ โมเดล Enhanced
│   │   ├── best_model.pth                       # โมเดลดีที่สุด
│   │   ├── training_history.pkl                 # ประวัติการเทรน
│   │   ├── training_results.png                 # กราฟผลลัพธ์
│   │   └── test_results.pkl                     # ผลการทดสอบ
│   │
│   ├── optimized_training/                      # โมเดล Optimized
│   │   ├── optimized_model.pth                  # ✨ โมเดล Optimized
│   │   ├── best_model.pth                       # โมเดลดีที่สุด
│   │   └── model_config.pkl                     # Configuration
│   │
│   └── quick_training/                          # โมเดล Quick Training
│       ├── quick_model.pth                      # ✨ โมเดล Quick
│       ├── model.pth                            # โมเดลสำรอง
│       └── config.pkl                           # Configuration
│
└── models/                                       # (ถ้ามี) โมเดลเก่าอื่นๆ
```

---

## 📋 รายละเอียดโมเดลแต่ละประเภท

### 1. **Simple GNN (Base Model)**
- **ตำแหน่ง:** `outputs/best_model.pth`
- **สถาปัตยกรรม:** SimpleMultiTaskGNN
- **Parameters:** ~5,254 parameters
- **ใช้สำหรับ:** 
  - Training พื้นฐาน
  - Testing เบื้องต้น
  - Baseline comparison
- **รันด้วย:** `python train.py`

### 2. **Enhanced GNN**
- **ตำแหน่ง:** `outputs/enhanced_training/enhanced_model.pth`
- **สถาปัตยกรรม:** EnhancedGNNModel (ST-GCN + Attention)
- **Parameters:** ~122,918 parameters
- **ใช้สำหรับ:**
  - Training ขั้นสูง
  - Spatio-temporal analysis
  - Production deployment
- **รันด้วย:** `python enhanced_train.py`

### 3. **Optimized GNN**
- **ตำแหน่ง:** `outputs/optimized_training/optimized_model.pth`
- **สถาปัตยกรรม:** Optimized architecture
- **ใช้สำหรับ:**
  - Hyperparameter tuning
  - Performance optimization
- **รันด้วย:** `python enhanced_train.py --optimize`

### 4. **Quick Training GNN**
- **ตำแหน่ง:** `outputs/quick_training/quick_model.pth`
- **สถาปัตยกรรม:** Quick training configuration
- **ใช้สำหรับ:**
  - Rapid prototyping
  - Quick testing
- **รันด้วย:** `python enhanced_train.py --quick`

---

## 🔍 วิธีตรวจสอบโมเดลที่มี

### วิธีที่ 1: ใช้ Dashboard (แนะนำ)
```powershell
streamlit run app/dashboard.py
```
- ไปที่ Tab "Training"
- มองหาส่วน **"Available Pre-trained Models"**
- จะแสดงโมเดลที่มีจริงใน folder `outputs/`

### วิธีที่ 2: ใช้ Python Script
```python
import os
from pathlib import Path

# ตรวจสอบโมเดลที่มี
model_paths = [
    "outputs/best_model.pth",
    "outputs/enhanced_training/enhanced_model.pth",
    "outputs/optimized_training/optimized_model.pth",
    "outputs/quick_training/quick_model.pth",
]

for path in model_paths:
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)  # MB
        print(f"✅ Found: {path} ({size:.1f} MB)")
    else:
        print(f"❌ Missing: {path}")
```

### วิธีที่ 3: ใช้ Command Line
```powershell
# Windows PowerShell
Get-ChildItem -Path outputs -Recurse -Filter *.pth | Select-Object FullName, Length
```

---

## 📊 ไฟล์อื่นๆ ที่เกี่ยวข้อง

### 1. Training History
- **ไฟล์:** `training_history.pkl`
- **เก็บข้อมูล:**
  - Train/Validation Loss
  - Accuracy ทั้ง 2 tasks
  - Learning rates
  - Best epoch

### 2. Evaluation Results
- **ไฟล์:** `evaluation_results.pkl`
- **เก็บข้อมูล:**
  - Predictions vs Actual
  - Accuracy scores
  - Classification metrics

### 3. Visualization Files
- **Confusion Matrix:** `confusion_matrices.png`
- **Training Curves:** `training_history.png`
- **Test Results:** `training_results.png`

### 4. Processed Data
- **ไฟล์:** `processed_data.pkl`
- **เก็บข้อมูล:**
  - Features (27 features)
  - Labels (Congestion + Rush Hour)
  - Preprocessed data ready for training

---

## 🎯 วิธีโหลดโมเดลใน Dashboard

### ขั้นตอน:
1. เปิด Dashboard: `streamlit run app/dashboard.py`
2. ไปที่ Tab **"Training"**
3. Scroll ลงไปส่วน **"Available Pre-trained Models"**
4. คลิก Expand โมเดลที่ต้องการ
5. คลิกปุ่ม **"Load Model"**
6. ไปที่ **Sidebar** (ซ้ายมือ)
7. เลือกโมเดลจาก dropdown **"Select Model"**

### หมายเหตุ:
- โมเดลจะแสดงเฉพาะที่มีไฟล์ `.pth` ใน folder
- ถ้าไม่เห็นโมเดล แสดงว่ายังไม่ได้เทรน
- ชื่อโมเดลจะแสดง accuracy และขนาดไฟล์

---

## ⚙️ Configuration Files

### model_config.pkl / config.pkl
เก็บ configuration ของโมเดล:
```python
{
    'architecture': 'EnhancedGNN',
    'hidden_dim': 64,
    'num_features': 27,
    'num_classes_congestion': 4,
    'num_classes_rush_hour': 2,
    'learning_rate': 0.001,
    'batch_size': 32,
    ...
}
```

---

## 🛠️ วิธีสร้างโมเดลใหม่

### 1. Simple GNN
```powershell
python train.py
# Output: outputs/best_model.pth
```

### 2. Enhanced GNN
```powershell
python enhanced_train.py
# Output: outputs/enhanced_training/enhanced_model.pth
```

### 3. Quick Training
```powershell
python enhanced_train.py --quick
# Output: outputs/quick_training/quick_model.pth
```

### 4. Optimized Training
```powershell
python enhanced_train.py --optimize
# Output: outputs/optimized_training/optimized_model.pth
```

---

## 🔄 Migration Path (ย้ายโมเดล)

ถ้าโมเดลอยู่ใน location เก่า:
```
D:\user\Data_project\Traffic_GNN_Classification\outputs\
```

ย้ายมาที่:
```
D:\user\Data_project\Project_data\Traffic_GNN_Classification\outputs\
```

### วิธีย้าย:
```powershell
# Copy ทั้ง folder
Copy-Item -Path "D:\user\Data_project\Traffic_GNN_Classification\outputs\*" `
          -Destination "D:\user\Data_project\Project_data\Traffic_GNN_Classification\outputs\" `
          -Recurse -Force
```

หรือใช้ไฟล์ `.bat` ที่สร้างไว้:
```powershell
MOVE_FILES.bat
```

---

## ✅ Checklist การตรวจสอบโมเดล

- [ ] มีไฟล์ `.pth` ใน `outputs/`
- [ ] มีไฟล์ `training_history.pkl`
- [ ] มีไฟล์ `confusion_matrices.png`
- [ ] Dashboard แสดงโมเดลใน "Available Pre-trained Models"
- [ ] สามารถโหลดโมเดลใน Sidebar ได้
- [ ] โมเดลแสดง accuracy และขนาดไฟล์

---

## 🆘 Troubleshooting

### ปัญหา: ไม่เห็นโมเดลใน Dashboard
**แก้ไข:**
1. ตรวจสอบว่ามีไฟล์ `.pth` ใน `outputs/`
2. Restart Dashboard: `Ctrl+C` แล้ว `streamlit run app/dashboard.py`
3. Clear cache: กด "C" ใน Dashboard
4. ตรวจสอบ path ใน `train.py` และ `enhanced_train.py`

### ปัญหา: โมเดลโหลดไม่ได้
**แก้ไข:**
1. ตรวจสอบว่าไฟล์ไม่ corrupted
2. ตรวจสอบ architecture ตรงกับโมเดลหรือไม่
3. ลองเทรนโมเดลใหม่

### ปัญหา: Path ผิด
**แก้ไข:**
1. แก้ไข `train.py` line 579
2. แก้ไข `enhanced_train.py` ถ้ามี
3. ใช้ absolute path แทน relative path

---

**อัปเดตล่าสุด:** 2025-01-06
**เวอร์ชัน:** 1.0
