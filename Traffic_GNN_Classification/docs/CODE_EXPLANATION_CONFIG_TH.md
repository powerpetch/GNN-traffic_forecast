# 📘 อธิบายโค้ด: config.py

## 📋 ข้อมูลไฟล์

- **ชื่อไฟล์:** `src/config/config.py`
- **หน้าที่:** ศูนย์กลางการตั้งค่าทั้งหมดของโปรเจค
- **จำนวนบรรทัด:** ~220 บรรทัด
- **ภาษา:** Python
- **พารามิเตอร์:** 60+ ตัว

---

## 🎯 ภาพรวม

ไฟล์ `config.py` เป็น **Configuration Center** ที่เก็บค่าพารามิเตอร์ทั้งหมดของโปรเจคไว้ที่เดียว เพื่อให้สามารถแก้ไขและจัดการได้ง่าย

### **ประโยชน์:**
✅ แก้ค่าที่เดียว → ทุกไฟล์อัปเดต  
✅ ไม่ต้อง hard-code ค่าในหลายที่  
✅ ทดลองการตั้งค่าต่างๆ ได้ง่าย  
✅ ป้องกันความผิดพลาดจากการใช้ค่าผิด

---

## 📂 โครงสร้างไฟล์

```python
config.py
├── 1. DATA CONFIGURATION (บรรทัด 11-44)
├── 2. MODEL CONFIGURATION (บรรทัด 46-68)
├── 3. TRAINING CONFIGURATION (บรรทัด 70-94)
├── 4. GRAPH CONFIGURATION (บรรทัด 96-108)
├── 5. OUTPUT CONFIGURATION (บรรทัด 110-126)
├── 6. DASHBOARD CONFIGURATION (บรรทัด 128-156)
├── 7. LOGGING CONFIGURATION (บรรทัด 158-168)
├── 8. THAI LOCALIZATION (บรรทัด 170-193)
├── 9. SYSTEM CONFIGURATION (บรรทัด 195-213)
└── 10. VALIDATION (บรรทัด 215-219)
```

---

## 1️⃣ DATA CONFIGURATION

### **📍 Path Configuration**

```python
DATA_PATH = os.path.join("..", "Data")
```

**อธิบาย:**
- กำหนด path ไปยังโฟลเดอร์ข้อมูลหลัก
- `".."` = ย้อนกลับ 1 โฟลเดอร์
- ผลลัพธ์: `../Data/`

**ตัวอย่างการใช้งาน:**
```python
from config import DATA_PATH
probe_path = os.path.join(DATA_PATH, "PROBE-202401")
# ได้: ../Data/PROBE-202401
```

---

```python
ROAD_NETWORK_PATH = os.path.join(DATA_PATH, "hotosm_tha_roads_lines_gpkg")
```

**อธิบาย:**
- เก็บตำแหน่งข้อมูลแผนที่ถนน
- ข้อมูลจาก OpenStreetMap (OSM)
- รูปแบบไฟล์: `.gpkg` (GeoPackage)

**ข้อมูลในไฟล์:**
- โครงข่ายถนนกรุงเทพฯ
- พิกัด (latitude, longitude)
- ประเภทถนน (highway, local road, etc.)
- ความยาวถนน

---

```python
PROBE_FOLDERS = [
    "PROBE-202401",
    "PROBE-202402",
    "PROBE-202403",
    ...
]
```

**อธิบาย:**
- รายชื่อโฟลเดอร์ที่เก็บข้อมูล GPS probe
- แต่ละโฟลเดอร์ = 1 เดือน
- ข้อมูลจาก: รถที่วิ่งบนถนนจริง

**โครงสร้างข้อมูล PROBE:**
```csv
timestamp,location_id,latitude,longitude,speed,heading,quality
2024-01-01 00:05:00,LOC001,13.7563,100.5018,45.5,90.0,0.85
```

**Fields อธิบาย:**
- `timestamp`: วันเวลาที่บันทึก
- `location_id`: รหัสสถานที่
- `latitude, longitude`: พิกัด GPS
- `speed`: ความเร็ว (km/h)
- `heading`: ทิศทาง (0-360°)
- `quality`: คะแนนคุณภาพ (0-1)

---

### **⚙️ Data Processing Parameters**

```python
AGGREGATION_MINUTES = 5
```

**อธิบาย:**
- รวมข้อมูลทุก **5 นาที**
- แปลงข้อมูล real-time → time series

**ตัวอย่าง:**
```python
# ข้อมูลดิบ (ทุกวินาที)
08:00:30 - 45 km/h
08:01:15 - 47 km/h
08:02:50 - 43 km/h
08:03:20 - 46 km/h
08:04:10 - 44 km/h

# หลัง Aggregation (ทุก 5 นาที)
08:00-08:05 - mean: 45.0 km/h, std: 1.58
```

**ทำไมต้อง aggregate?**
- ลด noise จากข้อมูล
- ลดขนาดข้อมูล (ประหยัด memory)
- ทำให้แนวโน้มชัดเจนขึ้น

---

```python
MIN_PROBES_PER_BIN = 3
```

**อธิบาย:**
- ต้องมีข้อมูลอย่างน้อย **3 รถ** ต่อช่วงเวลา
- ถ้าน้อยกว่า → ทิ้งข้อมูลนั้น

**เหตุผล:**
- 1-2 รถ = ไม่แทนสภาพจริง
- อาจเป็นรถแข่ง หรือ รถเต่า
- ข้อมูลไม่น่าเชื่อถือ

**ตัวอย่าง:**
```python
# ช่วง 08:00-08:05
locations = {
    'LOC001': 5 รถ  ✅ ใช้ได้
    'LOC002': 3 รถ  ✅ ใช้ได้ (ขั้นต่ำ)
    'LOC003': 2 รถ  ❌ ทิ้ง (น้อยเกินไป)
}
```

---

```python
MAX_SPEED_THRESHOLD = 150  # km/h
MIN_SPEED_THRESHOLD = 0
```

**อธิบาย:**
- กรองความเร็วที่ไม่สมเหตุสมผล
- ช่วง: **0-150 km/h**

**ตรรกะการกรอง:**
```python
def filter_speed(speed):
    if speed < MIN_SPEED_THRESHOLD:  # < 0
        return None  # ไม่มีความเร็วติดลบ
    elif speed > MAX_SPEED_THRESHOLD:  # > 150
        return None  # เกินความเป็นจริง
    else:
        return speed  # ✅ ปกติ
```

**ตัวอย่างข้อมูล:**
```python
speeds = [45, -10, 120, 250, 0, 80]

# หลังกรอง
speeds_filtered = [45, 120, 0, 80]
# ทิ้ง: -10 (ติดลบ), 250 (เกินไป)
```

---

```python
MIN_QUALITY_SCORE = 0.3
MAX_DISTANCE_TO_ROAD = 100  # meters
```

**อธิบาย:**

**1. MIN_QUALITY_SCORE:**
- คะแนนคุณภาพข้อมูล ≥ 0.3
- Quality score มาจาก GPS accuracy

**2. MAX_DISTANCE_TO_ROAD:**
- ข้อมูลต้องห่างจากถนนไม่เกิน 100 เมตร
- กรอง GPS drift (สัญญาณเบี่ยงเบน)

**ตัวอย่าง:**
```python
# ข้อมูล GPS
point1 = {
    'lat': 13.7563, 
    'lon': 100.5018,
    'quality': 0.85,
    'distance_to_road': 15  # เมตร
}
# ✅ ผ่าน: quality ≥ 0.3, distance ≤ 100

point2 = {
    'lat': 13.9999,
    'lon': 100.9999,
    'quality': 0.2,
    'distance_to_road': 500
}
# ❌ ไม่ผ่าน: quality < 0.3, distance > 100
```

---

## 2️⃣ MODEL CONFIGURATION

### **🏗️ Architecture Parameters**

```python
NUM_FEATURES = 10
```

**อธิบาย:**
- จำนวน input features = **10**
- ทุก features ป้อนเข้าโมเดล

**10 Features คืออะไร?**

| # | Feature | คำอธิบาย | ช่วงค่า |
|---|---------|----------|---------|
| 1 | `speed_mean` | ความเร็วเฉลี่ย | 0-150 km/h |
| 2 | `speed_median` | ความเร็วกลาง | 0-150 km/h |
| 3 | `speed_std` | ส่วนเบี่ยงเบนความเร็ว | 0-50 km/h |
| 4 | `hour_sin` | ชั่วโมง (sine) | -1 to 1 |
| 5 | `hour_cos` | ชั่วโมง (cosine) | -1 to 1 |
| 6 | `day_of_week` | วันในสัปดาห์ | 0-6 |
| 7 | `is_weekend` | วันหยุดหรือไม่ | 0 or 1 |
| 8 | `time_since_rush` | ห่างจากชั่วโมงเร่งด่วน | 0-12 ชม. |
| 9 | `nearby_congestion` | การจราจรบริเวณใกล้เคียง | 0-3 |
| 10 | `speed_lag_1` | ความเร็ว 5 นาทีก่อน | 0-150 km/h |

**ตัวอย่างข้อมูล 1 ตัวอย่าง:**
```python
sample = [
    45.5,    # speed_mean
    44.0,    # speed_median
    5.2,     # speed_std
    0.707,   # hour_sin (08:00)
    0.707,   # hour_cos
    1,       # day_of_week (อังคาร)
    0,       # is_weekend (ไม่ใช่)
    0.5,     # time_since_rush
    1,       # nearby_congestion
    48.0     # speed_lag_1
]
```

---

```python
HIDDEN_DIM = 64
```

**อธิบาย:**
- ขนาด hidden layer = **64 neurons**
- เป็นชั้นกลางระหว่าง input และ output

**สถาปัตยกรรม:**
```
Input Layer (10)
    ↓
Hidden Layer 1 (64) ← HIDDEN_DIM
    ↓
Hidden Layer 2 (64)
    ↓
Output Layer (4+2)
```

**ทำไมเลือก 64?**

| ขนาด | ข้อดี | ข้อเสีย |
|------|-------|---------|
| 16 | เร็ว, เบา | เรียนรู้ไม่พอ |
| **64** | **สมดุล** ✅ | **พอดี** |
| 256 | เรียนรู้ได้มาก | ช้า, overfitting |
| 1024 | แม่นมาก | ช้ามาก, memory เยอะ |

---

```python
NUM_CONGESTION_CLASSES = 4
NUM_RUSH_HOUR_CLASSES = 2
```

**อธิบาย:**
- โมเดลทำนาย **2 tasks** พร้อมกัน (Multi-Task Learning)

**Task 1: Congestion Classification (4 classes)**

| Class | Label | เงื่อนไข | ความหมาย |
|-------|-------|----------|----------|
| 0 | Gridlock | speed < 10 km/h | ติดขัดรุนแรง |
| 1 | Congested | 10 ≤ speed < 30 | ติดขัด |
| 2 | Moderate | 30 ≤ speed < 50 | ปานกลาง |
| 3 | Free Flow | speed ≥ 50 | คล่องตัว |

**Task 2: Rush Hour Classification (2 classes)**

| Class | Label | เงื่อนไข |
|-------|-------|----------|
| 0 | Non-Rush Hour | เวลาปกติ |
| 1 | Rush Hour | 7-9 น. หรือ 17-19 น. |

**ตัวอย่างผลลัพธ์:**
```python
prediction = {
    'congestion': 1,  # Congested
    'rush_hour': 1,   # Rush Hour
    'confidence': {
        'congestion': 0.85,
        'rush_hour': 0.92
    }
}
```

---

### **🚦 Classification Thresholds**

```python
CONGESTION_THRESHOLDS = {
    'gridlock': 10,
    'congested': 30,
    'moderate': 50,
    'free_flow': float('inf')
}
```

**อธิบาย:**
- เกณฑ์สำหรับแบ่งระดับการจราจร

**ฟังก์ชันจำแนก:**
```python
def classify_congestion(speed):
    if speed < 10:
        return 0, "Gridlock"
    elif speed < 30:
        return 1, "Congested"
    elif speed < 50:
        return 2, "Moderate"
    else:
        return 3, "Free Flow"
```

**ตัวอย่าง:**
```python
classify_congestion(5)   # → (0, "Gridlock")
classify_congestion(25)  # → (1, "Congested")
classify_congestion(45)  # → (2, "Moderate")
classify_congestion(80)  # → (3, "Free Flow")
```

**การปรับเกณฑ์:**
```python
# สำหรับเมืองที่รถช้ากว่า
CONGESTION_THRESHOLDS = {
    'gridlock': 5,   # เข้มงวดขึ้น
    'congested': 20,
    'moderate': 40,
    'free_flow': float('inf')
}

# สำหรับทางด่วน
CONGESTION_THRESHOLDS = {
    'gridlock': 20,  # ผ่อนปรนกว่า
    'congested': 50,
    'moderate': 80,
    'free_flow': float('inf')
}
```

---

```python
RUSH_HOUR_RANGES = [
    (7, 9),   # Morning: 7:00-9:00
    (17, 19)  # Evening: 17:00-19:00
]
```

**อธิบาย:**
- กำหนดช่วงชั่วโมงเร่งด่วน
- รูปแบบ: (เริ่ม, สิ้นสุด) 24-hour format

**ฟังก์ชันตรวจสอบ:**
```python
def is_rush_hour(hour, is_weekday=True):
    if not is_weekday:
        return False  # วันหยุดไม่มีชั่วโมงเร่งด่วน
    
    for start, end in RUSH_HOUR_RANGES:
        if start <= hour < end:
            return True
    return False
```

**ตัวอย่าง:**
```python
is_rush_hour(8, True)   # → True (เช้า)
is_rush_hour(18, True)  # → True (เย็น)
is_rush_hour(12, True)  # → False (เที่ยง)
is_rush_hour(8, False)  # → False (วันหยุด)
```

**การปรับช่วงเวลา:**
```python
# เพิ่มช่วงเที่ยง
RUSH_HOUR_RANGES = [
    (7, 9),
    (12, 13),  # เพิ่มเวลาพักเที่ยง
    (17, 19)
]

# เมืองที่ชั่วโมงเร่งด่วนยาวนานกว่า
RUSH_HOUR_RANGES = [
    (6, 10),   # เช้ายาวขึ้น
    (16, 20)   # เย็นยาวขึ้น
]
```

---

## 3️⃣ TRAINING CONFIGURATION

### **📚 Training Parameters**

```python
EPOCHS = 50
```

**อธิบาย:**
- จำนวนรอบการเทรน = **50 epochs**
- 1 epoch = โมเดลเห็นข้อมูลทั้งหมด 1 รอบ

**Timeline การเทรน:**
```
Epoch 1/50:  Loss = 2.345 (โมเดลงงมาก)
Epoch 10/50: Loss = 1.234 (เริ่มเข้าใจ)
Epoch 25/50: Loss = 0.567 (เก่งขึ้น)
Epoch 40/50: Loss = 0.234 (เก่งมาก)
Epoch 50/50: Loss = 0.198 (เก่งที่สุด) ✅
```

**การปรับ epochs:**
```python
# Dataset เล็ก
EPOCHS = 30  # เร็ว, อาจ underfit

# Dataset กลาง
EPOCHS = 50  # สมดุล ✅

# Dataset ใหญ่
EPOCHS = 100  # ช้า แต่แม่นยำกว่า
```

---

```python
BATCH_SIZE = 32
```

**อธิบาย:**
- ประมวลผลทีละ **32 ตัวอย่าง**
- ทำซ้ำจนครบทุกข้อมูล

**ตัวอย่าง:**
```python
total_samples = 1000
batch_size = 32
num_batches = 1000 / 32 = 31.25 ≈ 32 batches

# การแบ่ง batches
Batch 1:  samples [0:32]
Batch 2:  samples [32:64]
Batch 3:  samples [64:96]
...
Batch 31: samples [960:992]
Batch 32: samples [992:1000]  # 8 ตัวอย่าง
```

**Batch Size vs Performance:**

| Batch Size | ความเร็ว | Memory | เสถียรภาพ | แนะนำ |
|------------|----------|--------|-----------|-------|
| 8 | ช้า | น้อย | ไม่เสถียร | Dataset เล็ก |
| **32** | **กลาง** | **กลาง** | **ดี** | **Standard** ✅ |
| 64 | เร็ว | มาก | ดีมาก | GPU แรง |
| 128 | เร็วมาก | เยอะมาก | ดีที่สุด | GPU แรงมาก |

---

```python
LEARNING_RATE = 0.001
```

**อธิบาย:**
- อัตราการเรียนรู้ = **0.001** (1e-3)
- ควบคุมขนาดก้าวในการปรับ weights

**สูตรการอัปเดต:**
```python
new_weight = old_weight - (LEARNING_RATE × gradient)
```

**ตัวอย่าง:**
```python
old_weight = 0.5
gradient = 2.0
LEARNING_RATE = 0.001

new_weight = 0.5 - (0.001 × 2.0)
           = 0.5 - 0.002
           = 0.498
```

**Learning Rate Comparison:**

| LR | ขนาดก้าว | ความเร็ว | ความแม่นยำ | ปัญหา |
|----|-----------|----------|-------------|-------|
| 0.1 | ใหญ่มาก | เร็วมาก | ต่ำ | กระโดดข้ามจุดที่ดี |
| 0.01 | ใหญ่ | เร็ว | ปานกลาง | อาจพลาดบางจุด |
| **0.001** | **กลาง** | **พอดี** | **ดี** | **สมดุล** ✅ |
| 0.0001 | เล็ก | ช้า | สูง | ใช้เวลานาน |
| 0.00001 | เล็กมาก | ช้ามาก | สูงมาก | ช้าเกินไป |

**การใช้ Learning Rate Scheduler:**
```python
# เริ่มต้นด้วย LR สูง แล้วค่อยๆ ลด
Epoch 1-10:  LR = 0.001
Epoch 11-20: LR = 0.0005
Epoch 21-30: LR = 0.0001
Epoch 31+:   LR = 0.00005
```

---

```python
WEIGHT_DECAY = 1e-4  # 0.0001
```

**อธิบาย:**
- **L2 Regularization** = ลดขนาด weights
- ป้องกัน overfitting

**สูตร:**
```python
loss_total = loss_task + (WEIGHT_DECAY × sum(weight²))
```

**ตัวอย่าง:**
```python
# ไม่มี weight decay
weights = [1.5, 2.8, 3.2, 4.1]  # โตมาก
# → โมเดลจำข้อมูลเก่าตายตัว (overfit)

# มี weight decay
weights = [0.5, 0.8, 0.9, 1.2]  # เล็กกว่า
# → โมเดลยืดหยุ่นกว่า (generalize ดีขึ้น)
```

**ค่า Weight Decay:**

| ค่า | ผลกระทบ | แนะนำใช้ |
|-----|----------|----------|
| 0 | ไม่มี regularization | Dataset เล็ก |
| **1e-4** | **กลางๆ** ✅ | **Standard** |
| 1e-3 | แรงมาก | Dataset ใหญ่ + overfit มาก |

---

```python
CONGESTION_LOSS_WEIGHT = 1.0
RUSH_HOUR_LOSS_WEIGHT = 1.0
```

**อธิบาย:**
- น้ำหนักของแต่ละ task
- รวมเป็น total loss

**สูตร:**
```python
total_loss = (CONGESTION_LOSS_WEIGHT × congestion_loss) + 
             (RUSH_HOUR_LOSS_WEIGHT × rush_hour_loss)
```

**ตัวอย่าง:**
```python
# น้ำหนักเท่ากัน
congestion_loss = 0.5
rush_hour_loss = 0.3
total_loss = (1.0 × 0.5) + (1.0 × 0.3) = 0.8

# ให้ความสำคัญกับ rush hour มากกว่า
RUSH_HOUR_LOSS_WEIGHT = 2.0
total_loss = (1.0 × 0.5) + (2.0 × 0.3) = 1.1

# ให้ความสำคัญกับ congestion มากกว่า
CONGESTION_LOSS_WEIGHT = 3.0
RUSH_HOUR_LOSS_WEIGHT = 1.0
total_loss = (3.0 × 0.5) + (1.0 × 0.3) = 1.8
```

---

```python
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2
```

**อธิบาย:**
- แบ่งข้อมูล: **80% เทรน, 20% ตรวจสอบ**

**ตัวอย่าง:**
```python
total_data = 1000 ตัวอย่าง

# แบ่งข้อมูล
train_size = 1000 × 0.8 = 800 ตัวอย่าง
val_size = 1000 × 0.2 = 200 ตัวอย่าง

# Training Set (800)
- ใช้เทรนโมเดล
- โมเดล "เห็น" ข้อมูลนี้

# Validation Set (200)
- ใช้ตรวจสอบ
- โมเดล "ไม่เห็น" ตอนเทรน
- เช็คว่า overfit หรือไม่
```

**การแบ่งข้อมูลแบบอื่น:**

| Train | Val | Test | ใช้เมื่อ |
|-------|-----|------|----------|
| 60% | 20% | 20% | มีข้อมูลน้อย |
| 70% | 15% | 15% | ข้อมูลปานกลาง |
| **80%** | **20%** | **-** | **ไม่มี test set** ✅ |
| 80% | 10% | 10% | มีข้อมูลเยอะ |

---

```python
PATIENCE = 10
MIN_DELTA = 0.001
```

**อธิบาย:**
- **Early Stopping** = หยุดเทรนก่อนครบถ้าไม่ดีขึ้น

**พารามิเตอร์:**
- `PATIENCE`: รอ 10 epochs
- `MIN_DELTA`: ต้องดีขึ้นอย่างน้อย 0.001

**ตัวอย่าง:**
```python
Epoch 20: val_loss = 0.500
Epoch 21: val_loss = 0.498  # ดีขึ้น 0.002 > 0.001 ✅
Epoch 22: val_loss = 0.497  # ดีขึ้น 0.001 = 0.001 ✅
Epoch 23: val_loss = 0.497  # ไม่ดีขึ้น ❌ counter=1
Epoch 24: val_loss = 0.498  # แย่ลง ❌ counter=2
Epoch 25: val_loss = 0.497  # ไม่ดีขึ้น ❌ counter=3
...
Epoch 33: val_loss = 0.497  # ❌ counter=10
→ หยุดเทรน! (ครบ PATIENCE)

# บันทึกโมเดลจาก Epoch 22 (ดีที่สุด)
```

**ประโยชน์:**
- ประหยัดเวลา (ไม่เทรนครบ 50 epochs)
- ป้องกัน overfitting
- ได้โมเดลที่ดีที่สุด

---

## 4️⃣ GRAPH CONFIGURATION

### **🕸️ Graph Construction**

```python
MAX_EDGE_DISTANCE = 500  # meters
MIN_EDGE_DISTANCE = 10   # meters
```

**อธิบาย:**
- สร้าง edge (เส้นเชื่อม) ถ้าระยะห่าง 10-500 เมตร

**ตัวอย่าง:**
```python
# สถานที่ 2 แห่ง
A = (13.7465, 100.5326)  # Siam
B = (13.7443, 100.5300)  # MBK

distance = haversine(A, B) = 250 เมตร

if 10 <= distance <= 500:
    create_edge(A, B)  # ✅ สร้าง edge
```

**เหตุผล:**
```python
# < 10 เมตร
distance = 5 เมตร
# → ใกล้เกินไป (อาจเป็นตำแหน่งเดียวกัน)

# 10-500 เมตร ✅
distance = 250 เมตร
# → เหมาะสม (เชื่อมโยงกันได้)

# > 500 เมตร
distance = 2000 เมตร
# → ไกลเกินไป (ไม่เกี่ยวข้องกัน)
```

**กราฟที่ได้:**
```
[Siam] ----250m---- [MBK]
   |                  |
  300m               400m
   |                  |
[CentralWorld]----350m----[Platinum]
```

---

```python
SPATIAL_THRESHOLD = 0.001  # degrees
```

**อธิบาย:**
- ความละเอียดทางภูมิศาสตร์
- 0.001° ≈ 110 เมตร (ที่กรุงเทพ)

**การแปลงค่า:**
```python
# ที่ Latitude 13° (กรุงเทพ)
1 degree ≈ 111 km
0.001 degree ≈ 111 m
0.0001 degree ≈ 11 m

# สูตร
distance_meters = degrees × 111,000
```

**การใช้งาน:**
```python
def cluster_locations(locations, threshold=0.001):
    """จัดกลุ่มสถานที่ที่ใกล้กันมาก"""
    clusters = []
    for loc in locations:
        if distance_to_nearest_cluster(loc) < threshold:
            add_to_cluster(loc)
        else:
            create_new_cluster(loc)
    return clusters
```

---

```python
INCLUDE_EDGE_FEATURES = True
NORMALIZE_FEATURES = True
```

**อธิบาย:**

### **1. INCLUDE_EDGE_FEATURES**

```python
if INCLUDE_EDGE_FEATURES:
    edge_features = {
        'distance': 250,           # ระยะห่าง (เมตร)
        'speed_diff': 15,          # ผลต่างความเร็ว
        'congestion_diff': 1,      # ผลต่างการจราจร
        'same_district': 1,        # เขตเดียวกัน
        'road_type': 'highway'     # ประเภทถนน
    }
```

### **2. NORMALIZE_FEATURES**

```python
# ก่อน Normalize
features = {
    'speed': 45,        # 0-150
    'distance': 2500,   # 0-10000
    'hour': 8,          # 0-24
    'quality': 0.85     # 0-1
}

# หลัง Normalize (0-1)
features_norm = {
    'speed': 0.30,      # 45/150
    'distance': 0.25,   # 2500/10000
    'hour': 0.33,       # 8/24
    'quality': 0.85     # 0.85/1
}
```

**ทำไมต้อง normalize?**
- Features มีหน่วยต่างกัน
- ให้ทุก feature มีความสำคัญเท่ากัน
- โมเดลเทรนได้เร็วและแม่นยำขึ้น

---

## 5️⃣ OUTPUT CONFIGURATION

```python
OUTPUT_DIR = "outputs"
```

**โครงสร้างโฟลเดอร์:**
```
outputs/
├── best_model.pth              # โมเดลที่ดีที่สุด
├── best_enhanced_model.pth     # โมเดล enhanced ที่ดีที่สุด
├── processed_data.pkl          # ข้อมูลที่ประมวลผลแล้ว
├── training_history.pkl        # ประวัติการเทรน
├── evaluation_results.pkl      # ผลการประเมิน
├── traffic_gnn.log            # Log file
└── plots/                      # กราฟต่างๆ
    ├── loss_curve.png
    ├── accuracy_plot.png
    ├── confusion_matrix.png
    └── comparison_chart.png
```

---

```python
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
```

**อธิบาย:**
- บันทึกโมเดลที่มี validation loss ต่ำสุด

**ไฟล์ที่บันทึก:**
```python
torch.save({
    'epoch': 35,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': 0.234,
    'val_loss': 0.287,
    'val_accuracy': {
        'congestion': 0.954,
        'rush_hour': 0.982
    },
    'config': CONFIG_DICT
}, MODEL_SAVE_PATH)
```

**การโหลดกลับมาใช้:**
```python
checkpoint = torch.load('outputs/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Best epoch: {checkpoint['epoch']}")
print(f"Val accuracy: {checkpoint['val_accuracy']}")
```

---

```python
FIGURE_SIZE = (12, 8)
DPI = 300
SAVE_PLOTS = True
```

**อธิบาย:**

**1. FIGURE_SIZE:**
- ขนาดกราฟ = 12×8 นิ้ว
- เหมาะสำหรับนำเสนอ/พิมพ์

**2. DPI (Dots Per Inch):**
- 300 DPI = คุณภาพสูง (สำหรับพิมพ์)
- 72 DPI = คุณภาพปกติ (สำหรับหน้าจอ)

**3. SAVE_PLOTS:**
- `True` → บันทึกเป็นไฟล์
- `False` → แสดงบนหน้าจอเท่านั้น

**ตัวอย่าง:**
```python
import matplotlib.pyplot as plt

plt.figure(figsize=FIGURE_SIZE)
plt.plot(train_loss, label='Train')
plt.plot(val_loss, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

if SAVE_PLOTS:
    plt.savefig('outputs/plots/loss_curve.png', 
                dpi=DPI, 
                bbox_inches='tight')
else:
    plt.show()
```

---

## 6️⃣ DASHBOARD CONFIGURATION

```python
DASHBOARD_PORT = 8501
AUTO_REFRESH_SECONDS = 30
MAX_MAP_POINTS = 100
```

**อธิบาย:**

**1. DASHBOARD_PORT:**
- Dashboard เปิดที่: `http://localhost:8501`

**2. AUTO_REFRESH:**
- รีเฟรชข้อมูลทุก 30 วินาที
- Real-time monitoring

**3. MAX_MAP_POINTS:**
- แสดงสูงสุด 100 จุดบนแผนที่
- ป้องกันแผนที่ช้า

---

```python
DEFAULT_MAP_CENTER = [13.7563, 100.5018]  # Bangkok
DEFAULT_ZOOM = 11
MAP_STYLE = 'OpenStreetMap'
```

**อธิบาย:**

**1. MAP_CENTER:**
- จุดกึ่งกลาง = กรุงเทพมหานคร

**2. ZOOM Levels:**
```
1  - เห็นทั้งโลก
5  - เห็นประเทศ
11 - เห็นเมืองทั้งเมือง ✅
15 - เห็นย่าน
20 - เห็นบ้านเดียว
```

**3. MAP_STYLE:**
- `'OpenStreetMap'`: แผนที่มาตรฐาน
- `'CartoDB'`: แผนที่สะอาด
- `'Stamen Terrain'`: แสดงภูมิประเทศ

---

```python
CONGESTION_COLORS = {
    'Gridlock': '#FF4444',    # แดง
    'Congested': '#FF8800',   # ส้ม
    'Moderate': '#FFAA00',    # เหลือง
    'Free Flow': '#44FF44'    # เขียว
}

RUSH_HOUR_COLORS = {
    'Non-Rush Hour': '#4ECDC4',  # ฟ้า
    'Rush Hour': '#FF6B6B'       # แดง
}
```

**Color Scheme:**
```
🔴 #FF4444 Gridlock      → แดงเข้ม (อันตราย)
🟠 #FF8800 Congested     → ส้ม (เตือน)
🟡 #FFAA00 Moderate      → เหลือง (ระวัง)
🟢 #44FF44 Free Flow     → เขียว (ปลอดภัย)

🔵 #4ECDC4 Non-Rush Hour → ฟ้า (ปกติ)
🔴 #FF6B6B Rush Hour     → แดงอ่อน (เร่งด่วน)
```

---

## 7️⃣ LOGGING CONFIGURATION

```python
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(OUTPUT_DIR, 'traffic_gnn.log')
```

**Log Levels:**
```
DEBUG    → ทุกอย่างละเอียด
INFO     → ข้อมูลสำคัญ ✅
WARNING  → คำเตือน
ERROR    → ข้อผิดพลาด
CRITICAL → ข้อผิดพลาดร้ายแรง
```

**ตัวอย่าง Log Output:**
```
2024-10-05 08:30:45 - TrafficGNN - INFO - Starting training...
2024-10-05 08:30:50 - TrafficGNN - INFO - Epoch 1/50, Loss: 1.234
2024-10-05 08:31:00 - TrafficGNN - INFO - Epoch 2/50, Loss: 1.102
2024-10-05 09:15:20 - TrafficGNN - INFO - Training completed!
2024-10-05 09:15:25 - TrafficGNN - INFO - Best model saved.
```

---

## 8️⃣ THAI LOCALIZATION

```python
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

THAI_TIME_FORMAT = '%H:%M น.'
THAI_DATE_FORMAT = '%d/%m/%Y'
```

**ตัวอย่างการใช้:**
```python
# ภาษาอังกฤษ
label_en = "Gridlock"

# แปลเป็นภาษาไทย
label_th = THAI_LABELS['congestion'][label_en]
print(label_th)  # "ติดขัดรุนแรง"

# รูปแบบเวลา
from datetime import datetime
now = datetime.now()
time_th = now.strftime(THAI_TIME_FORMAT)
date_th = now.strftime(THAI_DATE_FORMAT)
print(f"{time_th} {date_th}")  # "08:30 น. 05/10/2024"
```

---

## 9️⃣ SYSTEM CONFIGURATION

```python
USE_GPU = True
NUM_WORKERS = 4
PIN_MEMORY = True
MAX_MEMORY_GB = 8
BATCH_SIZE_AUTO_ADJUST = True
RANDOM_SEED = 42
```

**อธิบาย:**

**1. USE_GPU:**
```python
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    device = torch.device('cpu')
    print("Using CPU")
```

**2. NUM_WORKERS:**
- จำนวน threads สำหรับโหลดข้อมูล
- 4 workers = โหลดพร้อมกัน 4 กระบวนการ

**3. PIN_MEMORY:**
- เก็บข้อมูลใน pinned memory
- ส่งข้อมูล CPU → GPU เร็วขึ้น

**4. BATCH_SIZE_AUTO_ADJUST:**
```python
try:
    train_model(batch_size=64)
except RuntimeError as e:
    if "out of memory" in str(e):
        # ลด batch size อัตโนมัติ
        train_model(batch_size=32)
```

**5. RANDOM_SEED:**
```python
import random
import numpy as np
import torch

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
# → ผลลัพธ์เหมือนกันทุกครั้ง
```

---

## 🔟 VALIDATION FUNCTION

```python
def validate_config():
    """ตรวจสอบความถูกต้องของการตั้งค่า"""
    
    # เช็คค่าต้องเป็นบวก
    assert AGGREGATION_MINUTES > 0
    assert MIN_PROBES_PER_BIN > 0
    assert EPOCHS > 0
    assert BATCH_SIZE > 0
    assert LEARNING_RATE > 0
    
    # เช็คค่าต้องอยู่ในช่วง 0-1
    assert 0 < TRAIN_SPLIT < 1
    assert TRAIN_SPLIT + VAL_SPLIT <= 1
    
    # สร้างโฟลเดอร์ outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("✅ Configuration validation passed!")
```

**การใช้งาน:**
```python
# ตอนเริ่มโปรแกรม
if __name__ == "__main__":
    validate_config()
    # จะ error ถ้าค่าไม่ถูกต้อง
```

---

## 📊 สรุป

### **จำนวนพารามิเตอร์แต่ละหมวด:**

| หมวดหมู่ | จำนวน | ความสำคัญ |
|---------|--------|----------|
| Data | 10 | ⭐⭐⭐⭐⭐ |
| Model | 8 | ⭐⭐⭐⭐⭐ |
| Training | 8 | ⭐⭐⭐⭐⭐ |
| Graph | 5 | ⭐⭐⭐⭐ |
| Output | 6 | ⭐⭐⭐ |
| Dashboard | 8 | ⭐⭐⭐ |
| Logging | 4 | ⭐⭐ |
| Thai | 3 | ⭐⭐ |
| System | 7 | ⭐⭐⭐⭐ |
| **รวม** | **59** | |

### **ข้อดีของการใช้ config.py:**

✅ **จัดการง่าย** - แก้ค่าที่เดียว  
✅ **ป้องกันข้อผิดพลาด** - มี validation  
✅ **ทดลองได้ง่าย** - สลับ config ได้  
✅ **อ่านง่าย** - เข้าใจได้ชัดเจน  
✅ **ใช้ซ้ำได้** - import ไปใช้ทุกไฟล์

---

## 🎓 ตัวอย่างการใช้งาน

```python
# ไฟล์อื่นๆ import config
from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    MODEL_SAVE_PATH
)

# ใช้ค่าจาก config
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE
)

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

for epoch in range(EPOCHS):
    train_one_epoch()
    
torch.save(model.state_dict(), MODEL_SAVE_PATH)
```

---

**สร้างเมื่อ:** 5 ตุลาคม 2025  
**เวอร์ชัน:** 1.0  
**ผู้เขียน:** Traffic GNN Classification Team
