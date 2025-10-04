# 📊 การประมวลผลข้อมูล - Data Processing (ภาษาไทย)

## 📚 สารบัญ

1. [ข้อมูลดิบที่ใช้](#ข้อมูลดิบที่ใช้)
2. [การทำความสะอาดข้อมูล](#การทำความสะอาดข้อมูล)
3. [Feature Engineering](#feature-engineering)
4. [การคำนวณระยะทาง](#การคำนวณระยะทาง)
5. [การแบ่งข้อมูล](#การแบ่งข้อมูล)
6. [Data Augmentation](#data-augmentation)

---

## 📥 ข้อมูลดิบที่ใช้

### **1. PROBE Data (ข้อมูล GPS)**

**ที่มา:** โฟลเดอร์ `Data/PROBE-202401/`, `PROBE-202402/`, etc.

**โครงสร้างไฟล์:**
```
PROBE-202401/
├── 20240101.csv.out
├── 20240102.csv.out
├── ...
└── 20240131.csv.out
```

**ข้อมูลในแต่ละไฟล์:**
```csv
timestamp,location_id,latitude,longitude,speed,heading,quality
2024-01-01 00:05:00,LOC001,13.7563,100.5018,45.5,90.0,0.85
2024-01-01 00:10:00,LOC001,13.7563,100.5018,42.0,90.0,0.90
2024-01-01 00:15:00,LOC002,13.7600,100.5050,38.5,180.0,0.75
...
```

**คำอธิบายคอลัมน์:**

| Column | คำอธิบาย | หน่วย | ช่วงค่า |
|--------|----------|------|---------|
| **timestamp** | วันเวลาที่บันทึก | datetime | 2024-01-01 00:00:00 |
| **location_id** | รหัสสถานที่ | string | LOC001-LOC217 |
| **latitude** | พิกัด latitude | degrees | 13.5-14.0 (กรุงเทพฯ) |
| **longitude** | พิกัด longitude | degrees | 100.3-100.8 (กรุงเทพฯ) |
| **speed** | ความเร็วของรถ | km/h | 0-120 |
| **heading** | ทิศทางการเคลื่อนที่ | degrees | 0-360 |
| **quality** | คุณภาพข้อมูล | score | 0.0-1.0 |

### **2. OpenStreetMap Data (โครงสร้างถนน)**

**ที่มา:** โฟลเดอร์ `Data/hotosm_tha_roads_lines_gpkg/`

**ข้อมูลที่ใช้:**
- โครงสร้างถนน (geometry)
- ชื่อถนน
- ประเภทถนน (highway type)
- ความเชื่อมต่อระหว่างถนน

### **3. Traffic Events (เหตุการณ์การจราจร)**

**ที่มา:** โฟลเดอร์ `iTIC-Longdo-Traffic-events-2022/`

**ประเภทเหตุการณ์:**
- อุบัติเหตุ (accidents)
- การก่อสร้าง (construction)
- เหตุการณ์พิเศษ (special events)
- การปิดถนน (road closures)

---

## 🧹 การทำความสะอาดข้อมูล

### **1. การลบข้อมูลผิดปกติ (Outlier Removal)**

**ปัญหา:** ข้อมูล GPS อาจมีค่าผิดปกติ

**วิธีการ:**

#### **A. ความเร็วผิดปกติ**
```python
# ลบความเร็วที่เป็นไปไม่ได้
df = df[df['speed'] >= 0]         # ไม่ติดลบ
df = df[df['speed'] <= 150]       # ไม่เกิน 150 km/h (ในเมือง)

# ลบความเร็วที่ผิดพลาดอย่างชัดเจน
# เช่น รถยืน แต่ speed = 100 km/h
df = df[~((df['speed'] > 80) & (df['location_type'] == 'intersection'))]
```

**คำอธิบาย:**
- **ความเร็วติดลบ:** เป็นไปไม่ได้ → ลบทิ้ง
- **ความเร็วสูงเกินไป:** ในเมือง > 150 km/h → น่าจะผิดพลาด
- **ขัดแย้งกับบริบท:** จุดตัดถนนแต่ความเร็วสูง → น่าจะผิด

#### **B. พิกัดผิดปกติ**
```python
# กรุงเทพฯ: lat 13.5-14.0, lon 100.3-100.8
BANGKOK_BOUNDS = {
    'lat_min': 13.5,
    'lat_max': 14.0,
    'lon_min': 100.3,
    'lon_max': 100.8
}

# กรองข้อมูลนอกขอบเขต
df = df[
    (df['latitude'] >= BANGKOK_BOUNDS['lat_min']) &
    (df['latitude'] <= BANGKOK_BOUNDS['lat_max']) &
    (df['longitude'] >= BANGKOK_BOUNDS['lon_min']) &
    (df['longitude'] <= BANGKOK_BOUNDS['lon_max'])
]
```

**คำอธิบาย:**
- ข้อมูลนอกพื้นที่ศึกษา → ลบทิ้ง
- ป้องกัน GPS signal drift

#### **C. ข้อมูลซ้ำ (Duplicates)**
```python
# ลบข้อมูลซ้ำ (เวลาและสถานที่เดียวกัน)
df = df.drop_duplicates(
    subset=['timestamp', 'location_id'],
    keep='first'  # เก็บตัวแรก
)
```

### **2. การจัดการข้อมูลหาย (Missing Data)**

**วิธีการต่างๆ:**

#### **A. Forward Fill (เติมด้วยค่าก่อนหน้า)**
```python
# สำหรับข้อมูลที่เปลี่ยนช้า (location_id, heading)
df['heading'] = df.groupby('location_id')['heading'].fillna(method='ffill')
```

**เหมาะกับ:**
- ทิศทางรถ (ไม่เปลี่ยนบ่อย)
- ตำแหน่ง

#### **B. Interpolation (แทรกค่า)**
```python
# สำหรับข้อมูลที่เปลี่ยนแปลงต่อเนื่อง (speed)
df['speed'] = df.groupby('location_id')['speed'].interpolate(
    method='linear'  # เชื่อมเป็นเส้นตรง
)
```

**ตัวอย่าง:**
```python
# ก่อน interpolate
time:  10:00  10:05  10:10  10:15  10:20
speed:  45    NaN    NaN    NaN    35

# หลัง interpolate (linear)
time:  10:00  10:05  10:10  10:15  10:20
speed:  45    42.5   40.0   37.5   35
```

**การคำนวณ:**
```python
# Linear interpolation
value_at_10:05 = 45 + (35 - 45) * (1/4) = 42.5
value_at_10:10 = 45 + (35 - 45) * (2/4) = 40.0
value_at_10:15 = 45 + (35 - 45) * (3/4) = 37.5
```

#### **C. Mean/Median Imputation**
```python
# สำหรับข้อมูลที่ขาดน้อย
# ใช้ค่าเฉลี่ยของกลุ่มเดียวกัน

# เติมด้วย mean ตามเวลาและสถานที่
df['speed'] = df.groupby(['location_id', 'hour'])['speed'].transform(
    lambda x: x.fillna(x.mean())
)
```

**ตัวอย่าง:**
```python
# ความเร็วเฉลี่ยที่ LOC001 ในช่วง 8:00-9:00 = 40 km/h
# ถ้าขาดข้อมูล → เติม 40
```

### **3. Quality Filtering (กรองตามคุณภาพ)**

```python
# เก็บเฉพาะข้อมูลคุณภาพสูง
QUALITY_THRESHOLD = 0.3

df = df[df['quality_score'] > QUALITY_THRESHOLD]
```

**Quality Score คำนวณจาก:**
```python
def calculate_quality_score(row):
    score = 1.0
    
    # ลดคะแนนถ้าจำนวน probes น้อย
    if row['count_probes'] < 5:
        score *= 0.7
    
    # ลดคะแนนถ้าความเร็วแปรปรวนมาก
    if row['speed_std'] > 20:
        score *= 0.8
    
    # ลดคะแนนถ้า GPS accuracy ต่ำ
    if row['gps_accuracy'] > 50:  # meters
        score *= 0.6
    
    return score
```

---

## 🔧 Feature Engineering

### **1. Speed Features (คุณสมบัติความเร็ว)**

```python
# คำนวณ statistics จาก speed ในช่วงเวลาหนึ่ง
# เช่น ทุก 5 นาที → aggregate เป็น 1 ค่า

aggregated = df.groupby(['location_id', 'time_window']).agg({
    'speed': ['mean', 'median', 'std', 'min', 'max']
})

# Features ที่ได้:
features = {
    'mean_speed': 45.5,      # ความเร็วเฉลี่ย
    'median_speed': 42.0,    # ความเร็วกลาง
    'speed_std': 5.2,        # ส่วนเบี่ยงเบน
    'min_speed': 35.0,       # ความเร็วต่ำสุด
    'max_speed': 60.0,       # ความเร็วสูงสุด
    'speed_range': 25.0      # พิสัย (max - min)
}
```

**คำอธิบาย:**

- **mean_speed:** ความเร็วโดยเฉลี่ย → บอกสภาพการจราจรโดยรวม
- **median_speed:** ความเร็วตรงกลาง → ไม่ได้รับผลจาก outliers
- **speed_std:** ความแปรปรวน → บอกความสม่ำเสมอของการจราจร
  - std ต่ำ = การจราจรสม่ำเสมอ
  - std สูง = การจราจรไม่แน่นอน (stop-and-go)

**ตัวอย่างการคำนวณ:**
```python
speeds = [40, 42, 45, 48, 50, 38, 41]

# Mean
mean = sum(speeds) / len(speeds)
     = (40+42+45+48+50+38+41) / 7
     = 304 / 7
     = 43.43 km/h

# Median (เรียงลำดับก่อน: [38,40,41,42,45,48,50])
median = 42 km/h  # ตัวกลาง

# Standard Deviation
std = sqrt(Σ(x - mean)² / n)
    = sqrt(((40-43.43)² + (42-43.43)² + ... + (41-43.43)²) / 7)
    = sqrt(57.43 / 7)
    = 2.87 km/h
```

### **2. Temporal Features (คุณสมบัติเวลา)**

#### **A. Hour Encoding (Cyclic)**

**ปัญหา:**
```python
# ❌ ใช้ตัวเลขธรรมดา
hour = 23  # 23:00
hour = 0   # 00:00

# 23 และ 0 ห่างกัน 23 → ผิด! ควรใกล้กัน
```

**วิธีแก้: Sine/Cosine Encoding**
```python
# ✅ ใช้ sine/cosine
import numpy as np

def encode_hour(hour):
    # แปลง hour (0-23) เป็น angle (0-2π)
    angle = 2 * np.pi * hour / 24
    
    hour_sin = np.sin(angle)
    hour_cos = np.cos(angle)
    
    return hour_sin, hour_cos
```

**ตัวอย่างการคำนวณ:**
```python
# Hour 0 (00:00)
angle = 2π * 0 / 24 = 0
sin(0) = 0.0
cos(0) = 1.0
→ (0.0, 1.0)

# Hour 6 (06:00)
angle = 2π * 6 / 24 = π/2
sin(π/2) = 1.0
cos(π/2) = 0.0
→ (1.0, 0.0)

# Hour 12 (12:00)
angle = 2π * 12 / 24 = π
sin(π) = 0.0
cos(π) = -1.0
→ (0.0, -1.0)

# Hour 18 (18:00)
angle = 2π * 18 / 24 = 3π/2
sin(3π/2) = -1.0
cos(3π/2) = 0.0
→ (-1.0, 0.0)

# Hour 23 (23:00)
angle = 2π * 23 / 24 = 23π/12
sin(23π/12) ≈ -0.26
cos(23π/12) ≈ 0.97
→ (-0.26, 0.97)
```

**ทำไมใกล้กัน?**
```python
# Distance between 23:00 and 00:00
distance = sqrt((0.0 - (-0.26))² + (1.0 - 0.97)²)
         = sqrt(0.26² + 0.03²)
         = sqrt(0.068 + 0.001)
         = 0.26  # ใกล้! ✅

# ถ้าใช้ตัวเลขธรรมดา
distance = |0 - 23| = 23  # ไกล! ❌
```

#### **B. Day of Week Encoding**

```python
def encode_day_of_week(dow):
    # dow: 0=Monday, 6=Sunday
    angle = 2 * np.pi * dow / 7
    
    dow_sin = np.sin(angle)
    dow_cos = np.cos(angle)
    
    return dow_sin, dow_cos
```

#### **C. Weekend Flag**

```python
# Binary feature
is_weekend = 1 if dow >= 5 else 0  # 5=Sat, 6=Sun
```

### **3. Count Features (คุณสมบัติการนับ)**

```python
# นับจำนวน probe vehicles
count_probes = len(speeds_in_window)

# ถ่วงน้ำหนักตามจำนวน
confidence = min(count_probes / 10, 1.0)
# ถ้า probes < 10 → confidence ต่ำ
```

### **4. Congestion Labels (ป้ายระดับความแออัด)**

**การจำแนกตามความเร็ว:**

```python
def classify_congestion(mean_speed):
    if mean_speed < 20:
        return 'gridlock'      # รถติดมาก (0)
    elif mean_speed < 40:
        return 'congested'     # รถติดปานกลาง (1)
    elif mean_speed < 60:
        return 'moderate'      # รถพอสะดวก (2)
    else:
        return 'free_flow'     # รถไหลสะดวก (3)
```

**Mapping to numbers:**
```python
congestion_map = {
    'gridlock': 0,
    'congested': 1,
    'moderate': 2,
    'free_flow': 3
}
```

**คำอธิบาย:**
- **Gridlock (0-20 km/h):** รถติดหนัก เคลื่อนที่ช้ามาก
- **Congested (20-40 km/h):** รถติด เคลื่อนที่ช้า
- **Moderate (40-60 km/h):** การจราจรปานกลาง
- **Free Flow (>60 km/h):** การจราจรสะดวก

### **5. Rush Hour Labels (ป้ายช่วงเวลาเร่งด่วน)**

```python
def is_rush_hour(hour, dow):
    # วันธรรมดา (Mon-Fri)
    if dow < 5:
        # เช้า 7:00-9:00 หรือ เย็น 17:00-19:00
        if (7 <= hour < 9) or (17 <= hour < 19):
            return 1  # Rush hour
    
    return 0  # Non-rush hour
```

**คำอธิบาย:**
- **Rush hour:** ช่วงเวลาเร่งด่วน (คนไปทำงาน/กลับบ้าน)
- **Non-rush hour:** ช่วงเวลาปกติ

---

## 📏 การคำนวณระยะทาง

### **Haversine Formula (สูตรคำนวณระยะทางบนโลก)**

**ปัญหา:** พิกัด latitude/longitude บนโลกกลม → ไม่สามารถใช้ Euclidean distance ได้

**สูตร Haversine:**

```python
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    คำนวณระยะทางบนพื้นผิวโลก
    
    Parameters:
        lat1, lon1: พิกัดจุดที่ 1 (degrees)
        lat2, lon2: พิกัดจุดที่ 2 (degrees)
    
    Returns:
        distance: ระยะทาง (kilometers)
    """
    # Radius ของโลก (km)
    R = 6371.0
    
    # แปลง degrees เป็น radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # ความแตกต่าง
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = np.sin(dlat/2)**2 + \
        np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    
    c = 2 * np.arcsin(np.sqrt(a))
    
    # ระยะทาง
    distance = R * c
    
    return distance
```

**ตัวอย่างการคำนวณ:**

```python
# จุด A: MBK Center
lat1 = 13.7447
lon1 = 100.5298

# จุด B: Siam Paragon
lat2 = 13.7467
lon2 = 100.5343

# คำนวณ
distance = haversine_distance(lat1, lon1, lat2, lon2)
print(f"Distance: {distance:.3f} km")
# Output: Distance: 0.523 km (ประมาณ 500 เมตร)
```

**ขั้นตอนการคำนวณแบบละเอียด:**

```python
# Step 1: แปลงเป็น radians
lat1_rad = 13.7447 * π/180 = 0.2399
lon1_rad = 100.5298 * π/180 = 1.7544
lat2_rad = 13.7467 * π/180 = 0.2400
lon2_rad = 100.5343 * π/180 = 1.7552

# Step 2: คำนวณความแตกต่าง
dlat = 0.2400 - 0.2399 = 0.0001
dlon = 1.7552 - 1.7544 = 0.0008

# Step 3: Haversine formula
a = sin²(dlat/2) + cos(lat1_rad) * cos(lat2_rad) * sin²(dlon/2)
  = sin²(0.00005) + cos(0.2399) * cos(0.2400) * sin²(0.0004)
  = 0.0000000025 + 0.9715 * 0.9715 * 0.00000016
  = 0.0000000025 + 0.0000001508
  = 0.0000001533

# Step 4: คำนวณ c
c = 2 * arcsin(sqrt(0.0000001533))
  = 2 * arcsin(0.000392)
  = 2 * 0.000392
  = 0.000784

# Step 5: ระยะทาง
distance = 6371 * 0.000784
         = 4.99 km
         ≈ 0.5 km ✅
```

**ทำไมต้องใช้ Haversine?**

```python
# ❌ Euclidean distance (ผิด!)
distance = sqrt((lat2-lat1)² + (lon2-lon1)²)
         = sqrt((0.002)² + (0.0045)²)
         = 0.0049  # ไม่มีความหมาย!

# ✅ Haversine (ถูกต้อง!)
distance = 0.523 km  # ระยะทางจริงบนโลก
```

### **การสร้าง Distance Matrix**

```python
# คำนวณระยะทางระหว่างสถานที่ทั้งหมด
locations = 217  # จำนวนสถานที่

distance_matrix = np.zeros((locations, locations))

for i in range(locations):
    for j in range(locations):
        if i != j:
            distance_matrix[i][j] = haversine_distance(
                lat[i], lon[i],
                lat[j], lon[j]
            )

# distance_matrix[i][j] = ระยะห่างจากสถานที่ i ไปยัง j
```

**ใช้ทำอะไร?**
- หาสถานที่ใกล้เคียง (neighbors)
- สร้าง graph edges
- คำนวณเส้นทาง

---

## 📊 การแบ่งข้อมูล

### **Train/Validation/Test Split**

```python
from sklearn.model_selection import train_test_split

# Total samples
n_samples = len(data)

# Split ratio: 70% train, 15% val, 15% test
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# Step 1: แบ่ง train + val vs test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=test_ratio,
    random_state=42,
    stratify=y  # เก็บสัดส่วน labels เท่าเดิม
)

# Step 2: แบ่ง train vs val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=val_ratio / (train_ratio + val_ratio),
    random_state=42,
    stratify=y_temp
)

print(f"Train: {len(X_train)} samples")
print(f"Val: {len(X_val)} samples")
print(f"Test: {len(X_test)} samples")
```

**ตัวอย่าง:**
```
Total: 2398 samples

Train: 1678 samples (70%)
Val:   359 samples  (15%)
Test:  361 samples  (15%)
```

**ทำไมต้องแบ่ง?**

- **Train Set:** ใช้เทรนโมเดล (โมเดลเห็นข้อมูลนี้)
- **Validation Set:** ใช้ปรับ hyperparameters (โมเดลไม่เห็น แต่เราใช้ตัดสินใจ)
- **Test Set:** ใช้ประเมินผลจริง (โมเดลและเราไม่เคยเห็น)

**Stratify คืออะไร?**

```python
# ❌ ไม่ stratify
train_labels = [0, 0, 0, 1, 1]  # 60% class 0
test_labels = [2, 2, 3, 3, 3]   # 0% class 0 → ไม่สมดุล!

# ✅ Stratify
# เก็บสัดส่วนเท่าเดิม
original = [0:40%, 1:30%, 2:20%, 3:10%]
train = [0:40%, 1:30%, 2:20%, 3:10%]  # สัดส่วนเท่าเดิม ✅
test = [0:40%, 1:30%, 2:20%, 3:10%]   # สัดส่วนเท่าเดิม ✅
```

---

## 🔄 Data Augmentation

### **1. Noise Injection (เพิ่มสัญญาณรบกวน)**

```python
def add_noise(X, noise_level=0.01):
    """
    เพิ่ม Gaussian noise
    
    Parameters:
        X: input data
        noise_level: ระดับ noise (std)
    
    Returns:
        X_noisy: data + noise
    """
    noise = np.random.randn(*X.shape) * noise_level
    X_noisy = X + noise
    
    return X_noisy
```

**ตัวอย่าง:**
```python
# Original
X = [45.5, 42.0, 5.2, 25, 0.85, ...]

# Add noise (1%)
noise = [0.3, -0.2, 0.1, 0.4, -0.01, ...]
X_noisy = [45.8, 41.8, 5.3, 25.4, 0.84, ...]

# เปลี่ยนแปลงเล็กน้อย แต่ทำให้โมเดล robust
```

**ทำไมต้องมี?**
- ข้อมูลจริงมี noise
- โมเดลเรียนรู้จัดการกับความไม่แน่นอน
- ลด overfitting

### **2. Feature Scaling**

**Standardization (Z-score normalization):**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit on training data
scaler.fit(X_train)

# Transform all sets
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

**สูตร:**
```python
X_scaled = (X - mean) / std
```

**ตัวอย่าง:**
```python
# ข้อมูลต้นฉบับ
mean_speed = [20, 30, 40, 50, 60]

# คำนวณ
mean = 40
std = 14.14

# Standardize
scaled = [(20-40)/14.14, (30-40)/14.14, (40-40)/14.14, 
          (50-40)/14.14, (60-40)/14.14]
       = [-1.41, -0.71, 0.0, 0.71, 1.41]

# Mean ≈ 0, Std ≈ 1
```

**ทำไมต้อง scale?**

```python
# ❌ ไม่ scale
feature1 = mean_speed (0-120)
feature2 = is_weekend (0-1)

# Neural network จะ "สนใจ" feature1 มากกว่า
# เพราะค่าใหญ่กว่า

# ✅ Scale
feature1_scaled = (-2 to 2)
feature2_scaled = (-1 to 1)

# ทุก features มีความสำคัญเท่ากัน
```

---

## 📊 สรุปกระบวนการ

### **Pipeline ทั้งหมด:**

```
1. Load Raw Data
   ↓
2. Clean Data (remove outliers, duplicates)
   ↓
3. Handle Missing Values (interpolate, fill)
   ↓
4. Feature Engineering
   ├─ Speed features (mean, median, std)
   ├─ Temporal features (hour_sin/cos, dow_sin/cos)
   ├─ Count features
   └─ Quality score
   ↓
5. Create Labels
   ├─ Congestion levels (0-3)
   └─ Rush hour (0-1)
   ↓
6. Calculate Distances (Haversine)
   ↓
7. Train/Val/Test Split (70/15/15)
   ↓
8. Feature Scaling (Standardization)
   ↓
9. Data Augmentation (noise injection)
   ↓
10. Ready for Training!
```

### **ตัวเลขสรุป:**

```python
# ข้อมูลดิบ
Raw data: ~100,000 GPS records/day

# หลังทำความสะอาด
Clean data: ~80,000 records/day

# หลัง aggregate (5-min windows)
Aggregated: ~2,400 samples/day

# สุดท้าย (เดือนเดียว)
Final dataset: ~2,398 samples

# แบ่งข้อมูล
Train: 1,678 samples (70%)
Val:   359 samples  (15%)
Test:  361 samples  (15%)
```

---

## 🎯 สรุป

การประมวลผลข้อมูลประกอบด้วย:

1. **ทำความสะอาด:** ลบ outliers, duplicates, missing values
2. **Feature Engineering:** สร้าง features ใหม่ที่มีประโยชน์
3. **Temporal Encoding:** ใช้ sine/cosine สำหรับเวลา
4. **Label Creation:** จำแนก congestion และ rush hour
5. **Distance Calculation:** ใช้ Haversine formula
6. **Data Split:** แบ่ง train/val/test แบบ stratified
7. **Scaling:** Standardization
8. **Augmentation:** เพิ่ม noise เพื่อ robustness

**ผลลัพธ์:** ข้อมูลสะอาด พร้อมใช้เทรนโมเดล!

อ่านเพิ่มเติม: [TRAINING_GUIDE_TH.md](./TRAINING_GUIDE_TH.md)
