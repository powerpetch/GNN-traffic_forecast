# 📘 อธิบายโค้ด: data_processor.py

## 📋 ข้อมูลไฟล์

- **ชื่อไฟล์:** `src/data/data_processor.py`
- **หน้าที่:** ประมวลผลข้อมูลการจราจรให้พร้อมสำหรับเทรนโมเดล
- **จำนวนบรรทัด:** ~530 บรรทัด
- **ภาษา:** Python
- **คลาสหลัก:** `TrafficDataProcessor`

---

## 🎯 ภาพรวม

ไฟล์นี้เป็น **Data Processing Pipeline** ที่แปลงข้อมูลดิบ (GPS probes) ให้กลายเป็นข้อมูลที่โมเดล GNN สามารถใช้ได้

### **Pipeline การประมวลผล:**

```
1. โหลดข้อมูล PROBE
   ↓
2. Map-matching กับถนน (OSM)
   ↓
3. Aggregate ข้อมูล (5 นาที)
   ↓
4. สร้าง Features (10 features)
   ↓
5. สร้าง Labels (4+2 classes)
   ↓
6. บันทึกข้อมูลที่พร้อมใช้
```

---

## 📂 โครงสร้างคลาส TrafficDataProcessor

```python
class TrafficDataProcessor:
    """คลาสหลักสำหรับประมวลผลข้อมูลการจราจร"""
    
    # Constructor
    __init__()
    
    # 1. โหลดข้อมูล
    load_probe_data()
    load_road_network()
    
    # 2. Map-matching
    map_match_to_network()
    build_spatial_index()
    find_nearest_road()
    
    # 3. Aggregate ข้อมูล
    aggregate_by_time()
    aggregate_by_location()
    
    # 4. Feature Engineering
    create_temporal_features()
    create_spatial_features()
    create_lag_features()
    create_statistical_features()
    
    # 5. Label Creation
    create_congestion_labels()
    create_rush_hour_labels()
    
    # 6. บันทึก/โหลด
    save_processed_data()
    load_processed_data()
    
    # 7. Utilities
    haversine_distance()
    validate_data()
    get_statistics()
```

---

## 1️⃣ Constructor - การเริ่มต้น

```python
def __init__(self, data_path: str = "..."):
    """
    เริ่มต้นคลาส TrafficDataProcessor
    
    Parameters:
        data_path (str): path ไปยังข้อมูลดิบ
    
    Attributes:
        self.road_network: เครือข่ายถนนจาก OSM
        self.spatial_index: R-tree index สำหรับค้นหาเร็ว
        self.processed_data: ข้อมูลที่ประมวลผลแล้ว
        self.speed_thresholds: เกณฑ์แบ่งระดับการจราจร
        self.rush_hours: ช่วงเวลาเร่งด่วน
    """
    
    self.data_path = data_path
    self.road_network = None
    self.spatial_index = None
    self.processed_data = None
    
    # เกณฑ์จำแนกการจราจร
    self.speed_thresholds = {
        'gridlock': 10,      # < 10 km/h
        'congested': 25,     # 10-25 km/h
        'moderate': 40,      # 25-40 km/h
        'free_flow': 999     # > 40 km/h
    }
    
    # ช่วงชั่วโมงเร่งด่วน
    self.rush_hours = [
        (7, 9),    # เช้า 7:00-9:00
        (17, 19)   # เย็น 17:00-19:00
    ]
```

**ตัวอย่างการใช้:**
```python
# สร้าง processor
processor = TrafficDataProcessor(
    data_path="Data/PROBE-202401"
)

print(processor.speed_thresholds)
# {'gridlock': 10, 'congested': 25, ...}
```

---

## 2️⃣ การโหลดข้อมูล

### **📥 load_probe_data() - โหลดข้อมูล GPS**

```python
def load_probe_data(self, date_pattern: str = "2024*") -> pd.DataFrame:
    """
    โหลดข้อมูล PROBE จากหลายไฟล์
    
    Parameters:
        date_pattern: pattern สำหรับหาไฟล์ (เช่น "2024*" = ทุกไฟล์ปี 2024)
    
    Returns:
        DataFrame ที่มีคอลัมน์:
        - timestamp: วันเวลา
        - latitude: ละติจูด
        - longitude: ลองจิจูด
        - speed: ความเร็ว (km/h)
        - heading: ทิศทาง (0-360°)
        - quality: คะแนนคุณภาพ (0-1)
    
    ขั้นตอน:
        1. หาทุกไฟล์ที่ตรงกับ pattern
        2. อ่านแต่ละไฟล์
        3. รวมเป็น DataFrame เดียว
        4. แปลง timestamp เป็น datetime
        5. กรองข้อมูลผิดปกติ
    """
    
    import glob
    
    # หาไฟล์ทั้งหมด
    files = glob.glob(f"{self.data_path}/{date_pattern}.csv")
    print(f"Found {len(files)} files")
    
    # อ่านทุกไฟล์
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    # รวมทุกไฟล์
    combined = pd.concat(dfs, ignore_index=True)
    
    # แปลง timestamp
    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    
    # กรองข้อมูล
    combined = combined[
        (combined['speed'] >= 0) &
        (combined['speed'] <= 150) &
        (combined['quality'] >= 0.3)
    ]
    
    return combined
```

**ตัวอย่างข้อมูล:**
```python
# โหลดข้อมูล
df = processor.load_probe_data("202401*")

print(df.head())
```

| timestamp | latitude | longitude | speed | heading | quality |
|-----------|----------|-----------|-------|---------|---------|
| 2024-01-01 00:05:00 | 13.7563 | 100.5018 | 45.5 | 90.0 | 0.85 |
| 2024-01-01 00:05:15 | 13.7565 | 100.5020 | 46.2 | 92.5 | 0.88 |
| 2024-01-01 00:05:30 | 13.7567 | 100.5022 | 44.8 | 89.0 | 0.82 |

---

### **🗺️ load_road_network() - โหลดแผนที่ถนน**

```python
def load_road_network(self, road_path: str) -> gpd.GeoDataFrame:
    """
    โหลดเครือข่ายถนนจาก HOTOSM (OpenStreetMap)
    
    Parameters:
        road_path: path ไปยังไฟล์ .gpkg (GeoPackage)
    
    Returns:
        GeoDataFrame ที่มี:
        - geometry: เส้นถนน (LineString)
        - highway: ประเภทถนน (primary, secondary, ...)
        - name: ชื่อถนน (ถ้ามี)
        - maxspeed: ความเร็วสูงสุด
    
    ขั้นตอน:
        1. อ่านไฟล์ GeoPackage
        2. เลือกเฉพาะถนนที่สำคัญ
        3. สร้าง Spatial Index (R-tree)
    """
    
    # อ่านไฟล์
    roads = gpd.read_file(road_path)
    
    # กรองเฉพาะถนนสำคัญ
    important_roads = [
        'motorway', 'trunk', 'primary', 
        'secondary', 'tertiary'
    ]
    roads = roads[roads['highway'].isin(important_roads)]
    
    # เก็บไว้
    self.road_network = roads
    
    # สร้าง spatial index
    self.build_spatial_index()
    
    return roads
```

**ตัวอย่าง:**
```python
# โหลดถนน
roads = processor.load_road_network(
    "Data/hotosm_tha_roads_lines_gpkg/roads.gpkg"
)

print(f"Loaded {len(roads)} roads")
print(roads[['highway', 'name']].head())
```

| highway | name |
|---------|------|
| primary | ถนนพระราม 1 |
| motorway | ทางด่วนศรีรัช |
| secondary | ถนนสุขุมวิท |

---

## 3️⃣ Map-Matching - จับคู่กับถนน

### **🎯 map_match_to_network() - จับคู่ GPS กับถนน**

```python
def map_match_to_network(self, df: pd.DataFrame, 
                         max_distance: float = 100) -> pd.DataFrame:
    """
    จับคู่จุด GPS กับถนนที่ใกล้ที่สุด
    
    Parameters:
        df: DataFrame ที่มีพิกัด GPS
        max_distance: ระยะสูงสุดที่ยอมรับได้ (เมตร)
    
    Returns:
        DataFrame + คอลัมน์เพิ่ม:
        - road_id: ID ของถนนที่จับคู่
        - distance_to_road: ระยะห่างจากถนน (เมตร)
        - matched_point: จุดบนถนนที่ใกล้ที่สุด
    
    ขั้นตอน:
        1. สำหรับแต่ละจุด GPS
        2. หาถนนที่ใกล้ที่สุด (ใช้ R-tree)
        3. คำนวณระยะห่าง
        4. ถ้า distance <= max_distance → จับคู่
        5. ถ้า distance > max_distance → ทิ้ง
    """
    
    matched_data = []
    
    for idx, row in df.iterrows():
        # สร้างจุด GPS
        point = Point(row['longitude'], row['latitude'])
        
        # หาถนนใกล้ที่สุด
        road_id, distance = self.find_nearest_road(point)
        
        # เช็คระยะห่าง
        if distance <= max_distance:
            row['road_id'] = road_id
            row['distance_to_road'] = distance
            matched_data.append(row)
    
    return pd.DataFrame(matched_data)
```

**ตัวอย่าง:**
```python
# ก่อน map-matching
print(df[['latitude', 'longitude', 'speed']].head())
```

| latitude | longitude | speed |
|----------|-----------|-------|
| 13.7563 | 100.5018 | 45.5 |
| 13.7565 | 100.5020 | 46.2 |

```python
# หลัง map-matching
matched = processor.map_match_to_network(df, max_distance=50)
print(matched[['latitude', 'longitude', 'speed', 'road_id', 'distance_to_road']].head())
```

| latitude | longitude | speed | road_id | distance_to_road |
|----------|-----------|-------|---------|------------------|
| 13.7563 | 100.5018 | 45.5 | ROAD_123 | 15.3 |
| 13.7565 | 100.5020 | 46.2 | ROAD_123 | 12.8 |

---

### **🔍 find_nearest_road() - หาถนนใกล้ที่สุด**

```python
def find_nearest_road(self, point: Point) -> Tuple[str, float]:
    """
    หาถนนที่ใกล้กับจุดที่สุด
    
    Parameters:
        point: จุด GPS (Point object)
    
    Returns:
        (road_id, distance): ID ถนน และระยะห่าง (เมตร)
    
    Algorithm:
        1. ใช้ R-tree spatial index ค้นหาถนนใกล้เคียง
        2. คำนวณระยะห่างแบบ Haversine
        3. เลือกถนนที่ระยะสั้นที่สุด
    """
    
    # ค้นหาถนนใกล้เคียงด้วย R-tree
    candidates = self.spatial_index.nearest(
        (point.x, point.y, point.x, point.y), 
        5  # ค้นหา 5 ถนนใกล้ที่สุด
    )
    
    min_distance = float('inf')
    nearest_road = None
    
    # หาถนนที่ใกล้ที่สุด
    for road_idx in candidates:
        road_geom = self.road_network.iloc[road_idx].geometry
        distance = point.distance(road_geom)
        
        if distance < min_distance:
            min_distance = distance
            nearest_road = self.road_network.iloc[road_idx]['road_id']
    
    # แปลงเป็นเมตร
    distance_meters = min_distance * 111000  # degrees → meters
    
    return nearest_road, distance_meters
```

**ภาพประกอบ:**
```
      GPS Point (●)
         |
         | 15 m (distance_to_road)
         |
    ═══════════════  Road (ROAD_123)
```

---

## 4️⃣ Aggregation - รวมข้อมูล

### **⏰ aggregate_by_time() - รวมตามเวลา**

```python
def aggregate_by_time(self, df: pd.DataFrame, 
                      bin_minutes: int = 5) -> pd.DataFrame:
    """
    รวมข้อมูลทุกๆ N นาที
    
    Parameters:
        df: DataFrame ที่มี timestamp และ speed
        bin_minutes: ช่วงเวลาที่ต้องการรวม (นาที)
    
    Returns:
        DataFrame ที่รวมแล้ว มีคอลัมน์:
        - time_bin: ช่วงเวลา (เช่น "08:00-08:05")
        - road_id: ID ถนน
        - count: จำนวนข้อมูล
        - mean_speed: ความเร็วเฉลี่ย
        - median_speed: ความเร็วกลาง
        - std_speed: ส่วนเบี่ยงเบน
        - min_speed: ความเร็วต่ำสุด
        - max_speed: ความเร็วสูงสุด
    
    ขั้นตอน:
        1. สร้าง time bins (ทุก 5 นาที)
        2. จัดกลุ่มตาม (time_bin, road_id)
        3. คำนวณสถิติแต่ละกลุ่ม
    """
    
    # สร้าง time bins
    df['time_bin'] = df['timestamp'].dt.floor(f'{bin_minutes}min')
    
    # จัดกลุ่มและคำนวณ
    aggregated = df.groupby(['time_bin', 'road_id']).agg({
        'speed': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'quality': 'mean'
    }).reset_index()
    
    # เปลี่ยนชื่อคอลัมน์
    aggregated.columns = [
        'time_bin', 'road_id', 'count', 
        'mean_speed', 'median_speed', 'std_speed',
        'min_speed', 'max_speed', 'mean_quality'
    ]
    
    # กรองเฉพาะที่มีข้อมูลเพียงพอ
    aggregated = aggregated[aggregated['count'] >= 3]
    
    return aggregated
```

**ตัวอย่าง:**

**ข้อมูลดิบ (ทุกวินาที):**
```python
timestamp            road_id  speed
08:00:00            ROAD_1   45
08:00:30            ROAD_1   47
08:01:15            ROAD_1   43
08:02:00            ROAD_1   46
08:03:30            ROAD_1   44
```

**หลัง Aggregate (ทุก 5 นาที):**
```python
time_bin         road_id  count  mean_speed  median_speed  std_speed
08:00-08:05     ROAD_1   5      45.0        45.0          1.58
```

---

### **📍 aggregate_by_location() - รวมตามสถานที่**

```python
def aggregate_by_location(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    รวมข้อมูลของสถานที่ใกล้เคียง
    
    Parameters:
        df: DataFrame ที่มีพิกัด
    
    Returns:
        DataFrame ที่รวมตามพื้นที่แล้ว
    
    ขั้นตอน:
        1. แบ่งกรุงเทพเป็น grid (เช่น 0.01° × 0.01°)
        2. จัดจุดแต่ละจุดเข้า grid
        3. รวมข้อมูลใน grid เดียวกัน
    """
    
    # สร้าง grid
    df['grid_lat'] = (df['latitude'] / 0.01).round() * 0.01
    df['grid_lon'] = (df['longitude'] / 0.01).round() * 0.01
    df['location_id'] = df['grid_lat'].astype(str) + '_' + df['grid_lon'].astype(str)
    
    # จัดกลุ่ม
    aggregated = df.groupby(['time_bin', 'location_id']).agg({
        'speed': ['mean', 'std'],
        'count': 'sum'
    })
    
    return aggregated
```

**ตัวอย่าง:**
```
Grid: 0.01° × 0.01° (≈ 1.1 km × 1.1 km)

┌─────────┬─────────┬─────────┐
│ (13.75, │ (13.75, │ (13.75, │
│ 100.50) │ 100.51) │ 100.52) │
│ 10 pts  │ 15 pts  │ 8 pts   │
├─────────┼─────────┼─────────┤
│ (13.74, │ (13.74, │ (13.74, │
│ 100.50) │ 100.51) │ 100.52) │
│ 12 pts  │ 20 pts  │ 5 pts   │
└─────────┴─────────┴─────────┘
```

---

## 5️⃣ Feature Engineering - สร้างคุณสมบัติ

### **⏰ create_temporal_features() - Features เวลา**

```python
def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    สร้าง features ที่เกี่ยวกับเวลา
    
    Features ที่สร้าง:
        1. hour_sin: sine encoding ของชั่วโมง
        2. hour_cos: cosine encoding ของชั่วโมง
        3. day_of_week: วันในสัปดาห์ (0-6)
        4. is_weekend: วันหยุดหรือไม่ (0/1)
        5. is_holiday: วันหยุดนักขัตฤกษ์ (0/1)
        6. time_since_rush_hour: เวลาห่างจากชั่วโมงเร่งด่วน (ชม.)
    
    Cyclical Encoding:
        ทำไมต้อง encode เป็น sin/cos?
        - เวลาเป็นวงกลม: 23:00 ใกล้กับ 00:00
        - ถ้าใช้ 0-23 ตรงๆ → โมเดลคิดว่า 23 ห่างจาก 0 มาก
        - ใช้ sin/cos → โมเดลรู้ว่าเป็นวงกลม
    """
    
    # แยกส่วนของเวลา
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding สำหรับชั่วโมง
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # เวลาห่างจากชั่วโมงเร่งด่วน
    df['time_since_rush_hour'] = df.apply(
        lambda row: self._calculate_time_since_rush(row['hour']), 
        axis=1
    )
    
    return df
```

**ตัวอย่าง Cyclical Encoding:**

| hour | hour_sin | hour_cos | ภาพประกอบ |
|------|----------|----------|-----------|
| 0 | 0.000 | 1.000 | 12 o'clock (บน) |
| 6 | 1.000 | 0.000 | 3 o'clock (ขวา) |
| 12 | 0.000 | -1.000 | 6 o'clock (ล่าง) |
| 18 | -1.000 | 0.000 | 9 o'clock (ซ้าย) |
| 23 | -0.259 | 0.966 | ใกล้ 12 |

**วิธีคำนวณ:**
```python
hour = 14  # 14:00
angle = 2 * π * 14 / 24 = 3.665 radians

hour_sin = sin(3.665) = -0.259
hour_cos = cos(3.665) = -0.966
```

---

### **📊 create_statistical_features() - Features สถิติ**

```python
def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    สร้าง features ทางสถิติ
    
    Features:
        1. speed_mean: ความเร็วเฉลี่ย
        2. speed_median: ความเร็วกลาง
        3. speed_std: ส่วนเบี่ยงเบนความเร็ว
        4. speed_percentile_25: Percentile 25
        5. speed_percentile_75: Percentile 75
        6. speed_range: ช่วงความเร็ว (max - min)
        7. speed_cv: Coefficient of Variation (std/mean)
    """
    
    # สถิติพื้นฐาน
    df['speed_mean'] = df['speed'].mean()
    df['speed_median'] = df['speed'].median()
    df['speed_std'] = df['speed'].std()
    
    # Percentiles
    df['speed_p25'] = df['speed'].quantile(0.25)
    df['speed_p75'] = df['speed'].quantile(0.75)
    
    # ค่าอื่นๆ
    df['speed_range'] = df['speed'].max() - df['speed'].min()
    df['speed_cv'] = df['speed_std'] / (df['speed_mean'] + 1e-6)
    
    return df
```

**ตัวอย่าง:**
```python
# ข้อมูลความเร็ว 5 นาที
speeds = [42, 45, 43, 46, 44, 47, 41, 45, 43, 46]

# Features ที่ได้
speed_mean = 44.2 km/h
speed_median = 44.5 km/h
speed_std = 1.93 km/h
speed_p25 = 43.0 km/h
speed_p75 = 46.0 km/h
speed_range = 6.0 km/h (47-41)
speed_cv = 0.044 (1.93/44.2)
```

**Coefficient of Variation (CV) อธิบาย:**
- CV = std / mean
- CV ต่ำ (< 0.2) = เสถียร (ความเร็วคงที่)
- CV สูง (> 0.5) = ผันแปร (ความเร็วกระโดด)

```python
# เสถียร
speeds = [44, 45, 44, 45, 44]
CV = 0.5 / 44.4 = 0.011  (เสถียรมาก)

# ผันแปร
speeds = [20, 60, 15, 70, 10]
CV = 25.5 / 35 = 0.729  (ผันแปรมาก)
```

---

### **🔄 create_lag_features() - Features ย้อนหลัง**

```python
def create_lag_features(self, df: pd.DataFrame, 
                       lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
    """
    สร้าง lag features (ข้อมูลช่วงก่อนหน้า)
    
    Parameters:
        lags: จำนวน time steps ย้อนหลัง
              [1, 2, 3] = ย้อน 5, 10, 15 นาที
    
    Features:
        - speed_lag_1: ความเร็ว 5 นาทีก่อน
        - speed_lag_2: ความเร็ว 10 นาทีก่อน
        - speed_lag_3: ความเร็ว 15 นาทีก่อน
        - speed_diff_1: ผลต่างความเร็ว (current - lag_1)
        - speed_trend: แนวโน้ม (เพิ่ม/ลด)
    """
    
    # เรียงตามเวลา
    df = df.sort_values(['road_id', 'time_bin'])
    
    # สร้าง lag features
    for lag in lags:
        df[f'speed_lag_{lag}'] = df.groupby('road_id')['speed'].shift(lag)
        df[f'speed_diff_{lag}'] = df['speed'] - df[f'speed_lag_{lag}']
    
    # แนวโน้ม
    df['speed_trend'] = np.sign(df['speed_diff_1'])
    
    # เติมค่าที่หายไป
    df = df.fillna(method='bfill')
    
    return df
```

**ตัวอย่าง:**

| time_bin | speed | speed_lag_1 | speed_lag_2 | speed_diff_1 | speed_trend |
|----------|-------|-------------|-------------|--------------|-------------|
| 08:00 | 45 | NaN | NaN | NaN | 0 |
| 08:05 | 48 | 45 | NaN | +3 | +1 |
| 08:10 | 43 | 48 | 45 | -5 | -1 |
| 08:15 | 47 | 43 | 48 | +4 | +1 |

**ประโยชน์ของ Lag Features:**
- โมเดล "รู้" ว่าเกิดอะไรขึ้นก่อนหน้า
- จับแนวโน้ม (กำลังติดขัด/คล่องตัว)
- ทำนายแม่นยำขึ้น

---

### **🌐 create_spatial_features() - Features พื้นที่**

```python
def create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    สร้าง features เกี่ยวกับพื้นที่
    
    Features:
        1. nearby_avg_speed: ความเร็วเฉลี่ยบริเวณใกล้เคียง
        2. nearby_congestion: ระดับการจราจรบริเวณใกล้เคียง
        3. distance_to_center: ระยะห่างจากใจกลางเมือง
        4. road_density: ความหนาแน่นของถนน
        5. is_main_road: ถนนสายหลักหรือไม่
    """
    
    # หาสถานที่ใกล้เคียง (ภายใน 500 เมตร)
    df['nearby_avg_speed'] = df.apply(
        lambda row: self._get_nearby_speed(
            row['latitude'], 
            row['longitude'], 
            radius=500
        ),
        axis=1
    )
    
    # ระยะห่างจากใจกลางเมือง (สยาม)
    center = (13.7465, 100.5326)  # Siam Square
    df['distance_to_center'] = df.apply(
        lambda row: self.haversine_distance(
            row['latitude'], row['longitude'],
            center[0], center[1]
        ),
        axis=1
    )
    
    # ถนนสายหลัก
    main_roads = ['motorway', 'trunk', 'primary']
    df['is_main_road'] = df['highway'].isin(main_roads).astype(int)
    
    return df
```

**ตัวอย่าง:**

| location | speed | nearby_avg_speed | distance_to_center | is_main_road |
|----------|-------|------------------|--------------------|--------------|
| Siam | 45 | 43.5 | 0 km | 1 |
| MBK | 42 | 43.5 | 0.5 km | 1 |
| Silom | 38 | 35.2 | 2.3 km | 1 |
| Sukhumvit | 50 | 48.7 | 3.5 km | 1 |

---

## 6️⃣ Label Creation - สร้างป้ายกำกับ

### **🚦 create_congestion_labels() - ป้ายการจราจร**

```python
def create_congestion_labels(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    สร้าง labels สำหรับระดับการจราจร (4 classes)
    
    Classes:
        0: Gridlock    (< 10 km/h)    ติดขัดรุนแรง
        1: Congested   (10-25 km/h)   ติดขัด
        2: Moderate    (25-40 km/h)   ปานกลาง
        3: Free Flow   (> 40 km/h)    คล่องตัว
    
    Returns:
        DataFrame + คอลัมน์:
        - congestion_label: class (0-3)
        - congestion_name: ชื่อ class
    """
    
    def classify_speed(speed):
        if speed < 10:
            return 0, "Gridlock"
        elif speed < 25:
            return 1, "Congested"
        elif speed < 40:
            return 2, "Moderate"
        else:
            return 3, "Free Flow"
    
    # สร้าง labels
    df[['congestion_label', 'congestion_name']] = df['mean_speed'].apply(
        lambda x: pd.Series(classify_speed(x))
    )
    
    return df
```

**ตัวอย่าง:**

| time_bin | mean_speed | congestion_label | congestion_name |
|----------|------------|------------------|-----------------|
| 08:00 | 8 | 0 | Gridlock |
| 08:05 | 18 | 1 | Congested |
| 08:10 | 32 | 2 | Moderate |
| 08:15 | 55 | 3 | Free Flow |

**การกระจายของ Classes:**
```python
# นับจำนวนแต่ละ class
print(df['congestion_label'].value_counts())

# Output:
# 3 (Free Flow)    45,230  (45%)
# 2 (Moderate)     28,150  (28%)
# 1 (Congested)    18,420  (18%)
# 0 (Gridlock)      8,200  (8%)
```

---

### **⏰ create_rush_hour_labels() - ป้ายชั่วโมงเร่งด่วน**

```python
def create_rush_hour_labels(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    สร้าง labels สำหรับชั่วโมงเร่งด่วน (2 classes)
    
    Classes:
        0: Non-Rush Hour   เวลาปกติ
        1: Rush Hour       ชั่วโมงเร่งด่วน (7-9, 17-19 น.)
    
    เงื่อนไข:
        - ต้องเป็นวันธรรมดา (จันทร์-ศุกร์)
        - อยู่ในช่วงเวลา 7-9 หรือ 17-19 น.
    """
    
    def is_rush_hour(row):
        # เช็ควันหยุด
        if row['is_weekend'] == 1:
            return 0, "Non-Rush Hour"
        
        hour = row['hour']
        
        # เช็คช่วงเวลา
        for start, end in self.rush_hours:
            if start <= hour < end:
                return 1, "Rush Hour"
        
        return 0, "Non-Rush Hour"
    
    # สร้าง labels
    df[['rush_hour_label', 'rush_hour_name']] = df.apply(
        lambda row: pd.Series(is_rush_hour(row)),
        axis=1
    )
    
    return df
```

**ตัวอย่าง:**

| timestamp | hour | is_weekend | rush_hour_label | rush_hour_name |
|-----------|------|------------|-----------------|----------------|
| 2024-01-15 08:00 | 8 | 0 | 1 | Rush Hour |
| 2024-01-15 12:00 | 12 | 0 | 0 | Non-Rush Hour |
| 2024-01-15 18:00 | 18 | 0 | 1 | Rush Hour |
| 2024-01-20 08:00 | 8 | 1 | 0 | Non-Rush Hour (วันหยุด) |

**การกระจาย:**
```python
print(df['rush_hour_label'].value_counts())

# Output:
# 0 (Non-Rush)  75,840  (76%)
# 1 (Rush Hour) 24,160  (24%)
```

---

## 7️⃣ บันทึกและโหลดข้อมูล

### **💾 save_processed_data() - บันทึก**

```python
def save_processed_data(self, df: pd.DataFrame, 
                        filepath: str = "outputs/processed_data.pkl"):
    """
    บันทึกข้อมูลที่ประมวลผลแล้ว
    
    Parameters:
        df: DataFrame ที่พร้อมใช้
        filepath: ที่อยู่ไฟล์
    
    รูปแบบ:
        - .pkl: Pickle (เร็ว, เก็บได้ครบ)
        - .csv: CSV (อ่านง่าย, แต่ช้า)
        - .parquet: Parquet (เร็ว, บีบอัดดี)
    """
    
    import pickle
    
    # บันทึก metadata
    metadata = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'created_at': pd.Timestamp.now()
    }
    
    # บันทึกทั้งข้อมูลและ metadata
    with open(filepath, 'wb') as f:
        pickle.dump({
            'data': df,
            'metadata': metadata
        }, f)
    
    print(f"✅ Saved {df.shape[0]} samples to {filepath}")
```

**ตัวอย่าง:**
```python
# บันทึก
processor.save_processed_data(
    processed_df,
    "outputs/processed_data_202401.pkl"
)

# Output:
# ✅ Saved 125,430 samples to outputs/processed_data_202401.pkl
# File size: 45.3 MB
```

---

### **📂 load_processed_data() - โหลด**

```python
def load_processed_data(self, filepath: str) -> pd.DataFrame:
    """
    โหลดข้อมูลที่เคยประมวลผลไว้
    
    Returns:
        DataFrame พร้อมใช้
    """
    
    import pickle
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    df = data['data']
    metadata = data['metadata']
    
    print(f"✅ Loaded {df.shape[0]} samples")
    print(f"   Created: {metadata['created_at']}")
    print(f"   Columns: {len(metadata['columns'])}")
    
    return df
```

**ตัวอย่าง:**
```python
# โหลด
df = processor.load_processed_data("outputs/processed_data_202401.pkl")

# Output:
# ✅ Loaded 125,430 samples
#    Created: 2024-10-05 08:30:45
#    Columns: 18
```

---

## 8️⃣ Utility Functions - ฟังก์ชันช่วยเหลือ

### **📏 haversine_distance() - คำนวณระยะทาง**

```python
def haversine_distance(self, lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
    """
    คำนวณระยะทางบนโลก (Great Circle Distance)
    
    Parameters:
        lat1, lon1: พิกัดจุดที่ 1
        lat2, lon2: พิกัดจุดที่ 2
    
    Returns:
        ระยะทาง (กิโลเมตร)
    
    สูตร Haversine:
        a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
        c = 2 × arcsin(√a)
        distance = R × c  (R = 6371 km)
    """
    
    # แปลงเป็น radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # คำนวณผลต่าง
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # สูตร Haversine
    a = np.sin(dlat/2)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # ระยะทาง (km)
    R = 6371  # รัศมีโลก
    distance = R * c
    
    return distance
```

**ตัวอย่าง:**
```python
# Siam Square → MBK Center
distance = processor.haversine_distance(
    13.7465, 100.5326,  # Siam
    13.7443, 100.5300   # MBK
)
print(f"Distance: {distance:.3f} km")
# Output: Distance: 0.323 km
```

---

### **✅ validate_data() - ตรวจสอบข้อมูล**

```python
def validate_data(self, df: pd.DataFrame) -> Dict:
    """
    ตรวจสอบคุณภาพข้อมูล
    
    Returns:
        dict ที่มี:
        - valid: ข้อมูลถูกต้องหรือไม่
        - errors: รายการข้อผิดพลาด
        - warnings: รายการคำเตือน
        - statistics: สถิติข้อมูล
    """
    
    errors = []
    warnings = []
    
    # เช็คคอลัมน์ที่จำเป็น
    required_columns = [
        'timestamp', 'latitude', 'longitude', 
        'speed', 'road_id'
    ]
    missing = set(required_columns) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")
    
    # เช็คค่าว่าง
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        warnings.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
    
    # เช็คค่าผิดปกติ
    if (df['speed'] < 0).any():
        errors.append("Negative speed values found")
    
    if (df['speed'] > 150).any():
        warnings.append(f"Very high speeds: {df[df['speed'] > 150]['speed'].max()} km/h")
    
    # สถิติ
    statistics = {
        'total_samples': len(df),
        'date_range': (df['timestamp'].min(), df['timestamp'].max()),
        'unique_locations': df['road_id'].nunique(),
        'avg_speed': df['speed'].mean(),
        'speed_range': (df['speed'].min(), df['speed'].max())
    }
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'statistics': statistics
    }
```

**ตัวอย่าง:**
```python
validation = processor.validate_data(df)

if validation['valid']:
    print("✅ Data is valid!")
else:
    print("❌ Errors found:")
    for error in validation['errors']:
        print(f"   - {error}")

print("\n📊 Statistics:")
for key, value in validation['statistics'].items():
    print(f"   {key}: {value}")
```

---

## 📊 ตัวอย่างการใช้งานทั้งหมด

```python
# 1. สร้าง processor
processor = TrafficDataProcessor(
    data_path="Data/PROBE-202401"
)

# 2. โหลดข้อมูล
print("Loading probe data...")
probe_data = processor.load_probe_data("202401*")
print(f"Loaded {len(probe_data)} GPS points")

# 3. โหลดแผนที่ถนน
print("Loading road network...")
roads = processor.load_road_network(
    "Data/hotosm_tha_roads_lines_gpkg/roads.gpkg"
)

# 4. Map-matching
print("Map-matching...")
matched = processor.map_match_to_network(probe_data, max_distance=50)

# 5. Aggregate ข้อมูล
print("Aggregating...")
aggregated = processor.aggregate_by_time(matched, bin_minutes=5)

# 6. สร้าง features
print("Creating features...")
features = processor.create_temporal_features(aggregated)
features = processor.create_statistical_features(features)
features = processor.create_lag_features(features, lags=[1, 2, 3])
features = processor.create_spatial_features(features)

# 7. สร้าง labels
print("Creating labels...")
final = processor.create_congestion_labels(features)
final = processor.create_rush_hour_labels(final)

# 8. ตรวจสอบข้อมูล
print("Validating...")
validation = processor.validate_data(final)
print(f"Valid: {validation['valid']}")

# 9. บันทึก
print("Saving...")
processor.save_processed_data(
    final,
    "outputs/processed_data_202401.pkl"
)

print("✅ Done!")
```

**Output:**
```
Loading probe data...
Loaded 2,450,320 GPS points

Loading road network...
Loaded 18,450 roads

Map-matching...
Matched 2,348,215 points (95.8%)

Aggregating...
Created 125,430 time bins

Creating features...
✓ Temporal features (6)
✓ Statistical features (7)
✓ Lag features (9)
✓ Spatial features (5)

Creating labels...
✓ Congestion labels (4 classes)
✓ Rush hour labels (2 classes)

Validating...
Valid: True

Saving...
✅ Saved 125,430 samples to outputs/processed_data_202401.pkl

✅ Done!
```

---

## 📈 สรุปฟังก์ชันทั้งหมด

| ฟังก์ชัน | หน้าที่ | Input | Output |
|---------|---------|-------|--------|
| **load_probe_data()** | โหลด GPS | Files | DataFrame |
| **load_road_network()** | โหลดถนน | GeoPackage | GeoDataFrame |
| **map_match_to_network()** | จับคู่กับถนน | GPS points | Matched points |
| **aggregate_by_time()** | รวมตามเวลา | Raw data | 5-min bins |
| **create_temporal_features()** | Features เวลา | DataFrame | +6 columns |
| **create_statistical_features()** | Features สถิติ | DataFrame | +7 columns |
| **create_lag_features()** | Features ย้อนหลัง | DataFrame | +9 columns |
| **create_spatial_features()** | Features พื้นที่ | DataFrame | +5 columns |
| **create_congestion_labels()** | Labels การจราจร | DataFrame | +2 columns |
| **create_rush_hour_labels()** | Labels ชั่วโมงเร่งด่วน | DataFrame | +2 columns |
| **save_processed_data()** | บันทึก | DataFrame | .pkl file |
| **load_processed_data()** | โหลด | .pkl file | DataFrame |

---

## 🎓 ข้อมูลที่ได้ในท้ายสุด

**DataFrame สุดท้าย มี 36 คอลัมน์:**

### **Raw Data (6 คอลัมน์):**
1. timestamp
2. latitude
3. longitude
4. speed
5. heading
6. quality

### **Map-Matched (2 คอลัมน์):**
7. road_id
8. distance_to_road

### **Temporal Features (6 คอลัมน์):**
9. hour_sin
10. hour_cos
11. day_of_week
12. is_weekend
13. is_holiday
14. time_since_rush_hour

### **Statistical Features (7 คอลัมน์):**
15. speed_mean
16. speed_median
17. speed_std
18. speed_p25
19. speed_p75
20. speed_range
21. speed_cv

### **Lag Features (9 คอลัมน์):**
22-24. speed_lag_1/2/3
25-27. speed_diff_1/2/3
28-30. congestion_lag_1/2/3

### **Spatial Features (5 คอลัมน์):**
31. nearby_avg_speed
32. nearby_congestion
33. distance_to_center
34. road_density
35. is_main_road

### **Labels (4 คอลัมน์):**
36. congestion_label (0-3)
37. congestion_name
38. rush_hour_label (0-1)
39. rush_hour_name

---

**สร้างเมื่อ:** 5 ตุลาคม 2025  
**เวอร์ชัน:** 1.0  
**ผู้เขียน:** Traffic GNN Classification Team
