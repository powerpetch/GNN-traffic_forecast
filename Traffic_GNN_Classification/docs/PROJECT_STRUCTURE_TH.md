# 📂 โครงสร้างโปรเจคและหน้าที่ของแต่ละไฟล์

## 🎯 ภาพรวม

โปรเจค Traffic GNN Classification แบ่งออกเป็น **6 ส่วนหลัก**:

```
Traffic_GNN_Classification/
├── 📁 src/          → โค้ดหลักของระบบ (ประมวลผล + โมเดล)
├── 📁 app/          → Dashboard แสดงผล
├── 📁 Data/         → ข้อมูลดิบ
├── 📁 outputs/      → ผลลัพธ์จากการเทรน
├── 📁 docs/         → เอกสารประกอบ
└── 📜 Scripts       → ไฟล์เทรนและเปรียบเทียบ
```

---

## 1️⃣ โฟลเดอร์ `src/` - โค้ดหลักของระบบ

### **📁 src/data/ - จัดการข้อมูล**

#### **📄 `data_processor.py`**
**หน้าที่:** ประมวลผลข้อมูลดิบให้พร้อมใช้

**ส่วนประกอบ:**

```python
class TrafficDataProcessor:
    """คลาสหลักสำหรับประมวลผลข้อมูลการจราจร"""
    
    # 1. โหลดข้อมูล
    def load_probe_data(self, file_path):
        """
        หน้าที่: อ่านไฟล์ PROBE data (CSV)
        Input: path ไฟล์ (เช่น PROBE-202401/20240101.csv.out)
        Output: DataFrame ที่มีคอลัมน์:
            - timestamp
            - location_id
            - latitude, longitude
            - speed
            - heading
            - quality
        
        วิธีทำงาน:
        1. อ่าน CSV file
        2. แปลง timestamp เป็น datetime
        3. กรองข้อมูลที่ผิดปกติ
        """
    
    # 2. ทำความสะอาดข้อมูล
    def clean_data(self, df):
        """
        หน้าที่: ลบข้อมูลที่ผิดพลาด
        
        ทำอะไร:
        1. ลบความเร็วติดลบหรือสูงเกินไป (>150 km/h)
        2. ลบพิกัดนอกกรุงเทพฯ
        3. ลบข้อมูลซ้ำ
        4. เติมค่าที่หายไป (interpolation)
        
        ตัวอย่าง:
        Before: [45, -10, 200, 40, 35]
        After:  [45, 40, 40, 40, 35]  # ลบ/เติมค่าผิดปกติ
        """
    
    # 3. สร้าง Features
    def create_features(self, df):
        """
        หน้าที่: สร้างคุณสมบัติ (features) สำหรับโมเดล
        
        Features ที่สร้าง (10 features):
        
        A. Speed Features (3 features):
           - mean_speed: ความเร็วเฉลี่ย
           - median_speed: ความเร็วกลาง
           - speed_std: ส่วนเบี่ยงเบนของความเร็ว
           
           คำนวณจาก: speeds ในช่วง 5 นาที
           
        B. Count Features (1 feature):
           - count_probes: จำนวนรถที่สำรวจได้
           
        C. Quality Feature (1 feature):
           - quality_score: คะแนนคุณภาพข้อมูล (0-1)
           
        D. Temporal Features (5 features):
           - hour_sin, hour_cos: ชั่วโมง encode ด้วย sine/cosine
           - dow_sin, dow_cos: วัน encode ด้วย sine/cosine
           - is_weekend: วันหยุด (0/1)
           
           ตัวอย่าง hour encoding:
           hour = 14  # 14:00
           angle = 2π × 14 / 24 = 3.665
           hour_sin = sin(3.665) = -0.258
           hour_cos = cos(3.665) = -0.966
        """
    
    # 4. สร้าง Labels
    def create_labels(self, df):
        """
        หน้าที่: สร้างป้ายกำกับ (labels) สำหรับเทรน
        
        A. Congestion Label (4 classes):
           if mean_speed < 20:  → 0 (Gridlock)
           elif mean_speed < 40: → 1 (Congested)
           elif mean_speed < 60: → 2 (Moderate)
           else:                 → 3 (Free Flow)
        
        B. Rush Hour Label (2 classes):
           if (7 <= hour < 9) or (17 <= hour < 19):
               and is_weekday:
               → 1 (Rush Hour)
           else:
               → 0 (Non-Rush Hour)
        """
    
    # 5. บันทึกและโหลดข้อมูล
    def save_processed_data(self, data, path):
        """บันทึกข้อมูลที่ประมวลผลแล้วเป็น .pkl"""
    
    def load_processed_data(self, path):
        """โหลดข้อมูลที่เคยประมวลผลไว้"""
```

**สรุปหน้าที่:**
- 📥 **โหลดข้อมูล** จาก PROBE files
- 🧹 **ทำความสะอาด** ลบข้อมูลผิดพลาด
- 🔧 **สร้าง Features** 10 features สำหรับโมเดล
- 🏷️ **สร้าง Labels** congestion + rush hour
- 💾 **บันทึก/โหลด** ข้อมูลที่ประมวลผลแล้ว

---

### **📁 src/models/ - โมเดล Neural Network**

#### **📄 `multi_task_gnn.py`**
**หน้าที่:** โมเดล GNN สำหรับทำนาย

**ส่วนประกอบ:**

```python
# โมเดลที่ 1: Simple Model (พื้นฐาน)
class SimpleMultiTaskGNN(torch.nn.Module):
    """
    โมเดล GNN แบบง่าย สำหรับการจำแนก 2 tasks
    
    สถาปัตยกรรม:
    Input (10 features)
        ↓
    Linear Layer 1 (10 → 64)
        ↓
    ReLU Activation
        ↓
    Linear Layer 2 (64 → 64)
        ↓
    ReLU Activation
        ↓
        ├──→ Congestion Head (64 → 4)
        └──→ Rush Hour Head (64 → 2)
    """
    
    def __init__(self, num_features=10, hidden_dim=64):
        """
        Parameters:
            num_features: จำนวน input features (10)
            hidden_dim: จำนวน neurons ในชั้นซ่อน (64)
        
        Layers:
            - fc1: Linear(10, 64) + bias
              Parameters: 10×64 + 64 = 704
            
            - fc2: Linear(64, 64) + bias
              Parameters: 64×64 + 64 = 4,160
            
            - congestion_head: Linear(64, 4)
              Parameters: 64×4 + 4 = 260
            
            - rush_hour_head: Linear(64, 2)
              Parameters: 64×2 + 2 = 130
        
        Total Parameters: 704 + 4,160 + 260 + 130 = 5,254
        """
    
    def forward(self, x):
        """
        Forward pass - คำนวณผลลัพธ์
        
        Input:
            x: tensor shape (batch_size, 10)
            เช่น (32, 10) = batch 32 ตัวอย่าง
        
        ขั้นตอน:
        1. x → fc1 → (32, 64)
        2. ReLU → (32, 64) # เปลี่ยนค่าติดลบเป็น 0
        3. x → fc2 → (32, 64)
        4. ReLU → (32, 64)
        5. Branch ออกเป็น 2 ทาง:
           - congestion_head → (32, 4) # 4 classes
           - rush_hour_head → (32, 2)  # 2 classes
        
        Output:
            dict {
                'congestion_logits': (32, 4),
                'rush_hour_logits': (32, 2)
            }
        """

# โมเดลที่ 2: Enhanced Model (ขั้นสูง)
class EnhancedGNNModel(torch.nn.Module):
    """
    โมเดล GNN ขั้นสูง มีเทคนิคเพิ่มเติม
    
    สถาปัตยกรรม:
    Input (10 features)
        ↓
    [Linear 128 + BatchNorm + ReLU + Dropout] ─┐
        ↓                                       │
    [Linear 128 + BatchNorm + ReLU + Dropout] ←┘ Residual
        ↓                                       │
    [Linear 128 + BatchNorm + ReLU + Dropout] ←┘ Residual
        ↓
    Multi-Head Attention (4 heads)
        ↓
        ├──→ Deep Congestion Head (128→64→4)
        └──→ Deep Rush Hour Head (128→64→2)
    """
    
    def __init__(self, num_features=10, hidden_dim=128, dropout=0.3):
        """
        Parameters:
            hidden_dim: 128 neurons (มากกว่า Simple)
            dropout: 0.3 (ปิด 30% ของ neurons)
        
        เทคนิคที่เพิ่ม:
        1. BatchNorm - normalize ข้อมูลในแต่ละ batch
        2. Residual - เพิ่ม shortcut connections
        3. Attention - โมเดล "สนใจ" ข้อมูลสำคัญ
        4. Dropout - ป้องกัน overfitting
        5. Deep Heads - classification heads ที่ลึกขึ้น
        
        Total Parameters: ~62,000 (มากกว่า Simple 12 เท่า)
        """
    
    def forward(self, x):
        """
        Forward pass แบบขั้นสูง
        
        ขั้นตอน:
        1. Layer 1 + BatchNorm + ReLU + Dropout
        2. Layer 2 + residual connection
        3. Layer 3 + residual connection
        4. Multi-head attention (4 heads)
        5. Deep classification heads
        
        Residual connection ทำงานอย่างไร:
        out = Layer(x) + x  # เพิ่ม input กลับมา
        
        ตัวอย่าง:
        x = [1, 2, 3]
        Layer(x) = [4, 5, 6]
        out = [4, 5, 6] + [1, 2, 3] = [5, 7, 9]
        """
```

**สรุปหน้าที่:**
- 🧠 **SimpleMultiTaskGNN** - โมเดลพื้นฐาน (5K parameters)
- 🚀 **EnhancedGNNModel** - โมเดลขั้นสูง (62K parameters)
- 📊 **Multi-Task Learning** - ทำนาย 2 tasks พร้อมกัน
- ⚡ **Forward Pass** - คำนวณผลลัพธ์จาก input

---

### **📁 src/utils/ - ฟังก์ชันช่วยเหลือ**

#### **📄 `graph_constructor.py`**
**หน้าที่:** สร้างกราฟจากข้อมูลถนน

**ส่วนประกอบ:**

```python
class GraphConstructor:
    """สร้างและจัดการกราฟของเครือข่ายถนน"""
    
    def __init__(self, locations):
        """
        Parameters:
            locations: list ของ (latitude, longitude)
            เช่น [(13.7563, 100.5018), (13.7600, 100.5050), ...]
        """
    
    def build_graph(self):
        """
        หน้าที่: สร้างกราฟจากสถานที่
        
        ขั้นตอน:
        1. คำนวณระยะห่างทุกคู่ (Haversine)
        2. สร้าง edges สำหรับสถานที่ใกล้เคียง
        3. สร้าง adjacency matrix
        
        Output:
            - nodes: จำนวนสถานที่ (217)
            - edges: การเชื่อมต่อ [(0,1), (0,5), (1,2), ...]
            - distances: ระยะห่างของแต่ละ edge
        """
    
    def calculate_haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        หน้าที่: คำนวณระยะทางบนโลก
        
        สูตร Haversine:
        a = sin²(Δlat/2) + cos(lat1)×cos(lat2)×sin²(Δlon/2)
        c = 2 × arcsin(√a)
        distance = R × c  (R = 6371 km)
        
        ตัวอย่าง:
        Input: 
            lat1=13.7447, lon1=100.5298  (MBK)
            lat2=13.7467, lon2=100.5343  (Siam)
        Output:
            distance = 0.523 km
        """
    
    def find_neighbors(self, node_id, k=5):
        """
        หน้าที่: หาสถานที่ใกล้เคียง k ตัว
        
        Parameters:
            node_id: สถานที่ที่สนใจ
            k: จำนวนเพื่อนบ้านที่ต้องการ
        
        ตัวอย่าง:
        Input: node_id=0, k=5
        Output: [1, 5, 12, 23, 45]  # 5 สถานที่ใกล้ที่สุด
        """
    
    def create_adjacency_matrix(self):
        """
        หน้าที่: สร้างเมทริกซ์แสดงการเชื่อมต่อ
        
        Output: matrix (217, 217)
        [
          [0, 1, 0, 0, 1, ...],  # node 0 เชื่อมกับ 1 และ 4
          [1, 0, 1, 0, 0, ...],  # node 1 เชื่อมกับ 0 และ 2
          ...
        ]
        
        1 = เชื่อมกัน, 0 = ไม่เชื่อม
        """
```

**สรุปหน้าที่:**
- 📍 **สร้างกราฟ** จากพิกัดสถานที่
- 📏 **คำนวณระยะทาง** ด้วย Haversine
- 🔗 **หาเพื่อนบ้าน** สถานที่ใกล้เคียง
- 📊 **Adjacency Matrix** แสดงการเชื่อมต่อ

---

## 2️⃣ โฟลเดอร์ `app/` - Dashboard

#### **📄 `dashboard.py`**
**หน้าที่:** แอปพลิเคชัน Streamlit แสดงผล

**ส่วนประกอบ:**

```python
# ส่วนที่ 1: Setup
import streamlit as st
import sys
import os

# เพิ่ม path เพื่อ import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import modules
from config import *
from utils import *
from tabs import home, prediction, analysis, model_info, route_optimizer

# ส่วนที่ 2: Page Configuration
st.set_page_config(
    page_title="Traffic GNN Classification",
    page_icon="🚦",
    layout="wide"
)

# ส่วนที่ 3: Load Model
@st.cache_resource
def load_model():
    """
    หน้าที่: โหลดโมเดลที่เทรนไว้
    Cache เพื่อไม่ต้องโหลดซ้ำ
    """
    model = torch.load('outputs/best_model.pth')
    return model

# ส่วนที่ 4: Sidebar - เลือกโมเดล
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Simple GNN", "Enhanced GNN"]
)

# ส่วนที่ 5: Main Content - แท็บต่างๆ
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Home",
    "🔮 Prediction",
    "📊 Analysis",
    "🧠 Model Info",
    "🗺️ Route Optimizer"
])

with tab1:
    home.show()  # แสดงหน้าแรก

with tab2:
    prediction.show()  # แสดงหน้าทำนาย

with tab3:
    analysis.show()  # แสดงหน้าวิเคราะห์

with tab4:
    model_info.show()  # แสดงข้อมูลโมเดล

with tab5:
    route_optimizer.show()  # แสดงหน้าหาเส้นทาง
```

#### **📁 `app/tabs/` - แท็บต่างๆ**

```python
# tabs/home.py
def show():
    """
    หน้าที่: แสดงหน้าแรก
    - ภาพรวมโปรเจค
    - สถิติทั่วไป
    - Quick links
    """

# tabs/prediction.py
def show():
    """
    หน้าที่: หน้าทำนาย
    
    ทำอะไร:
    1. ให้ผู้ใช้ป้อน input:
       - ความเร็วเฉลี่ย
       - เวลา
       - วัน
    2. ส่งเข้าโมเดล
    3. แสดงผลการทำนาย:
       - ระดับความแออัด (Gridlock/Congested/Moderate/Free)
       - Rush Hour (Yes/No)
       - ความมั่นใจ (Confidence)
    """

# tabs/analysis.py
def show():
    """
    หน้าที่: วิเคราะห์ข้อมูล
    
    แสดง:
    - กราฟสถิติการจราจร
    - Heatmap ตามเวลา
    - แนวโน้มความเร็ว
    - เปรียบเทียบช่วงเวลา
    """

# tabs/model_info.py
def show():
    """
    หน้าที่: แสดงข้อมูลโมเดล
    
    แสดง:
    - สถาปัตยกรรมโมเดล
    - จำนวน parameters
    - ผลการเทรน (accuracy, loss)
    - Confusion matrix
    """

# tabs/route_optimizer.py
def show():
    """
    หน้าที่: หาเส้นทางที่ดีที่สุด
    
    ทำอะไร:
    1. ให้เลือกจุดเริ่มต้น-ปลายทาง (217 สถานที่)
    2. คำนวณระยะทาง (Haversine)
    3. ทำนายสภาพการจราจรในเส้นทาง
    4. แสดงบนแผนที่ (Folium)
    5. ประมาณเวลาเดินทาง
    """
```

**สรุปหน้าที่:**
- 🖥️ **dashboard.py** - แอปหลัก Streamlit
- 🏠 **home** - หน้าแรก
- 🔮 **prediction** - ทำนายการจราจร
- 📊 **analysis** - วิเคราะห์ข้อมูล
- 🧠 **model_info** - ข้อมูลโมเดล
- 🗺️ **route_optimizer** - หาเส้นทาง

---

## 3️⃣ โฟลเดอร์ `Data/` - ข้อมูลดิบ

```
Data/
├── PROBE-202401/           # ข้อมูล GPS มกราคม 2024
│   ├── 20240101.csv.out   # 1 ไฟล์ต่อ 1 วัน
│   ├── 20240102.csv.out
│   └── ...
│
├── PROBE-202402/           # กุมภาพันธ์ 2024
├── PROBE-202403/           # มีนาคม 2024
├── ...                     # เดือนอื่นๆ
│
├── hotosm_tha_roads_lines_gpkg/  # โครงสร้างถนนจาก OpenStreetMap
│   └── *.gpkg             # ไฟล์ GeoPackage
│
└── iTIC-Longdo-Traffic-events-2022/  # เหตุการณ์การจราจร
    ├── 01/                # มกราคม
    ├── 02/                # กุมภาพันธ์
    └── ...
```

**รูปแบบข้อมูล PROBE:**
```csv
timestamp,location_id,latitude,longitude,speed,heading,quality
2024-01-01 00:05:00,LOC001,13.7563,100.5018,45.5,90.0,0.85
```

---

## 4️⃣ โฟลเดอร์ `outputs/` - ผลลัพธ์

```
outputs/
├── 📄 processed_data.pkl              # ข้อมูลที่ประมวลผลแล้ว
│   → ใช้โดย: data_processor.py
│   → เก็บ: DataFrame พร้อม features + labels
│
├── 📄 best_model.pth                  # โมเดล Simple ที่ดีที่สุด
│   → สร้างโดย: train.py
│   → เก็บ: model weights + optimizer state
│   → ขนาด: ~50 KB
│
├── 📄 best_enhanced_model.pth         # โมเดล Enhanced ที่ดีที่สุด
│   → สร้างโดย: enhanced_train.py
│   → เก็บ: model weights + optimizer state
│   → ขนาด: ~250 KB
│
├── 📄 training_history.pkl            # ประวัติการเทรน Simple
│   → เก็บ: train_loss, val_loss, accuracy แต่ละ epoch
│
├── 📄 training_history.png            # กราฟการเทรน Simple
│   → แสดง: Loss curves, Accuracy curves
│
├── 📄 enhanced_training_history.png   # กราฟการเทรน Enhanced
│
├── 📄 enhanced_confusion_matrices.png # Confusion matrices
│   → แสดง: ผลการจำแนกแต่ละ class
│
├── 📄 model_comparison.png            # เปรียบเทียบโมเดล
│   → สร้างโดย: compare_models.py
│
└── 📄 evaluation_results.pkl          # ผลการประเมิน
    → เก็บ: predictions, targets, metrics
```

---

## 5️⃣ Scripts - ไฟล์เทรนและเครื่องมือ

### **📜 `train.py` - เทรนโมเดล Simple (ปรับปรุงแล้ว)**

**หน้าที่:** เทรนโมเดล SimpleMultiTaskGNN พร้อมการปรับปรุง 5 อย่าง

**วิธีรัน:**
```powershell
py train.py --epochs 100 --batch_size 32 --patience 20
```

**ผลลัพธ์:**
- `outputs/best_model.pth`
- `outputs/training_history.png`
- Accuracy ~98%

---

### **📜 `enhanced_train.py` - เทรนโมเดล Enhanced**

**หน้าที่:** เทรนโมเดล EnhancedGNNModel

**วิธีรัน:**
```powershell
py enhanced_train.py --epochs 100 --batch_size 64 --hidden_dim 128
```

**ผลลัพธ์:**
- `outputs/best_enhanced_model.pth`
- `outputs/enhanced_training_history.png`
- Accuracy ~98-99%

---

## 🔄 Flow การทำงานทั้งหมด

### **1. การเตรียมข้อมูล:**
```
PROBE Data (CSV)
    ↓
data_processor.py → load_probe_data()
    ↓
clean_data() → ลบข้อมูลผิดพลาด
    ↓
create_features() → สร้าง 10 features
    ↓
create_labels() → สร้าง labels
    ↓
processed_data.pkl (บันทึก)
```

### **2. การเทรนโมเดล:**
```
processed_data.pkl
    ↓
train.py → prepare_data()
    ↓
create_simple_datasets() → 70/15/15 split
    ↓
train_simple_model()
    ├─ SimpleMultiTaskGNN
    ├─ AdamW optimizer
    ├─ ReduceLROnPlateau scheduler
    └─ Training loop (100 epochs)
    ↓
best_model.pth (บันทึก)
```

### **3. การใช้งาน Dashboard:**
```
dashboard.py (เปิด)
    ↓
load_model() → โหลด best_model.pth
    ↓
User เลือก tab → show()
    ├─ Home
    ├─ Prediction → ทำนาย
    ├─ Analysis → วิเคราะห์
    ├─ Model Info → แสดงข้อมูล
    └─ Route Optimizer → หาเส้นทาง
```

---

## 📊 สรุปหน้าที่แต่ละไฟล์

| ไฟล์ | หน้าที่หลัก | Input | Output |
|------|-------------|-------|--------|
| **data_processor.py** | ประมวลผลข้อมูล | PROBE CSV | processed_data.pkl |
| **multi_task_gnn.py** | โมเดล GNN | Features (10) | Predictions (4+2) |
| **graph_constructor.py** | สร้างกราฟ | Locations | Graph structure |
| **train.py** | เทรนโมเดล Simple | processed_data | best_model.pth |
| **enhanced_train.py** | เทรนโมเดล Enhanced | processed_data | best_enhanced_model.pth |
| **compare_models.py** | เปรียบเทียบ | 2 models | Comparison plots |
| **dashboard.py** | แสดงผล Web | Model + Data | Interactive UI |

---

## 🎯 สรุปการทำงานแต่ละส่วน

### **Data (ข้อมูล):**
```
PROBE files → data_processor.py → Features + Labels
```

### **Model (โมเดล):**
```
Features → multi_task_gnn.py → Predictions
```

### **Training (การเทรน):**
```
Data → train.py/enhanced_train.py → Trained Model
```

### **Application (แอปพลิเคชัน):**
```
User Input → dashboard.py → Model → Predictions → Display
```

---

หวังว่าจะเข้าใจชัดเจนแล้วนะครับว่าแต่ละไฟล์ทำอะไร! 🎓✨
