# 🚗 Traffic GNN Classification - คู่มือฉบับภาษาไทย

## 📚 สารบัญ

1. [ภาพรวมโครงการ](#ภาพรวมโครงการ)
2. [เทคโนโลยีที่ใช้](#เทคโนโลยีที่ใช้)
3. [โครงสร้างโปรเจค](#โครงสร้างโปรเจค)
4. [วิธีการติดตั้ง](#วิธีการติดตั้ง)
5. [วิธีการใช้งาน](#วิธีการใช้งาน)
6. [เอกสารเพิ่มเติม](#เอกสารเพิ่มเติม)

---

## 📖 ภาพรวมโครงการ

### **โครงการนี้คืออะไร?**

โปรเจค **Traffic GNN Classification** เป็นระบบปัญญาประดิษฐ์ (AI) ที่ใช้ **Graph Neural Network (GNN)** ในการทำนายสภาพการจราจร โดยมีเป้าหมาย 2 อย่าง:

1. **จำแนกระดับความแออัด** (Congestion Classification)
   - แบ่งเป็น 4 ระดับ: รถติดมาก, รถติดปานกลาง, รถพอสะดวก, รถไหลสะดวก
   
2. **ทำนายช่วงเวลาเร่งด่วน** (Rush Hour Prediction)
   - ทำนายว่าเป็น Rush Hour หรือไม่

### **ทำไมต้องใช้ GNN?**

**Graph Neural Network (เครือข่ายประสาทเทียมแบบกราฟ)** เหมาะกับปัญหาการจราจรเพราะ:

1. **ถนนเป็นกราฟตามธรรมชาติ**
   - จุดตัด (Node) = สถานที่ต่างๆ
   - เส้นทาง (Edge) = ถนนเชื่อมต่อ

2. **ความสัมพันธ์เชิงพื้นที่**
   - ถนนติดกันมีผลต่อกัน
   - ถ้าถนน A ติด → ถนน B ติดตาม

3. **ข้อมูลมีโครงสร้าง**
   - แต่ละจุดมีคุณสมบัติ (ความเร็ว, เวลา)
   - ความสัมพันธ์ระหว่างจุดสำคัญ

### **ข้อมูลที่ใช้**

โปรเจคใช้ข้อมูลจาก:

1. **PROBE Data** (ข้อมูลจาก GPS ของรถ)
   - ความเร็วเฉลี่ย
   - ความเร็วสูงสุด-ต่ำสุด
   - จำนวนรถที่สำรวจได้
   - เวลาที่บันทึก

2. **OpenStreetMap (OSM)**
   - โครงสร้างถนน
   - ตำแหน่งพิกัด
   - การเชื่อมต่อของถนน

3. **ข้อมูลเหตุการณ์** (iTIC-Longdo)
   - อุบัติเหตุ
   - การปิดถนน
   - เหตุการณ์พิเศษ

---

## 🔧 เทคโนโลยีที่ใช้

### **1. Python 3.13.2**
**คืออะไร:** ภาษาโปรแกรมหลักของโปรเจค

**ทำไมใช้:** 
- รองรับไลบรารี Machine Learning ครบ
- เขียนโค้ดง่าย อ่านง่าย
- Community ใหญ่

### **2. PyTorch**
**คืออะไร:** Framework สำหรับสร้าง Neural Network

**ทำไมใช้:**
- ยืดหยุ่น ปรับแต่งได้ง่าย
- รองรับ GPU (การคำนวณเร็ว)
- เป็นมาตรฐานในงานวิจัย

**คำศัพท์สำคัญ:**
- **Tensor** = Array หลายมิติ (เหมือน numpy array)
- **Autograd** = คำนวณ gradient อัตโนมัติ
- **Module** = class สำหรับสร้างโมเดล

### **3. PyTorch Geometric (PyG)**
**คืออะไร:** ไลบรารีสำหรับ Graph Neural Networks

**ทำไมใช้:**
- มี GNN layers สำเร็จรูป
- รองรับข้อมูลกราฟ
- เร็วและมีประสิทธิภาพ

**คำศัพท์สำคัญ:**
- **Data.x** = Node features (คุณสมบัติของแต่ละจุด)
- **Data.edge_index** = ความเชื่อมโยงระหว่างจุด
- **Data.y** = Label (คำตอบที่ถูกต้อง)

### **4. Streamlit**
**คืออะไร:** Framework สำหรับสร้าง Web Dashboard

**ทำไมใช้:**
- สร้าง UI ได้เร็ว
- Interactive (โต้ตอบได้)
- ไม่ต้องเขียน HTML/CSS/JavaScript

### **5. Pandas & NumPy**
**คืออะไร:** ไลบรารีจัดการข้อมูล

**Pandas:**
- DataFrame = ตารางข้อมูล (เหมือน Excel)
- อ่าน/เขียน CSV, Excel ง่าย
- จัดการข้อมูลที่ซับซ้อน

**NumPy:**
- Array คำนวณเร็ว
- ฟังก์ชันคณิตศาสตร์
- พื้นฐานของ ML

### **6. Scikit-learn**
**คืออะไร:** ไลบรารี Machine Learning แบบดั้งเดิม

**ใช้ทำอะไร:**
- แบ่งข้อมูล Train/Val/Test
- วัดประสิทธิภาพ (Accuracy, Precision, Recall)
- Preprocessing ข้อมูล

---

## 📁 โครงสร้างโปรเจค

```
Traffic_GNN_Classification/
│
├── 📂 app/                          # แอปพลิเคชัน Dashboard
│   ├── dashboard.py                 # ไฟล์หลัก Streamlit
│   ├── config.py                    # การตั้งค่าต่างๆ
│   ├── utils.py                     # ฟังก์ชันช่วยเหลือ
│   └── tabs/                        # แท็บต่างๆ ใน Dashboard
│       ├── home.py                  # หน้าแรก
│       ├── prediction.py            # ทำนายการจราจร
│       ├── analysis.py              # วิเคราะห์ข้อมูล
│       ├── model_info.py            # ข้อมูลโมเดล
│       └── route_optimizer.py       # หาเส้นทางที่ดีที่สุด
│
├── 📂 src/                          # Source code หลัก
│   ├── 📂 data/                     # จัดการข้อมูล
│   │   ├── data_loader.py          # โหลดข้อมูล
│   │   └── data_processor.py       # ประมวลผลข้อมูล
│   │
│   ├── 📂 models/                   # โมเดล Neural Network
│   │   └── multi_task_gnn.py       # โมเดล GNN หลัก
│   │
│   └── 📂 utils/                    # ฟังก์ชันช่วยเหลือ
│       └── graph_constructor.py    # สร้างกราฟจากข้อมูล
│
├── 📂 Data/                         # ข้อมูลดิบ
│   ├── PROBE-202401/               # ข้อมูล GPS มกราคม 2024
│   ├── PROBE-202402/               # ข้อมูล GPS กุมภาพันธ์ 2024
│   ├── ...                         # ข้อมูลเดือนอื่นๆ
│   └── iTIC-Longdo-Traffic-events-2022/  # เหตุการณ์การจราจร
│
├── 📂 outputs/                      # ผลลัพธ์จากการเทรน
│   ├── best_model.pth              # โมเดลที่ดีที่สุด (Simple)
│   ├── best_enhanced_model.pth     # โมเดลที่ดีที่สุด (Enhanced)
│   ├── processed_data.pkl          # ข้อมูลที่ประมวลผลแล้ว
│   ├── training_history.png        # กราฟการเทรน
│   └── evaluation_results.pkl      # ผลการประเมิน
│
├── 📂 docs/                         # เอกสารประกอบ
│   ├── README_TH.md                # คู่มือภาษาไทย (ไฟล์นี้)
│   ├── TECHNICAL_DETAILS_TH.md     # รายละเอียดทางเทคนิค
│   ├── MODEL_ARCHITECTURE_TH.md    # สถาปัตยกรรมโมเดล
│   └── DATA_PROCESSING_TH.md       # การประมวลผลข้อมูล
│
├── 📜 train.py                      # สคริปต์เทรนโมเดล (ปรับปรุงแล้ว)
├── 📜 enhanced_train.py             # สคริปต์เทรนโมเดลขั้นสูง
├── 📜 compare_models.py             # เปรียบเทียบโมเดล
├── 📜 hyperparameter_search.py     # หาพารามิเตอร์ที่ดีที่สุด
│
├── 📜 requirements.txt              # รายการ Python packages
├── 📜 README.md                     # คู่มือภาษาอังกฤษ
│
└── 📜 *.bat                         # ไฟล์ Batch สำหรับรันง่ายๆ
    ├── START_DASHBOARD.bat         # เปิด Dashboard
    ├── RUN_TRAINING.bat            # เทรนโมเดล
    └── RUN_ENHANCED_TRAINING.bat   # เทรนโมเดลขั้นสูง
```

### **คำอธิบายโฟลเดอร์สำคัญ**

#### **📂 app/** - แอปพลิเคชัน Web
ใช้สำหรับ:
- แสดงผลการทำนาย
- วิเคราะห์การจราจร
- หาเส้นทางที่ดีที่สุด
- ดูข้อมูลโมเดล

#### **📂 src/** - โค้ดหลัก
ประกอบด้วย:
- **data_processor.py** - แปลงข้อมูลดิบให้ใช้ได้
- **multi_task_gnn.py** - โมเดล AI สำหรับทำนาย
- **graph_constructor.py** - สร้างกราฟจากข้อมูลถนน

#### **📂 outputs/** - ผลลัพธ์
เก็บ:
- โมเดลที่เทรนเสร็จ (.pth)
- กราฟแสดงผล (.png)
- ข้อมูลที่ประมวลผล (.pkl)

---

## 🚀 วิธีการติดตั้ง

### **ขั้นตอนที่ 1: ติดตั้ง Python**

1. ดาวน์โหลด Python 3.13.2 จาก [python.org](https://www.python.org/)
2. ติดตั้ง โดยเลือก **"Add Python to PATH"**
3. ตรวจสอบการติดตั้ง:
   ```powershell
   py --version
   ```
   ควรแสดง: `Python 3.13.2`

### **ขั้นตอนที่ 2: Clone โปรเจค**

```powershell
# Clone จาก GitHub
git clone https://github.com/powerpetch/GNN-traffic_forecast.git

# เข้าไปในโฟลเดอร์
cd Traffic_GNN_Classification
```

### **ขั้นตอนที่ 3: ติดตั้ง Dependencies**

```powershell
# ติดตั้งแพ็คเกจทั้งหมด
py -m pip install -r requirements.txt
```

**แพ็คเกจที่จะติดตั้ง:**
- torch (PyTorch)
- torch-geometric (GNN)
- streamlit (Dashboard)
- pandas (จัดการข้อมูล)
- numpy (คำนวณ)
- matplotlib (วาดกราฟ)
- scikit-learn (ML tools)
- และอื่นๆ

### **ขั้นตอนที่ 4: ตรวจสอบการติดตั้ง**

```powershell
# ทดสอบ import
py -c "import torch; import torch_geometric; print('OK!')"
```

ถ้าแสดง `OK!` แสดงว่าติดตั้งสำเร็จ!

---

## 💻 วิธีการใช้งาน

### **1. เทรนโมเดล (Training)**

#### **วิธีที่ 1: ใช้ไฟล์ Batch (ง่ายที่สุด)**

**เทรนโมเดลปกติ:**
```
ดับเบิลคลิก: RUN_TRAINING.bat
```

**เทรนโมเดลขั้นสูง:**
```
ดับเบิลคลิก: RUN_ENHANCED_TRAINING.bat
```

#### **วิธีที่ 2: ใช้ Command Line**

**เทรนโมเดลปกติ (Simple GNN):**
```powershell
py train.py --epochs 100 --batch_size 32 --patience 20
```

**พารามิเตอร์:**
- `--epochs 100` = เทรน 100 รอบ
- `--batch_size 32` = ประมวลผลครั้งละ 32 ตัวอย่าง
- `--patience 20` = หยุดถ้าไม่ดีขึ้นใน 20 รอบ

**เทรนโมเดลขั้นสูง (Enhanced GNN):**
```powershell
py enhanced_train.py --epochs 100 --batch_size 64 --hidden_dim 128
```

**พารามิเตอร์เพิ่มเติม:**
- `--hidden_dim 128` = ใช้ 128 neurons ในชั้นซ่อน
- `--dropout 0.3` = ป้องกัน overfitting

#### **ผลลัพธ์ที่ได้:**

หลังเทรนเสร็จ จะได้:
1. **โมเดล** → `outputs/best_model.pth`
2. **กราฟการเทรน** → `outputs/training_history.png`
3. **ผลการประเมิน** → Terminal output

**ตัวอย่างผลลัพธ์:**
```
=== Training Summary ===
Total epochs trained: 30
Best validation loss: 0.2519
Best congestion accuracy: 98.34%
Best rush hour accuracy: 99.45%
```

---

### **2. เปิด Dashboard**

#### **วิธีที่ 1: ใช้ไฟล์ Batch**
```
ดับเบิลคลิก: START_DASHBOARD.bat
```

#### **วิธีที่ 2: ใช้ Command Line**
```powershell
py -m streamlit run app/dashboard.py
```

**Dashboard จะเปิดที่:**
- Local: http://localhost:8501
- Network: http://192.168.x.x:8501

#### **ฟีเจอร์ใน Dashboard:**

1. **🏠 Home (หน้าแรก)**
   - ภาพรวมโปรเจค
   - สถิติทั่วไป
   - Quick links

2. **🔮 Prediction (ทำนาย)**
   - ป้อนข้อมูลการจราจร
   - ทำนายระดับความแออัด
   - ทำนาย Rush Hour
   - แสดงความมั่นใจ (Confidence)

3. **📊 Analysis (วิเคราะห์)**
   - กราฟสถิติการจราจร
   - แนวโน้มตามเวลา
   - Heatmap การจราจร
   - เปรียบเทียบช่วงเวลา

4. **🧠 Model Info (ข้อมูลโมเดล)**
   - สถาปัตยกรรมโมเดล
   - จำนวน Parameters
   - ผลการเทรน
   - ประสิทธิภาพ

5. **🗺️ Route Optimizer (หาเส้นทาง)**
   - เลือกจุดเริ่มต้น-ปลายทาง
   - คำนวณเส้นทางที่ดีที่สุด
   - แสดงบนแผนที่
   - ประมาณเวลาเดินทาง

---

### **3. เปรียบเทียบโมเดล**

```powershell
py compare_models.py
```

**จะเปรียบเทียบ:**
- Simple GNN vs Enhanced GNN
- ความแม่นยำ
- เวลาที่ใช้
- จำนวน Parameters

**ผลลัพธ์:**
- กราฟเปรียบเทียบ → `outputs/model_comparison.png`
- ตารางสรุป → แสดงใน Terminal

---

### **4. หาพารามิเตอร์ที่ดีที่สุด**

```powershell
# Quick search (เร็ว)
py hyperparameter_search.py --quick

# Full search (ครบถ้วน)
py hyperparameter_search.py
```

**จะทดสอบ:**
- Epochs: 50, 100, 150
- Batch size: 16, 32, 64
- Hidden dim: 64, 128, 256
- Dropout: 0.2, 0.3, 0.5
- Learning rate: 0.001, 0.0005

**ผลลัพธ์:**
- ตาราง CSV → `outputs/hyperparameter_search_results.csv`
- พารามิเตอร์ที่ดีที่สุด → แสดงใน Terminal

---

## 📚 เอกสารเพิ่มเติม

สำหรับข้อมูลเชิงลึก กรุณาอ่านเอกสารต่อไปนี้:

### **📖 คู่มือทางเทคนิค**

1. **[TECHNICAL_DETAILS_TH.md](./TECHNICAL_DETAILS_TH.md)**
   - อธิบาย Graph Neural Network
   - วิธีการทำงานของ GNN
   - Forward/Backward Propagation
   - Loss Functions
   - Optimization

2. **[MODEL_ARCHITECTURE_TH.md](./MODEL_ARCHITECTURE_TH.md)**
   - สถาปัตยกรรมโมเดลแบบละเอียด
   - SimpleMultiTaskGNN
   - EnhancedGNNModel
   - Layer ต่างๆ
   - Activation Functions

3. **[DATA_PROCESSING_TH.md](./DATA_PROCESSING_TH.md)**
   - การเตรียมข้อมูล
   - Data Cleaning
   - Feature Engineering
   - Normalization
   - Train/Val/Test Split

4. **[TRAINING_GUIDE_TH.md](./TRAINING_GUIDE_TH.md)**
   - คู่มือการเทรนโมเดล
   - Hyperparameter Tuning
   - Overfitting/Underfitting
   - Learning Rate Scheduling
   - Early Stopping

5. **[DASHBOARD_GUIDE_TH.md](./DASHBOARD_GUIDE_TH.md)**
   - คู่มือใช้งาน Dashboard
   - ฟีเจอร์ทั้งหมด
   - Tips & Tricks
   - Troubleshooting

---

## 🎓 ศัพท์เทคนิคที่ควรรู้

### **Machine Learning**

| ศัพท์ | คำอธิบาย |
|-------|----------|
| **Training** | การเทรนโมเดล = สอนให้ AI เรียนรู้จากข้อมูล |
| **Validation** | การตรวจสอบ = ทดสอบว่าโมเดลเรียนรู้ดีหรือไม่ |
| **Testing** | การทดสอบจริง = วัดประสิทธิภาพกับข้อมูลใหม่ |
| **Epoch** | รอบการเทรน = โมเดลเห็นข้อมูลทั้งหมด 1 ครั้ง |
| **Batch** | ชุดข้อมูล = ประมวลผลครั้งละกี่ตัวอย่าง |
| **Loss** | ค่าความผิดพลาด = วัดว่าโมเดลทำนายผิดแค่ไหน |
| **Accuracy** | ความแม่นยำ = ทำนายถูกกี่เปอร์เซ็นต์ |
| **Overfitting** | เรียนรู้จำเจาะจง = เก่งกับข้อมูลเทรน แต่แย่กับข้อมูลใหม่ |
| **Underfitting** | เรียนรู้ไม่พอ = ไม่เก่งทั้งข้อมูลเทรนและใหม่ |

### **Neural Networks**

| ศัพท์ | คำอธิบาย |
|-------|----------|
| **Neuron** | เซลล์ประสาท = หน่วยคำนวณพื้นฐาน |
| **Layer** | ชั้น = กลุ่มของ neurons |
| **Weight** | น้ำหนัก = ค่าที่โมเดลเรียนรู้ |
| **Bias** | ค่าชดเชย = ปรับ output ให้เหมาะสม |
| **Activation** | ฟังก์ชันกระตุ้น = ทำให้โมเดลเรียนรู้ได้ซับซ้อน |
| **Forward Pass** | การคำนวณไปข้างหน้า = ป้อนข้อมูล → ได้ผลลัพธ์ |
| **Backward Pass** | การคำนวณย้อนกลับ = คำนวณ gradient เพื่ออัปเดต |
| **Gradient** | ความชัน = บอกทิศทางที่ควรปรับ weight |

### **Graph Neural Networks**

| ศัพท์ | คำอธิบาย |
|-------|----------|
| **Graph** | กราฟ = โครงสร้างที่มี nodes และ edges |
| **Node** | จุด = จุดตัดหรือสถานที่ (ในโปรเจคนี้) |
| **Edge** | เส้น = การเชื่อมต่อระหว่าง nodes (ถนน) |
| **Node Feature** | คุณสมบัติของจุด = ข้อมูลที่จุดนั้นมี |
| **Edge Feature** | คุณสมบัติของเส้น = ข้อมูลของการเชื่อมต่อ |
| **Message Passing** | ส่งข้อความ = nodes แชร์ข้อมูลกัน |
| **Aggregation** | รวบรวม = สรุปข้อมูลจาก nodes ข้างเคียง |
| **Graph Convolution** | Convolution บนกราฟ = คล้าย CNN แต่บนกราฟ |

### **Optimization**

| ศัพท์ | คำอธิบาย |
|-------|----------|
| **Optimizer** | ตัวปรับแต่ง = วิธีอัปเดต weights (Adam, AdamW, SGD) |
| **Learning Rate** | อัตราการเรียนรู้ = ขนาดของการปรับ weight แต่ละครั้ง |
| **Scheduler** | ตัวปรับ LR = ปรับ learning rate ให้เหมาะสม |
| **Gradient Clipping** | ตัดความชัน = ป้องกัน gradient ใหญ่เกินไป |
| **Weight Decay** | ลด weight = regularization ป้องกัน overfitting |
| **Early Stopping** | หยุดก่อน = หยุดเทรนถ้าไม่ดีขึ้น |
| **Checkpoint** | จุดบันทึก = บันทึกโมเดลที่ดีที่สุด |

---

## ❓ คำถามที่พบบ่อย (FAQ)

### **Q1: ทำไมโมเดลเทรนนานมาก?**
**A:** 
- ใช้ CPU (ช้า) → ถ้ามี GPU จะเร็วกว่ามาก
- ข้อมูลเยอะ → ลดด้วย `--batch_size` ที่ใหญ่ขึ้น
- Epochs เยอะ → ลด epochs หรือใช้ early stopping

### **Q2: Accuracy ต่ำ ทำอย่างไร?**
**A:**
- เทรนนานขึ้น → เพิ่ม `--epochs`
- โมเดลใหญ่ขึ้น → เพิ่ม `--hidden_dim`
- ปรับ learning rate → ใช้ hyperparameter search
- ตรวจสอบข้อมูล → อาจมีข้อมูลคุณภาพต่ำ

### **Q3: Dashboard เปิดไม่ได้?**
**A:**
```powershell
# ตรวจสอบ streamlit
py -m pip install streamlit --upgrade

# เปิดด้วย command
py -m streamlit run app/dashboard.py
```

### **Q4: ต้องการข้อมูลใหม่ ทำอย่างไร?**
**A:**
- ใส่ไฟล์ข้อมูลใน `Data/`
- รัน `--force_reprocess` เพื่อประมวลผลใหม่
```powershell
py train.py --force_reprocess
```

### **Q5: เปรียบเทียบ Simple vs Enhanced อย่างไร?**
**A:**
```powershell
# เทรนทั้งสองโมเดล
py train.py --epochs 100
py enhanced_train.py --epochs 100

# เปรียบเทียบ
py compare_models.py
```

---

## 📞 การติดต่อ & สนับสนุน

- **GitHub Issues:** [Report bugs](https://github.com/powerpetch/GNN-traffic_forecast/issues)
- **Documentation:** อ่านเอกสารใน `docs/`
- **Email:** powerpetch05@gmail.com

---

## 🎉 สรุป

โปรเจค **Traffic GNN Classification** เป็นระบบ AI ที่:

✅ ใช้ **Graph Neural Network** ทำนายการจราจร  
✅ มีความแม่นยำสูง (98%+)  
✅ รองรับ **Multi-Task Learning** (2 งานพร้อมกัน)  
✅ มี **Dashboard** ใช้งานง่าย  
✅ มี **เอกสารภาษาไทย** ครบถ้วน  

**เริ่มต้นง่ายๆ:**
1. ติดตั้ง Python & Dependencies
2. รัน `RUN_TRAINING.bat`
3. รัน `START_DASHBOARD.bat`
4. เริ่มใช้งาน!

**อ่านเพิ่มเติม:**
- [รายละเอียดทางเทคนิค](./TECHNICAL_DETAILS_TH.md)
- [สถาปัตยกรรมโมเดล](./MODEL_ARCHITECTURE_TH.md)
- [การประมวลผลข้อมูล](./DATA_PROCESSING_TH.md)

**ขอให้สนุกกับการเรียนรู้ AI! 🚀**
