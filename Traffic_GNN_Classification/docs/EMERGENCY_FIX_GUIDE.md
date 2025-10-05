# 🔧 Emergency Fix - Model & Cache Issues

## 🚨 ปัญหาที่พบ

1. ❌ **กราฟ Speed Prediction ไม่เปลี่ยนตามโมเดล**
2. ❌ **GNN Graph vs Traffic Congestion Levels ไม่ตรงกัน**
3. ❌ **โมเดลใน Training Tab vs Sidebar ไม่ตรงกัน**

## ✅ การแก้ไขที่ทำ

### 1. สร้าง Shared Model Manager (`model_manager.py`)

**ปัญหา:** Dashboard และ Training Tab ใช้ function ต่างกันในการสแกนโมเดล

**แก้ไข:** สร้าง central module สำหรับจัดการโมเดล

```python
# app/model_manager.py

def scan_available_models():
    """สแกนโมเดลจาก outputs/ - ใช้ร่วมกันทั้งหมด"""
    
def get_model_list_for_selector():
    """รายชื่อโมเดลสำหรับ dropdown - ใช้ใน Sidebar"""
```

**ผลลัพธ์:**
- ✅ Sidebar และ Training Tab ใช้ข้อมูลเดียวกัน
- ✅ ไม่มีความขัดแย้งระหว่าง tabs
- ✅ โมเดลใหม่จะปรากฏทุกที่พร้อมกัน

---

### 2. Force Clear ALL Caches เมื่อเปลี่ยนโมเดล

**ปัญหา:** Cache ไม่ถูก clear หมด ทำให้กราฟไม่อัปเดต

**แก้ไข:** Clear ทั้ง Session State + Streamlit Cache

```python
# dashboard.py

# Clear ALL session state ยกเว้นที่จำเป็น
keys_to_keep = ['previous_model', 'model_selector', 'trained_models']
keys_to_clear = [k for k in st.session_state.keys() if k not in keys_to_keep]

# Clear Streamlit cache
st.cache_data.clear()

# Force rerun
st.rerun()
```

**ผลลัพธ์:**
- ✅ กราฟ regenerate ใหม่ทุกครั้งที่เปลี่ยนโมเดล
- ✅ ไม่มี stale data
- ✅ Analytics แสดงข้อมูลที่ถูกต้อง

---

### 3. เพิ่ม Debug Information

**วัตถุประสงค์:** ตรวจสอบว่าข้อมูลถูกต้องหรือไม่

```python
# tab_gnn_graph.py

# แสดง: Time, Rush Hour, Night, Sample, Counts
st.info(f"🔍 Debug: Time={forecast_hour}:00 | Rush={is_rush_hour} | 
         Night={is_night} | Sample={congestion_sample} | 
         Counts={congestion_counts_debug.tolist()}")
```

**ผลลัพธ์:**
- ✅ เห็นข้อมูลที่ใช้จริง
- ✅ ตรวจสอบว่า congestion distribution ถูกต้อง
- ✅ Debug ได้ง่ายขึ้น

---

## 🚀 วิธีทดสอบ (สำคัญมาก!)

### ขั้นตอนที่ 1: Restart Dashboard แบบสมบูรณ์

```powershell
# 1. หยุด Dashboard (Ctrl+C)

# 2. Clear browser cache
# - กด Ctrl+Shift+Delete
# - เลือก "Cached images and files"
# - Clear

# 3. รัน Dashboard ใหม่
cd D:\user\Data_project\Project_data\Traffic_GNN_Classification
streamlit run app/dashboard.py

# 4. เปิดใน Incognito/Private Mode (แนะนำ)
# - Chrome: Ctrl+Shift+N
# - Firefox: Ctrl+Shift+P
```

**ทำไมต้อง Restart แบบนี้?**
- Browser cache อาจเก็บ JavaScript/CSS เก่า
- Session state ใน server อาจยังไม่ clear
- Streamlit cache ต้อง rebuild

---

### ขั้นตอนที่ 2: ทดสอบการเปลี่ยนโมเดล

```
1. ดูที่ Sidebar → "Model Selection"
   ✅ ควรเห็น: Simple GNN (Base), Enhanced GNN, ฯลฯ
   ✅ มี indicator: "✅ Using trained models"

2. จด Model ปัจจุบัน: [ชื่อโมเดล]

3. ไปที่ Tab "Model Performance"
   ✅ ดู Title กราฟ: "Speed Predictions Over Time - [ชื่อโมเดล]"
   ✅ ควรตรงกับที่เลือก

4. กลับไป Sidebar → เลือกโมเดลอื่น

5. รอ notification:
   🔄 "Switching to [โมเดลใหม่]..."
   ✅ "Cleared X cached items + Streamlit cache"
   
6. Dashboard จะ refresh อัตโนมัติ

7. ไปที่ Tab "Model Performance" อีกครั้ง
   ✅ Title ควรเปลี่ยนเป็นโมเดลใหม่
   ✅ กราฟควรแตกต่างจากเดิม
```

---

### ขั้นตอนที่ 3: ตรวจสอบ GNN Graph & Congestion

```
1. ไปที่ Tab "GNN Graph View"

2. ดู Debug Info ด้านบน:
   🔍 Debug: Time=3:00 | Rush=False | Night=True | 
            Sample=[3, 2, 3, 3, 2, ...] | 
            Counts=[15, 18, 50, 86]

3. ตรวจสอบความสมเหตุสมผล:
   - Night time → ควรมี Free-flow (3) เยอะ
   - Rush hour → ควรมี Gridlock (0) เยอะ
   
4. เลื่อนลงดู "Traffic Congestion Levels"
   - Gridlock:    15 locations (9%)
   - Congested:   18 locations (11%)
   - Moderate:    50 locations (29%)
   - Free-flow:   86 locations (51%)
   
5. ตรวจสอบว่าตัวเลขตรงกับ Debug Info:
   ✅ Counts=[15, 18, 50, 86] ตรงกับด้านล่าง
   
6. ดูกราฟด้านบน:
   ✅ สีของ nodes ควรตรงกับ % 
   ✅ เขียวเยอะ = Free-flow เยอะ
   ✅ แดงเยอะ = Gridlock เยอะ
```

---

### ขั้นตอนที่ 4: ตรวจสอบ Training Tab

```
1. ไปที่ Tab "Training"

2. Scroll ลงไปส่วน "Available Pre-trained Models"

3. เปรียบเทียบกับ Sidebar:
   ✅ โมเดลที่แสดงควรเหมือนกัน
   ✅ ชื่อควรตรงกัน
   
4. ลองคลิก "Load Model" ในโมเดลหนึ่ง

5. กลับไปดู Sidebar:
   ✅ โมเดลนั้นควรปรากฏใน dropdown
```

---

## 🐛 Troubleshooting

### ปัญหา: กราฟยังไม่เปลี่ยน

**ลองทำตามลำดับ:**

1. **Hard Refresh Browser:**
   ```
   - Chrome/Edge: Ctrl+Shift+R
   - Firefox: Ctrl+F5
   ```

2. **Clear Browser Data:**
   ```
   - กด Ctrl+Shift+Delete
   - เลือก "Last hour"
   - ลบ Cookies, Cache
   ```

3. **ใช้ Incognito Mode:**
   ```
   - เปิด Private/Incognito window
   - ไปที่ localhost:8501
   ```

4. **Kill All Streamlit Processes:**
   ```powershell
   # PowerShell
   Get-Process streamlit | Stop-Process -Force
   
   # รัน Dashboard ใหม่
   streamlit run app/dashboard.py
   ```

5. **Delete .streamlit cache:**
   ```powershell
   # ลบ cache folder
   Remove-Item -Recurse -Force .streamlit/cache
   ```

---

### ปัญหา: Congestion Levels ยังไม่ตรงกับกราฟ

**ตรวจสอบ:**

1. **ดู Debug Info:**
   ```
   - Time ถูกต้องหรือไม่?
   - Rush/Night status ถูกต้องหรือไม่?
   - Sample มีค่าสมเหตุสมผลหรือไม่?
   ```

2. **ตรวจสอบ Forecast Time:**
   ```
   - ดู Slider ว่าเลื่อนไปที่เวลาไหน
   - เปลี่ยน Forecast Time ดู
   - ข้อมูลควรเปลี่ยนตาม
   ```

3. **รัน Test Script:**
   ```powershell
   python test_congestion_distribution.py
   ```
   
   ดูว่า distribution ถูกต้องหรือไม่

---

### ปัญหา: โมเดลไม่ตรงกัน

**แก้ไข:**

1. **ตรวจสอบว่าใช้ model_manager:**
   ```python
   # ใน dashboard.py
   from model_manager import get_model_list_for_selector
   
   # ใน tab_training.py  
   from model_manager import scan_available_models
   ```

2. **Restart Dashboard:**
   ```powershell
   # หยุด (Ctrl+C)
   # รันใหม่
   streamlit run app/dashboard.py
   ```

3. **ตรวจสอบ outputs/ folder:**
   ```powershell
   python test_model_detection.py
   ```

---

## 📋 Checklist หลัง Restart

- [ ] Dashboard รันได้โดยไม่มี error
- [ ] Sidebar แสดง "✅ Using trained models"
- [ ] โมเดลใน Sidebar ตรงกับ Training Tab
- [ ] เปลี่ยนโมเดล → เห็น "Clearing cache" message
- [ ] กราฟ Speed Prediction เปลี่ยนตาม (ดู Title)
- [ ] Debug Info แสดงใน GNN Graph tab
- [ ] Congestion Levels ตรงกับ Debug Counts
- [ ] สีกราฟตรงกับ % ที่แสดง

---

## 🎯 Expected Behavior

### เมื่อเปลี่ยนโมเดล:

```
1. User: เลือก "Enhanced GNN" จาก Sidebar

2. Dashboard:
   🔄 "Switching to Enhanced GNN..."
   ✅ "Cleared 47 cached items + Streamlit cache"
   
3. Auto Refresh

4. Tab Analytics:
   🔄 "Generating new analytics for Enhanced GNN..."
   ✅ "Analytics generated for Enhanced GNN"
   📊 "Showing performance for: Enhanced GNN | Cache Key: analytics_data_Enhanced GNN"
   
5. กราฟ:
   Title: "Speed Predictions Over Time - Enhanced GNN"
   Subtitle: "ST-GCN with Attention"
```

### เมื่อดู GNN Graph:

```
Debug Info:
🔍 Debug: Time=15:00 | Rush=False | Night=False | 
          Sample=[2, 1, 2, 3, 2, 1, 3, 2, 1, 2] | 
          Counts=[25, 42, 58, 44]
          
Congestion Levels:
Gridlock    25 locations (15%)  [แดง]
Congested   42 locations (25%)  [ส้ม]
Moderate    58 locations (34%)  [เหลือง]
Free-flow   44 locations (26%)  [เขียว]

กราฟด้านบน:
- มี node สีต่างๆ ตาม %
- ประมาณ 15% แดง, 25% ส้ม, 34% เหลือง, 26% เขียว
```

---

## 🚨 ถ้ายังไม่ได้ผล

### Last Resort Solutions:

1. **Reinstall Streamlit:**
   ```powershell
   pip uninstall streamlit
   pip install streamlit
   ```

2. **ลบ __pycache__:**
   ```powershell
   Get-ChildItem -Recurse -Directory __pycache__ | Remove-Item -Recurse -Force
   ```

3. **สร้าง Fresh Virtual Environment:**
   ```powershell
   python -m venv venv_fresh
   .\venv_fresh\Scripts\Activate
   pip install -r requirements.txt
   streamlit run app/dashboard.py
   ```

4. **แจ้งข้อมูล Debug:**
   - Browser: Chrome/Firefox/Edge?
   - Version: Streamlit version? (`streamlit version`)
   - Screenshot: Debug Info ใน GNN Graph tab
   - Terminal: Error messages?

---

**สิ่งที่ต้องทำตอนนี้:**

1. ✅ Restart Dashboard (Hard restart)
2. ✅ เปิดใน Incognito Mode
3. ✅ ทดสอบเปลี่ยนโมเดล
4. ✅ Capture screenshot Debug Info
5. ✅ บอกผลลัพธ์

แล้วเราค่อยดูกันต่อครับ! 🎯
