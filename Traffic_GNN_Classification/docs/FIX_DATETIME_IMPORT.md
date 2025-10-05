# 🐛 Fix: datetime Import Conflict

## ❌ Error Message
```
AttributeError: module 'datetime' has no attribute 'now'
```

## 🔍 Root Cause

### The Problem:
**Import conflict ระหว่าง module และ class ชื่อเดียวกัน!**

### บรรทัด 11 (ถูกต้อง):
```python
from datetime import datetime  # Import class datetime
# ใช้: datetime.now() ✅
```

### บรรทัด 187 (ผิด!):
```python
import datetime  # Import module datetime
# Override class datetime ข้างบน!
# ตอนนี้ datetime = module, ไม่ใช่ class
```

### บรรทัด 190 (พยายามใช้):
```python
timestamp = datetime.datetime.now()  # ✅ ถูกต้องถ้า import module
# แต่ตอนนี้ datetime ถูก override แล้ว!
```

### บรรทัด 249 (error!):
```python
datetime.now()  # ❌ ผิด! datetime ตอนนี้เป็น module
# module datetime ไม่มี method .now()
# ต้องใช้ datetime.datetime.now()
```

---

## ✅ Solution

### Fix 1: บรรทัด 187-190
```python
# ก่อนแก้ (ผิด!)
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# หลังแก้ (ถูก!)
from datetime import datetime as dt_module
timestamp = dt_module.now().strftime("%Y%m%d_%H%M%S")
```

### Fix 2: บรรทัด 249
```python
# ก่อนแก้ (อาจมีปัญหา)
model_name = st.text_input("Model Name", 
             value=f"{model_architecture}_{datetime.now().strftime('%Y%m%d_%H%M')}")

# หลังแก้ (แน่ใจ!)
from datetime import datetime as dt
model_name = st.text_input("Model Name", 
             value=f"{model_architecture}_{dt.now().strftime('%Y%m%d_%H%M')}")
```

---

## 📚 Explanation

### datetime Module vs Class:

Python มี:
1. **Module:** `datetime` (ไฟล์ datetime.py)
2. **Class:** `datetime.datetime` (class ภายใน module)

### วิธี Import ที่ถูกต้อง:

#### Option 1: Import module
```python
import datetime
# ใช้: datetime.datetime.now()
```

#### Option 2: Import class
```python
from datetime import datetime
# ใช้: datetime.now()
```

#### Option 3: Import with alias (แนะนำ!)
```python
from datetime import datetime as dt
# ใช้: dt.now()
# ไม่ conflict กับชื่ออื่น!
```

---

## 🎯 Best Practices

### 1. ใช้ alias เมื่อชื่อซ้ำ:
```python
from datetime import datetime as dt
from datetime import timedelta as td
```

### 2. Import ที่ต้องการเท่านั้น:
```python
# ดี
from datetime import datetime, timedelta

# หลีกเลี่ยง
import datetime
```

### 3. ไม่ควร import ซ้ำในฟังก์ชัน:
```python
# ดี - import ครั้งเดียวที่บนสุด
from datetime import datetime

def my_function():
    return datetime.now()

# หลีกเลี่ยง - import ซ้ำ
def my_function():
    import datetime  # ไม่ควร!
    return datetime.now()
```

---

## 🧪 Testing

### Test Case 1: Model Name Generation
```python
# ควรสร้างชื่อแบบนี้:
"Enhanced GNN_20250106_1430"
```

### Test Case 2: Custom Model Name
```python
# ควรสร้างชื่อแบบนี้:
"Custom_20250106_143025"
```

---

## 📝 Files Modified

1. **`app/tab_training.py`** (2 fixes)
   - Line 187: `import datetime` → `from datetime import datetime as dt_module`
   - Line 249: Added local import `from datetime import datetime as dt`

---

## 🚀 Verification Steps

1. **Kill all Streamlit processes**
   ```powershell
   Get-Process streamlit | Stop-Process -Force
   ```

2. **Clear Python cache**
   ```powershell
   Get-ChildItem -Recurse __pycache__ | Remove-Item -Recurse -Force
   ```

3. **Restart Dashboard**
   ```powershell
   streamlit run app/dashboard.py
   ```

4. **Test Training Tab:**
   - Go to "Training" tab
   - Scroll to "Model Saving"
   - Check model name shows: `[Architecture]_YYYYMMDD_HHMM`
   - Should NOT show error

5. **Test Start Training:**
   - Click "Start Training"
   - Should show progress
   - Should NOT show datetime error

---

## ⚠️ Common Pitfalls

### Pitfall 1: Import Order Matters
```python
# ถ้าทำแบบนี้
from datetime import datetime  # datetime = class
import datetime                # datetime = module (override!)

# datetime ตอนนี้เป็น module, ไม่ใช่ class!
```

### Pitfall 2: Module Caching
```python
# Python cache imports
# การแก้ไขอาจไม่ทำงานจนกว่าจะ:
# 1. Restart process
# 2. Clear __pycache__
```

### Pitfall 3: Nested Imports
```python
# Import ใน function อาจ conflict กับ import ข้างบน
# ควร import ที่บนสุดของไฟล์
```

---

## 📖 References

- [Python datetime documentation](https://docs.python.org/3/library/datetime.html)
- [Import system](https://docs.python.org/3/reference/import.html)

---

**Status:** ✅ Fixed  
**Test Required:** Yes (รัน FORCE_RESTART.bat)
