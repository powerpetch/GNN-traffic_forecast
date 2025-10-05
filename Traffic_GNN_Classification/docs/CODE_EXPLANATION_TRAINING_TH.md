# 📘 อธิบายโค้ด: train.py & enhanced_train.py

## 📋 ข้อมูลไฟล์

- **ชื่อไฟล์:** `train.py` และ `enhanced_train.py`
- **หน้าที่:** เทรนโมเดล GNN สำหรับทำนายการจราจร
- **จำนวนบรรทัด:** train.py (~450), enhanced_train.py (~520)
- **ภาษา:** Python + PyTorch

---

## 🎯 ภาพรวม

ไฟล์ทั้ง 2 นี้เป็น **Training Pipeline** ที่ครบวงจร:
- `train.py` → เทรน **SimpleMultiTaskGNN** (โมเดลพื้นฐาน)
- `enhanced_train.py` → เทรน **EnhancedGNNModel** (โมเดลขั้นสูง)

### **Training Pipeline:**
```
โหลดข้อมูล → สร้างโมเดล → เทรน → ประเมินผล → บันทึก
```

---

## 📂 โครงสร้างไฟล์

### **train.py (Simple Model):**
```python
train.py
├── 1. load_and_prepare_data()      → โหลดข้อมูล
├── 2. create_simple_model()        → สร้างโมเดล
├── 3. train_simple_model()         → เทรนโมเดล
├── 4. evaluate_simple_model()      → ประเมินผล
├── 5. plot_training_history()      → plot กราฟ
└── 6. main()                       → ฟังก์ชันหลัก
```

### **enhanced_train.py (Enhanced Model):**
```python
enhanced_train.py
├── 1. load_and_prepare_data()      → โหลดข้อมูล
├── 2. create_enhanced_model()      → สร้างโมเดล
├── 3. train_enhanced_model()       → เทรนโมเดล (+ techniques)
├── 4. evaluate_enhanced_model()    → ประเมินผล
├── 5. plot_enhanced_results()      → plot กราฟ
├── 6. save_confusion_matrices()    → confusion matrix
└── 7. main()                       → ฟังก์ชันหลัก
```

---

## 1️⃣ โหลดและเตรียมข้อมูล

### **load_and_prepare_data() - โหลดข้อมูล**

```python
def load_and_prepare_data(data_path='outputs/processed_data.pkl'):
    """
    โหลดข้อมูลที่ประมวลผลแล้ว
    
    Returns:
        X_train, X_val, y_train, y_val
    """
    
    # โหลดข้อมูล
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    df = data['data']
    
    # แยก features และ labels
    feature_cols = [
        'speed_mean', 'speed_median', 'speed_std',
        'hour_sin', 'hour_cos', 'day_of_week',
        'is_weekend', 'time_since_rush_hour',
        'nearby_congestion', 'speed_lag_1'
    ]
    
    X = df[feature_cols].values
    y_congestion = df['congestion_label'].values
    y_rush_hour = df['rush_hour_label'].values
    
    # แบ่ง train/val (80/20)
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_cong_train, y_cong_val, y_rush_train, y_rush_val = \
        train_test_split(X, y_congestion, y_rush_hour, 
                         test_size=0.2, random_state=42)
    
    # แปลงเป็น PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    y_cong_train = torch.LongTensor(y_cong_train)
    y_cong_val = torch.LongTensor(y_cong_val)
    y_rush_train = torch.LongTensor(y_rush_train)
    y_rush_val = torch.LongTensor(y_rush_val)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    
    return X_train, X_val, (y_cong_train, y_rush_train), (y_cong_val, y_rush_val)
```

**ตัวอย่าง Output:**
```
Loading data from outputs/processed_data.pkl...
Training samples: 80,000
Validation samples: 20,000

Features: 10
  - speed_mean, speed_median, speed_std
  - hour_sin, hour_cos
  - day_of_week, is_weekend
  - time_since_rush_hour
  - nearby_congestion
  - speed_lag_1

Labels:
  - Congestion: 4 classes (0-3)
  - Rush Hour: 2 classes (0-1)
```

---

## 2️⃣ สร้างโมเดล

### **train.py: create_simple_model()**

```python
def create_simple_model(num_features=10, hidden_dim=64):
    """
    สร้างโมเดล SimpleMultiTaskGNN
    """
    
    model = SimpleMultiTaskGNN(
        num_features=num_features,
        hidden_dim=hidden_dim
    )
    
    # นับจำนวน parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: SimpleMultiTaskGNN")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model
```

### **enhanced_train.py: create_enhanced_model()**

```python
def create_enhanced_model(num_features=10, hidden_dim=128, dropout=0.3):
    """
    สร้างโมเดล EnhancedGNNModel
    """
    
    model = EnhancedGNNModel(
        num_features=num_features,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: EnhancedGNNModel")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Dropout: {dropout}")
    
    return model
```

**ตัวอย่าง Output:**
```
# Simple Model:
Model: SimpleMultiTaskGNN
  Total parameters: 9,222
  Trainable parameters: 9,222

# Enhanced Model:
Model: EnhancedGNNModel
  Total parameters: 62,146
  Trainable parameters: 62,146
  Hidden dim: 128
  Dropout: 0.3
```

---

## 3️⃣ Training Loop - การเทรน

### **train_simple_model() - เทรนโมเดล Simple**

```python
def train_simple_model(model, train_loader, val_loader, 
                       num_epochs=50, learning_rate=0.001,
                       device='cuda'):
    """
    เทรนโมเดล SimpleMultiTaskGNN
    """
    
    # Move model to device
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Loss functions
    criterion_congestion = nn.CrossEntropyLoss()
    criterion_rush_hour = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_congestion_acc': [],
        'val_congestion_acc': [],
        'train_rush_hour_acc': [],
        'val_rush_hour_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Learning rate: {learning_rate}")
    print(f"Optimizer: AdamW")
    print(f"Scheduler: ReduceLROnPlateau")
    
    for epoch in range(num_epochs):
        # ============ Training Phase ============
        model.train()
        train_loss = 0.0
        train_congestion_correct = 0
        train_rush_hour_correct = 0
        train_total = 0
        
        for batch_X, batch_y_cong, batch_y_rush in train_loader:
            batch_X = batch_X.to(device)
            batch_y_cong = batch_y_cong.to(device)
            batch_y_rush = batch_y_rush.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Calculate losses
            loss_cong = criterion_congestion(
                outputs['congestion_logits'], 
                batch_y_cong
            )
            loss_rush = criterion_rush_hour(
                outputs['rush_hour_logits'],
                batch_y_rush
            )
            
            # Total loss
            loss = loss_cong + loss_rush
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calculate accuracy
            _, pred_cong = outputs['congestion_logits'].max(dim=1)
            _, pred_rush = outputs['rush_hour_logits'].max(dim=1)
            
            train_congestion_correct += (pred_cong == batch_y_cong).sum().item()
            train_rush_hour_correct += (pred_rush == batch_y_rush).sum().item()
            train_total += batch_X.size(0)
            train_loss += loss.item()
        
        # Average training metrics
        train_loss /= len(train_loader)
        train_congestion_acc = train_congestion_correct / train_total
        train_rush_hour_acc = train_rush_hour_correct / train_total
        
        # ============ Validation Phase ============
        model.eval()
        val_loss = 0.0
        val_congestion_correct = 0
        val_rush_hour_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y_cong, batch_y_rush in val_loader:
                batch_X = batch_X.to(device)
                batch_y_cong = batch_y_cong.to(device)
                batch_y_rush = batch_y_rush.to(device)
                
                # Forward pass
                outputs = model(batch_X)
                
                # Calculate losses
                loss_cong = criterion_congestion(
                    outputs['congestion_logits'],
                    batch_y_cong
                )
                loss_rush = criterion_rush_hour(
                    outputs['rush_hour_logits'],
                    batch_y_rush
                )
                
                loss = loss_cong + loss_rush
                
                # Calculate accuracy
                _, pred_cong = outputs['congestion_logits'].max(dim=1)
                _, pred_rush = outputs['rush_hour_logits'].max(dim=1)
                
                val_congestion_correct += (pred_cong == batch_y_cong).sum().item()
                val_rush_hour_correct += (pred_rush == batch_y_rush).sum().item()
                val_total += batch_X.size(0)
                val_loss += loss.item()
        
        # Average validation metrics
        val_loss /= len(val_loader)
        val_congestion_acc = val_congestion_correct / val_total
        val_rush_hour_acc = val_rush_hour_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_congestion_acc'].append(train_congestion_acc)
        history['val_congestion_acc'].append(val_congestion_acc)
        history['train_rush_hour_acc'].append(train_rush_hour_acc)
        history['val_rush_hour_acc'].append(val_rush_hour_acc)
        
        # Print progress
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train Cong Acc: {train_congestion_acc:.4f} | Val Cong Acc: {val_congestion_acc:.4f}")
        print(f"  Train Rush Acc: {train_rush_hour_acc:.4f} | Val Rush Acc: {val_rush_hour_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_congestion_acc': val_congestion_acc,
                'val_rush_hour_acc': val_rush_hour_acc
            }, 'outputs/best_model.pth')
            
            print(f"  ✓ Best model saved! (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
    
    return model, history
```

### **ตัวอย่าง Training Output:**

```
Starting training for 50 epochs...
Device: cuda
Learning rate: 0.001
Optimizer: AdamW
Scheduler: ReduceLROnPlateau

Epoch [1/50]
  Train Loss: 1.2345 | Val Loss: 1.1234
  Train Cong Acc: 0.4523 | Val Cong Acc: 0.4712
  Train Rush Acc: 0.7234 | Val Rush Acc: 0.7456
  ✓ Best model saved! (Val Loss: 1.1234)

Epoch [2/50]
  Train Loss: 0.9876 | Val Loss: 0.9234
  Train Cong Acc: 0.5834 | Val Cong Acc: 0.6012
  Train Rush Acc: 0.8123 | Val Rush Acc: 0.8234
  ✓ Best model saved! (Val Loss: 0.9234)

...

Epoch [35/50]
  Train Loss: 0.2145 | Val Loss: 0.2876
  Train Cong Acc: 0.9234 | Val Cong Acc: 0.9012
  Train Rush Acc: 0.9523 | Val Rush Acc: 0.9434
  ✓ Best model saved! (Val Loss: 0.2876)

Epoch [36/50]
  Train Loss: 0.2098 | Val Loss: 0.2901
  Train Cong Acc: 0.9256 | Val Cong Acc: 0.9001
  Train Rush Acc: 0.9534 | Val Rush Acc: 0.9423
  Patience: 1/10

...

Early stopping triggered at epoch 45
Training completed!
Best validation loss: 0.2876
```

---

## 4️⃣ Evaluation - ประเมินผล

### **evaluate_simple_model() - ประเมินโมเดล**

```python
def evaluate_simple_model(model, test_loader, device='cuda'):
    """
    ประเมินโมเดลบน test set
    """
    
    model.eval()
    model = model.to(device)
    
    all_pred_cong = []
    all_true_cong = []
    all_pred_rush = []
    all_true_rush = []
    
    with torch.no_grad():
        for batch_X, batch_y_cong, batch_y_rush in test_loader:
            batch_X = batch_X.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            
            # Get predictions
            _, pred_cong = outputs['congestion_logits'].max(dim=1)
            _, pred_rush = outputs['rush_hour_logits'].max(dim=1)
            
            all_pred_cong.extend(pred_cong.cpu().numpy())
            all_true_cong.extend(batch_y_cong.numpy())
            all_pred_rush.extend(pred_rush.cpu().numpy())
            all_true_rush.extend(batch_y_rush.numpy())
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Congestion metrics
    cong_acc = accuracy_score(all_true_cong, all_pred_cong)
    cong_precision = precision_score(all_true_cong, all_pred_cong, average='weighted')
    cong_recall = recall_score(all_true_cong, all_pred_cong, average='weighted')
    cong_f1 = f1_score(all_true_cong, all_pred_cong, average='weighted')
    
    # Rush hour metrics
    rush_acc = accuracy_score(all_true_rush, all_pred_rush)
    rush_precision = precision_score(all_true_rush, all_pred_rush, average='weighted')
    rush_recall = recall_score(all_true_rush, all_pred_rush, average='weighted')
    rush_f1 = f1_score(all_true_rush, all_pred_rush, average='weighted')
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print("\nCongestion Classification:")
    print(f"  Accuracy:  {cong_acc:.4f}")
    print(f"  Precision: {cong_precision:.4f}")
    print(f"  Recall:    {cong_recall:.4f}")
    print(f"  F1-Score:  {cong_f1:.4f}")
    
    print("\nRush Hour Classification:")
    print(f"  Accuracy:  {rush_acc:.4f}")
    print(f"  Precision: {rush_precision:.4f}")
    print(f"  Recall:    {rush_recall:.4f}")
    print(f"  F1-Score:  {rush_f1:.4f}")
    
    # Confusion matrices
    from sklearn.metrics import confusion_matrix
    
    cm_cong = confusion_matrix(all_true_cong, all_pred_cong)
    cm_rush = confusion_matrix(all_true_rush, all_pred_rush)
    
    print("\nCongestion Confusion Matrix:")
    print(cm_cong)
    
    print("\nRush Hour Confusion Matrix:")
    print(cm_rush)
    
    return {
        'congestion': {
            'accuracy': cong_acc,
            'precision': cong_precision,
            'recall': cong_recall,
            'f1': cong_f1,
            'confusion_matrix': cm_cong
        },
        'rush_hour': {
            'accuracy': rush_acc,
            'precision': rush_precision,
            'recall': rush_recall,
            'f1': rush_f1,
            'confusion_matrix': cm_rush
        }
    }
```

**ตัวอย่าง Output:**
```
==================================================
EVALUATION RESULTS
==================================================

Congestion Classification:
  Accuracy:  0.9234
  Precision: 0.9256
  Recall:    0.9234
  F1-Score:  0.9241

Rush Hour Classification:
  Accuracy:  0.9523
  Precision: 0.9534
  Recall:    0.9523
  F1-Score:  0.9527

Congestion Confusion Matrix:
[[985  12   3   0]    # Gridlock
 [ 15 940  32  13]    # Congested
 [  2  38 932  28]    # Moderate
 [  0   5  25 970]]   # Free Flow

Rush Hour Confusion Matrix:
[[4850  150]    # Non-Rush Hour
 [  40 4960]]   # Rush Hour
```

---

## 5️⃣ Visualization - พล็อตกราฟ

### **plot_training_history() - พล็อตประวัติการเทรน**

```python
def plot_training_history(history, save_path='outputs/training_history.png'):
    """
    พล็อตกราฟการเทรน
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Congestion accuracy
    axes[0, 1].plot(history['train_congestion_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_congestion_acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Congestion Classification Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Rush hour accuracy
    axes[1, 0].plot(history['train_rush_hour_acc'], label='Train Acc')
    axes[1, 0].plot(history['val_rush_hour_acc'], label='Val Acc')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Rush Hour Classification Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Combined accuracy
    axes[1, 1].plot(
        (np.array(history['train_congestion_acc']) + 
         np.array(history['train_rush_hour_acc'])) / 2,
        label='Train Avg Acc'
    )
    axes[1, 1].plot(
        (np.array(history['val_congestion_acc']) + 
         np.array(history['val_rush_hour_acc'])) / 2,
        label='Val Avg Acc'
    )
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Average Accuracy (Both Tasks)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nTraining history plot saved to {save_path}")
```

---

## 6️⃣ Main Function

### **main() - ฟังก์ชันหลัก**

```python
def main():
    """
    Main training pipeline
    """
    
    print("="*60)
    print("TRAFFIC GNN CLASSIFICATION - TRAINING")
    print("="*60)
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 1. Load data
    print("\n" + "-"*60)
    print("Step 1: Loading data...")
    print("-"*60)
    X_train, X_val, y_train, y_val = load_and_prepare_data()
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train[0], y_train[1])
    val_dataset = TensorDataset(X_val, y_val[0], y_val[1])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 2. Create model
    print("\n" + "-"*60)
    print("Step 2: Creating model...")
    print("-"*60)
    model = create_simple_model(num_features=10, hidden_dim=64)
    
    # 3. Train model
    print("\n" + "-"*60)
    print("Step 3: Training model...")
    print("-"*60)
    model, history = train_simple_model(
        model, train_loader, val_loader,
        num_epochs=50,
        learning_rate=0.001,
        device=device
    )
    
    # 4. Evaluate model
    print("\n" + "-"*60)
    print("Step 4: Evaluating model...")
    print("-"*60)
    results = evaluate_simple_model(model, val_loader, device=device)
    
    # 5. Plot results
    print("\n" + "-"*60)
    print("Step 5: Plotting results...")
    print("-"*60)
    plot_training_history(history)
    
    # 6. Save results
    print("\n" + "-"*60)
    print("Step 6: Saving results...")
    print("-"*60)
    
    with open('outputs/evaluation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    with open('outputs/training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print("\nSaved files:")
    print("  - outputs/best_model.pth")
    print("  - outputs/training_history.pkl")
    print("  - outputs/training_history.png")
    print("  - outputs/evaluation_results.pkl")

if __name__ == "__main__":
    main()
```

---

## 🆚 เปรียบเทียบ train.py vs enhanced_train.py

| Feature | train.py | enhanced_train.py |
|---------|----------|-------------------|
| **โมเดล** | SimpleMultiTaskGNN | EnhancedGNNModel |
| **Parameters** | ~9K | ~62K |
| **Hidden Dim** | 64 | 128 |
| **Dropout** | 0.2 | 0.3 |
| **Batch Size** | 32 | 64 |
| **Learning Rate** | 0.001 | 0.0005 |
| **Optimizer** | AdamW | AdamW |
| **Scheduler** | ReduceLROnPlateau | CosineAnnealingWarmRestarts |
| **Early Stopping** | ✅ (patience=10) | ✅ (patience=15) |
| **Gradient Clipping** | ✅ (max_norm=1.0) | ✅ (max_norm=1.0) |
| **Mixed Precision** | ❌ | ✅ |
| **Label Smoothing** | ❌ | ✅ (0.1) |
| **Training Time** | ~5 min | ~45 min |
| **Accuracy** | ~92% | ~98% |

---

## 💡 เทคนิคขั้นสูงใน enhanced_train.py

### **1. Mixed Precision Training:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(batch_X)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# ประโยชน์:
# - เร็วขึ้น 2-3 เท่า
# - ใช้ memory น้อยลง
```

### **2. Label Smoothing:**
```python
# แทนที่ hard labels: [0, 1, 0, 0]
# ด้วย smooth labels: [0.025, 0.925, 0.025, 0.025]

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ประโยชน์:
# - ลด overfitting
# - โมเดลมั่นใจน้อยลง (calibrated)
```

### **3. Cosine Annealing with Warm Restarts:**
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # รีสตาร์ททุก 10 epochs
    T_mult=2  # เพิ่มระยะเวลา 2 เท่า
)

# Learning rate pattern:
# Epoch 0-10:   1e-3 → 1e-5 (ลดลงแบบ cosine)
# Epoch 10:     รีสตาร์ทเป็น 1e-3
# Epoch 10-30:  1e-3 → 1e-5 (20 epochs)
# Epoch 30:     รีสตาร์ทเป็น 1e-3
```

### **4. Advanced Early Stopping:**
```python
# หยุดเมื่อ validation loss ไม่ดีขึ้น 15 epochs
# แต่ยังเช็คว่า training loss ลดลงอยู่หรือไม่

if val_loss < best_val_loss - min_delta:
    best_val_loss = val_loss
    patience_counter = 0
    save_checkpoint()
else:
    patience_counter += 1
    
    if patience_counter >= patience:
        if train_loss > prev_train_loss:
            # Train loss ไม่ลดแล้ว = converged
            break
        else:
            # Train loss ยังลด = overfitting
            # ลด learning rate
            for g in optimizer.param_groups:
                g['lr'] *= 0.5
            patience_counter = 0
```

---

## 🎯 สรุป

### **ขั้นตอนการเทรน:**
1. โหลดข้อมูล → แบ่ง train/val
2. สร้างโมเดล → นับ parameters
3. เทรนโมเดล → forward/backward/optimize
4. ประเมินผล → accuracy, F1, confusion matrix
5. พล็อตกราฟ → loss curves, accuracy curves
6. บันทึกผล → model, history, results

### **Key Techniques:**
- ✅ Multi-task learning
- ✅ Early stopping
- ✅ Learning rate scheduling
- ✅ Gradient clipping
- ✅ Mixed precision (enhanced)
- ✅ Label smoothing (enhanced)

---

**สร้างเมื่อ:** 5 ตุลาคม 2025  
**เวอร์ชัน:** 1.0  
**ผู้เขียน:** Traffic GNN Classification Team
