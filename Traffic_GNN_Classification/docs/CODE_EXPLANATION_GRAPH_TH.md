# 📘 อธิบายโค้ด: graph_constructor.py

## 📋 ข้อมูลไฟล์

- **ชื่อไฟล์:** `src/utils/graph_constructor.py`
- **หน้าที่:** สร้างและจัดการกราฟเครือข่ายถนน
- **จำนวนบรรทัด:** ~407 บรรทัด
- **ภาษา:** Python + NetworkX + PyTorch Geometric

---

## 🎯 ภาพรวม

ไฟล์นี้สร้าง **กราฟจากเครือข่ายถนน** เพื่อให้โมเดล GNN สามารถประมวลผลได้

### **Pipeline:**
```
Road Network (OSM) → Graph Construction → PyTorch Geometric Data
    ↓
Nodes = จุดตัด (intersections)
Edges = ถนน (road segments)
Features = ข้อมูลการจราจร
```

---

## 📂 โครงสร้างคลาส GraphConstructor

```python
class GraphConstructor:
    """
    สร้างและจัดการกราฟเครือข่ายถนน
    """
    
    # Constructor
    __init__(road_network)
    
    # 1. สร้างกราฟ
    build_road_graph()
    create_adjacency_matrix()
    
    # 2. ค้นหา neighbors
    find_k_nearest_neighbors()
    compute_haversine_distance()
    
    # 3. สร้าง PyTorch Geometric Data
    create_pytorch_geometric_data()
    prepare_for_training()
    
    # 4. บันทึก/โหลด
    save_graph()
    load_graph()
    
    # 5. วิเคราะห์
    get_graph_statistics()
    visualize_graph()
```

---

## 1️⃣ สร้างกราฟ

### **build_road_graph() - สร้างกราฟจากถนน**

```python
def build_road_graph(self, connection_threshold: float = 0.001) -> nx.Graph:
    """
    สร้าง NetworkX Graph จากเครือข่ายถนน
    
    Args:
        connection_threshold: ระยะห่างสูงสุดที่ถือว่าเป็นจุดเดียวกัน (degrees)
                             0.001° ≈ 110 เมตร
    
    Returns:
        NetworkX Graph object
    
    ขั้นตอน:
        1. แยกจุดปลายของแต่ละเส้นถนน (endpoints)
        2. จัดกลุ่มจุดที่ใกล้กันเป็น intersections (จุดตัด)
        3. สร้าง nodes = intersections
        4. สร้าง edges = ถนนที่เชื่อมระหว่าง intersections
    """
```

**ตัวอย่างการทำงาน:**

#### **Step 1: แยกจุดปลายถนน**
```
Road 1: A ────────── B
Road 2: B ────────── C
Road 3: A ────────── D

Endpoints:
Road 1: (A, B)
Road 2: (B, C)
Road 3: (A, D)
```

#### **Step 2: จัดกลุ่มเป็น Intersections**
```python
# จุด A และ A ห่างกัน < threshold → เป็น intersection เดียวกัน
# จุด B และ B ห่างกัน < threshold → เป็น intersection เดียวกัน

Intersections:
  Intersection 0: จุด A (เชื่อม Road 1, 3)
  Intersection 1: จุด B (เชื่อม Road 1, 2)
  Intersection 2: จุด C (เชื่อม Road 2)
  Intersection 3: จุด D (เชื่อม Road 3)
```

#### **Step 3: สร้าง Graph**
```python
G = nx.Graph()

# Add nodes (intersections)
G.add_node(0, pos=(13.7563, 100.5018), num_connections=2)
G.add_node(1, pos=(13.7565, 100.5020), num_connections=2)
G.add_node(2, pos=(13.7567, 100.5022), num_connections=1)
G.add_node(3, pos=(13.7560, 100.5015), num_connections=1)

# Add edges (roads)
G.add_edge(0, 1, edge_id='ROAD_1', length_km=0.25)
G.add_edge(1, 2, edge_id='ROAD_2', length_km=0.30)
G.add_edge(0, 3, edge_id='ROAD_3', length_km=0.20)
```

**ผลลัพธ์:**
```
Graph:
  (0) ---- ROAD_1 ---- (1)
   |                    |
ROAD_3               ROAD_2
   |                    |
  (3)                  (2)
```

---

### **create_adjacency_matrix() - สร้าง Adjacency Matrix**

```python
def create_adjacency_matrix(self, normalize: bool = True) -> np.ndarray:
    """
    สร้าง adjacency matrix จากกราฟ
    
    Args:
        normalize: normalize แถวให้รวมเป็น 1
    
    Returns:
        Adjacency matrix shape (num_nodes, num_nodes)
    
    Normalization:
        1. เพิ่ม self-loops (diagonal = 1)
        2. Normalize แต่ละแถว: A_ij = A_ij / sum(A_i)
    """
    
    # สร้าง adjacency matrix
    adj_matrix = nx.adjacency_matrix(self.graph).toarray()
    
    # เช่น:
    # [[0, 1, 0, 1],   # Node 0 เชื่อมกับ 1, 3
    #  [1, 0, 1, 0],   # Node 1 เชื่อมกับ 0, 2
    #  [0, 1, 0, 0],   # Node 2 เชื่อมกับ 1
    #  [1, 0, 0, 0]]   # Node 3 เชื่อมกับ 0
    
    if normalize:
        # เพิ่ม self-loops
        adj_matrix += np.eye(adj_matrix.shape[0])
        
        # Row normalization
        row_sums = adj_matrix.sum(axis=1, keepdims=True)
        adj_matrix = adj_matrix / row_sums
    
    return adj_matrix
```

**ตัวอย่าง:**
```python
# Before normalization:
[[0, 1, 0, 1],
 [1, 0, 1, 0],
 [0, 1, 0, 0],
 [1, 0, 0, 0]]

# After adding self-loops:
[[1, 1, 0, 1],   # sum = 3
 [1, 1, 1, 0],   # sum = 3
 [0, 1, 1, 0],   # sum = 2
 [1, 0, 0, 1]]   # sum = 2

# After normalization:
[[0.33, 0.33, 0.00, 0.33],   # แต่ละแถวรวมเป็น 1
 [0.33, 0.33, 0.33, 0.00],
 [0.00, 0.50, 0.50, 0.00],
 [0.50, 0.00, 0.00, 0.50]]
```

---

## 2️⃣ ค้นหา Neighbors

### **find_k_nearest_neighbors() - หาเพื่อนบ้าน K ตัว**

```python
def find_k_nearest_neighbors(self, locations: pd.DataFrame, 
                             k: int = 5) -> Dict:
    """
    หาสถานที่ใกล้ที่สุด K ตัว
    
    Args:
        locations: DataFrame ที่มี (latitude, longitude)
        k: จำนวนเพื่อนบ้าน
    
    Returns:
        dict: {location_id: [neighbor_ids]}
    """
    
    neighbors = {}
    
    for idx, row in locations.iterrows():
        # คำนวณระยะห่างกับสถานที่อื่นทั้งหมด
        distances = []
        for idx2, row2 in locations.iterrows():
            if idx != idx2:
                dist = self.compute_haversine_distance(
                    row['latitude'], row['longitude'],
                    row2['latitude'], row2['longitude']
                )
                distances.append((idx2, dist))
        
        # เรียงตามระยะห่าง
        distances.sort(key=lambda x: x[1])
        
        # เลือก k ตัวแรก
        neighbors[idx] = [loc_id for loc_id, _ in distances[:k]]
    
    return neighbors
```

**ตัวอย่าง:**
```python
locations = pd.DataFrame({
    'id': ['A', 'B', 'C', 'D', 'E'],
    'latitude': [13.75, 13.76, 13.74, 13.77, 13.73],
    'longitude': [100.50, 100.51, 100.49, 100.52, 100.48]
})

neighbors = find_k_nearest_neighbors(locations, k=2)

# Output:
{
    'A': ['B', 'C'],  # A ใกล้ B และ C ที่สุด
    'B': ['A', 'D'],  # B ใกล้ A และ D ที่สุด
    'C': ['A', 'E'],
    'D': ['B', 'A'],
    'E': ['C', 'A']
}
```

---

### **compute_haversine_distance() - คำนวณระยะทาง**

```python
def compute_haversine_distance(self, lat1, lon1, lat2, lon2) -> float:
    """
    คำนวณระยะทางบนโลก (Great Circle Distance)
    
    สูตร Haversine:
        a = sin²(Δlat/2) + cos(lat1)×cos(lat2)×sin²(Δlon/2)
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
    R = 6371
    distance = R * c
    
    return distance
```

**ตัวอย่าง:**
```python
# Siam Square → MBK Center
distance = compute_haversine_distance(
    13.7465, 100.5326,  # Siam
    13.7443, 100.5300   # MBK
)
print(f"Distance: {distance:.3f} km")
# Output: Distance: 0.323 km
```

---

## 3️⃣ สร้าง PyTorch Geometric Data

### **create_pytorch_geometric_data() - แปลงเป็น PyG Data**

```python
def create_pytorch_geometric_data(self, features_df: pd.DataFrame,
                                  labels_df: pd.DataFrame) -> Data:
    """
    สร้าง PyTorch Geometric Data object
    
    Args:
        features_df: DataFrame ที่มี features (num_nodes, num_features)
        labels_df: DataFrame ที่มี labels
    
    Returns:
        PyTorch Geometric Data object
    """
    
    # Node features
    x = torch.FloatTensor(features_df.values)  # [num_nodes, num_features]
    
    # Labels
    y_congestion = torch.LongTensor(labels_df['congestion_label'].values)
    y_rush_hour = torch.LongTensor(labels_df['rush_hour_label'].values)
    
    # Edge index
    edge_list = []
    for node1, node2 in self.graph.edges():
        node1_idx = self.node_mapping[node1]
        node2_idx = self.node_mapping[node2]
        edge_list.append([node1_idx, node2_idx])
        edge_list.append([node2_idx, node1_idx])  # Undirected
    
    edge_index = torch.LongTensor(edge_list).t()  # [2, num_edges]
    
    # สร้าง Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y_congestion=y_congestion,
        y_rush_hour=y_rush_hour
    )
    
    return data
```

**ตัวอย่าง:**
```python
# Features (217 nodes, 10 features)
features_df = pd.DataFrame({
    'speed_mean': [45.2, 38.5, ...],
    'speed_std': [5.1, 7.2, ...],
    'hour_sin': [0.707, 0.707, ...],
    ...  # 10 features total
})

# Labels
labels_df = pd.DataFrame({
    'congestion_label': [2, 1, 3, ...],  # 0-3
    'rush_hour_label': [1, 1, 0, ...]    # 0-1
})

# สร้าง PyG Data
data = create_pytorch_geometric_data(features_df, labels_df)

print(data)
# Output:
# Data(x=[217, 10], edge_index=[2, 1234], 
#      y_congestion=[217], y_rush_hour=[217])
```

---

### **prepare_for_training() - เตรียมข้อมูลสำหรับเทรน**

```python
def prepare_for_training(self, processed_df: pd.DataFrame,
                        train_split: float = 0.8) -> Tuple:
    """
    แบ่งข้อมูลเป็น train/val/test
    
    Returns:
        (train_data, val_data, test_data)
    """
    
    # สร้าง PyG Data
    data = self.create_pytorch_geometric_data(
        processed_df[feature_columns],
        processed_df[label_columns]
    )
    
    # แบ่งข้อมูล
    num_nodes = data.x.shape[0]
    indices = np.random.permutation(num_nodes)
    
    train_size = int(num_nodes * 0.7)
    val_size = int(num_nodes * 0.15)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data
```

**ตัวอย่าง:**
```python
# เตรียมข้อมูล
data = prepare_for_training(processed_df, train_split=0.7)

print(f"Total nodes: {data.x.shape[0]}")
print(f"Train nodes: {data.train_mask.sum().item()}")
print(f"Val nodes: {data.val_mask.sum().item()}")
print(f"Test nodes: {data.test_mask.sum().item()}")

# Output:
# Total nodes: 217
# Train nodes: 152 (70%)
# Val nodes: 33 (15%)
# Test nodes: 32 (15%)
```

---

## 4️⃣ บันทึก/โหลดกราฟ

### **save_graph() - บันทึกกราฟ**

```python
def save_graph(self, filepath: str):
    """บันทึกกราฟและข้อมูลที่เกี่ยวข้อง"""
    
    graph_data = {
        'graph': self.graph,
        'node_mapping': self.node_mapping,
        'edge_mapping': self.edge_mapping,
        'adjacency_matrix': self.adjacency_matrix
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(graph_data, f)
    
    print(f"Graph saved to {filepath}")
```

### **load_graph() - โหลดกราฟ**

```python
def load_graph(self, filepath: str):
    """โหลดกราฟที่บันทึกไว้"""
    
    with open(filepath, 'rb') as f:
        graph_data = pickle.load(f)
    
    self.graph = graph_data['graph']
    self.node_mapping = graph_data['node_mapping']
    self.edge_mapping = graph_data['edge_mapping']
    self.adjacency_matrix = graph_data['adjacency_matrix']
    
    print(f"Graph loaded from {filepath}")
```

---

## 5️⃣ วิเคราะห์กราฟ

### **get_graph_statistics() - สถิติกราฟ**

```python
def get_graph_statistics(self) -> Dict:
    """
    คำนวณสถิติของกราฟ
    """
    
    stats = {
        'num_nodes': self.graph.number_of_nodes(),
        'num_edges': self.graph.number_of_edges(),
        'avg_degree': np.mean([d for _, d in self.graph.degree()]),
        'density': nx.density(self.graph),
        'num_components': nx.number_connected_components(self.graph),
        'diameter': nx.diameter(self.graph) if nx.is_connected(self.graph) else None,
        'avg_clustering': nx.average_clustering(self.graph)
    }
    
    return stats
```

**ตัวอย่าง:**
```python
stats = graph_constructor.get_graph_statistics()

print(stats)
# Output:
# {
#     'num_nodes': 217,
#     'num_edges': 542,
#     'avg_degree': 4.99,
#     'density': 0.023,
#     'num_components': 1,
#     'diameter': 12,
#     'avg_clustering': 0.15
# }
```

---

## 💡 ตัวอย่างการใช้งานทั้งหมด

```python
import geopandas as gpd
import pandas as pd

# 1. โหลดเครือข่ายถนน
road_network = gpd.read_file("Data/hotosm_tha_roads_lines_gpkg/roads.gpkg")

# 2. สร้าง GraphConstructor
graph_constructor = GraphConstructor(road_network)

# 3. สร้างกราฟ
G = graph_constructor.build_road_graph(connection_threshold=0.001)
print(f"Created graph with {G.number_of_nodes()} nodes")

# 4. สร้าง adjacency matrix
adj_matrix = graph_constructor.create_adjacency_matrix(normalize=True)
print(f"Adjacency matrix shape: {adj_matrix.shape}")

# 5. โหลดข้อมูลที่ประมวลผลแล้ว
processed_df = pd.read_pickle("outputs/processed_data.pkl")

# 6. สร้าง PyTorch Geometric Data
data = graph_constructor.prepare_for_training(
    processed_df,
    train_split=0.7
)

print(f"PyG Data: {data}")
print(f"  Nodes: {data.x.shape}")
print(f"  Edges: {data.edge_index.shape}")
print(f"  Train: {data.train_mask.sum()}")
print(f"  Val: {data.val_mask.sum()}")
print(f"  Test: {data.test_mask.sum()}")

# 7. บันทึกกราฟ
graph_constructor.save_graph("outputs/road_graph.pkl")

# 8. วิเคราะห์กราฟ
stats = graph_constructor.get_graph_statistics()
for key, value in stats.items():
    print(f"  {key}: {value}")

# 9. ใช้กับโมเดล GNN
from src.models.multi_task_gnn import MultiTaskTrafficGNN

model = MultiTaskTrafficGNN(num_features=10, hidden_dim=64)
outputs = model(data)

print(f"Congestion predictions: {outputs['congestion_logits'].shape}")
print(f"Rush hour predictions: {outputs['rush_hour_logits'].shape}")
```

---

## 📊 สถิติกราฟ

### **กราฟเครือข่ายถนนกรุงเทพ:**

```
Nodes (Intersections): 217
Edges (Road Segments): 542
Average Degree: 4.99
Graph Density: 0.023
Connected Components: 1
Diameter: 12 hops
Average Clustering: 0.15
```

### **Degree Distribution:**
```python
degrees = [d for _, d in G.degree()]

# สถิติ
min_degree: 2
max_degree: 12
mean_degree: 4.99
median_degree: 5

# Distribution:
Degree 2: 15 nodes (7%)
Degree 3: 42 nodes (19%)
Degree 4: 65 nodes (30%)
Degree 5: 58 nodes (27%)
Degree 6+: 37 nodes (17%)
```

---

## 🎓 สรุป

### **Key Concepts:**

1. **Graph Construction:**
   - Nodes = จุดตัดถนน (intersections)
   - Edges = เส้นถนน (road segments)
   - เชื่อมตามโครงสร้างจริง

2. **Adjacency Matrix:**
   - แสดงการเชื่อมต่อ
   - Normalize สำหรับ GNN
   - Sparse matrix (ไม่หนาแน่น)

3. **Neighbor Finding:**
   - K-nearest neighbors
   - Haversine distance
   - Spatial proximity

4. **PyTorch Geometric:**
   - Data object สำหรับ GNN
   - edge_index format
   - train/val/test masks

---

**สร้างเมื่อ:** 5 ตุลาคม 2025  
**เวอร์ชัน:** 1.0  
**ผู้เขียน:** Traffic GNN Classification Team
