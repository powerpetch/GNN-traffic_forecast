"""
Graph Construction Module for Traffic GNN
=========================================

This module handles:
1. Creating road network graph from processed data
2. Building adjacency matrices
3. Preparing data for PyTorch Geometric
4. Graph-based feature engineering
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_undirected
import geopandas as gpd
from shapely.geometry import Point, LineString
from typing import Dict, List, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')

class GraphConstructor:
    """Build road network graph for GNN training"""
    
    def __init__(self, road_network: gpd.GeoDataFrame = None):
        self.road_network = road_network
        self.graph = None
        self.node_mapping = {}
        self.edge_mapping = {}
        self.adjacency_matrix = None
        
    def build_road_graph(self, connection_threshold: float = 0.001) -> nx.Graph:
        """
        Build NetworkX graph from road network
        
        Args:
            connection_threshold: Distance threshold for connecting roads (in degrees)
        """
        print("Building road network graph...")
        
        if self.road_network is None:
            raise ValueError("Road network not provided")
        
        G = nx.Graph()
        
        # Extract endpoints of each road segment
        road_endpoints = {}
        
        for idx, row in self.road_network.iterrows():
            edge_id = row['edge_id']
            geom = row['geometry']
            
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                start_point = coords[0]
                end_point = coords[-1]
                
                road_endpoints[edge_id] = {
                    'start': start_point,
                    'end': end_point,
                    'geometry': geom,
                    'attributes': {
                        'highway': row.get('highway', 'unknown'),
                        'lanes': row.get('lanes', 1),
                        'length_km': row.get('length_km', 0)
                    }
                }
        
        # Create nodes (intersections) by clustering nearby endpoints
        intersections = {}
        intersection_id = 0
        
        for edge_id, road_data in road_endpoints.items():
            for endpoint_type in ['start', 'end']:
                point = road_data[endpoint_type]
                
                # Check if this point is close to existing intersection
                found_intersection = None
                for int_id, int_data in intersections.items():
                    distance = np.sqrt((point[0] - int_data['coords'][0])**2 + 
                                     (point[1] - int_data['coords'][1])**2)
                    if distance < connection_threshold:
                        found_intersection = int_id
                        break
                
                if found_intersection is None:
                    # Create new intersection
                    intersections[intersection_id] = {
                        'coords': point,
                        'connected_roads': [(edge_id, endpoint_type)]
                    }
                    intersection_id += 1
                else:
                    # Add to existing intersection
                    intersections[found_intersection]['connected_roads'].append((edge_id, endpoint_type))
        
        # Add nodes to graph
        for int_id, int_data in intersections.items():
            G.add_node(int_id, 
                      pos=int_data['coords'],
                      num_connections=len(int_data['connected_roads']))
        
        # Add edges (road segments) to graph
        for edge_id, road_data in road_endpoints.items():
            start_intersection = None
            end_intersection = None
            
            # Find which intersections this road connects
            for int_id, int_data in intersections.items():
                for road_id, endpoint_type in int_data['connected_roads']:
                    if road_id == edge_id:
                        if endpoint_type == 'start':
                            start_intersection = int_id
                        else:
                            end_intersection = int_id
            
            if start_intersection is not None and end_intersection is not None:
                G.add_edge(start_intersection, end_intersection,
                          edge_id=edge_id,
                          highway=road_data['attributes']['highway'],
                          lanes=road_data['attributes']['lanes'],
                          length_km=road_data['attributes']['length_km'])
        
        self.graph = G
        
        # Create mappings
        self.node_mapping = {node_id: idx for idx, node_id in enumerate(G.nodes())}
        self.edge_mapping = {edge_id: idx for idx, (_, _, edge_data) in enumerate(G.edges(data=True))
                           for edge_id in [edge_data.get('edge_id')] if edge_id is not None}
        
        print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def create_adjacency_matrix(self, normalize: bool = True) -> np.ndarray:
        """Create adjacency matrix from graph"""
        if self.graph is None:
            raise ValueError("Graph not built yet. Call build_road_graph() first.")
        
        # Get adjacency matrix from NetworkX
        adj_matrix = nx.adjacency_matrix(self.graph, nodelist=sorted(self.graph.nodes())).toarray()
        
        if normalize:
            # Add self-loops
            adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0])
            
            # Row normalization
            row_sums = adj_matrix.sum(axis=1)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            adj_matrix = adj_matrix / row_sums[:, np.newaxis]
        
        self.adjacency_matrix = adj_matrix
        print(f"Created adjacency matrix of shape {adj_matrix.shape}")
        
        return adj_matrix
    
    def prepare_node_features(self, processed_data: pd.DataFrame, 
                            feature_columns: List[str]) -> Dict[int, np.ndarray]:
        """
        Prepare node features from processed traffic data
        
        Args:
            processed_data: DataFrame with processed traffic data
            feature_columns: List of column names to use as features
        """
        print("Preparing node features...")
        
        # Group by edge_id and get latest features for each road
        latest_data = processed_data.groupby('edge_id').last().reset_index()
        
        node_features = {}
        
        # Map edge features to nodes (intersections)
        for node_id in self.graph.nodes():
            # Find connected edges for this node
            connected_edges = []
            for neighbor in self.graph.neighbors(node_id):
                edge_data = self.graph[node_id][neighbor]
                if 'edge_id' in edge_data:
                    connected_edges.append(edge_data['edge_id'])
            
            if connected_edges:
                # Aggregate features from connected edges
                edge_features = []
                for edge_id in connected_edges:
                    edge_data = latest_data[latest_data['edge_id'] == edge_id]
                    if not edge_data.empty:
                        features = edge_data[feature_columns].values[0]
                        edge_features.append(features)
                
                if edge_features:
                    # Average features from connected edges
                    node_features[node_id] = np.mean(edge_features, axis=0)
                else:
                    # Use zero features if no data available
                    node_features[node_id] = np.zeros(len(feature_columns))
            else:
                node_features[node_id] = np.zeros(len(feature_columns))
        
        print(f"Created features for {len(node_features)} nodes")
        return node_features
    
    def create_temporal_sequences(self, processed_data: pd.DataFrame,
                                sequence_length: int = 12,
                                prediction_length: int = 6,
                                feature_columns: List[str] = None) -> List[Dict]:
        """
        Create temporal sequences for GNN training
        
        Args:
            processed_data: Processed traffic data
            sequence_length: Number of time steps for input (T_in)
            prediction_length: Number of time steps to predict (T_out)
            feature_columns: Feature columns to use
        """
        print(f"Creating temporal sequences (T_in={sequence_length}, T_out={prediction_length})...")
        
        if feature_columns is None:
            feature_columns = [
                'mean_speed', 'median_speed', 'speed_std', 'count_probes',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend'
            ]
        
        # Sort data by edge_id and timestamp
        data_sorted = processed_data.sort_values(['edge_id', 'timestamp'])
        
        sequences = []
        
        # Group by edge_id
        for edge_id, group in data_sorted.groupby('edge_id'):
            group = group.reset_index(drop=True)
            
            # Create sliding windows
            for i in range(len(group) - sequence_length - prediction_length + 1):
                # Input sequence
                input_seq = group.iloc[i:i + sequence_length]
                # Target sequence  
                target_seq = group.iloc[i + sequence_length:i + sequence_length + prediction_length]
                
                if len(input_seq) == sequence_length and len(target_seq) == prediction_length:
                    sequence_data = {
                        'edge_id': edge_id,
                        'input_features': input_seq[feature_columns].values,
                        'target_speed': target_seq['mean_speed'].values,
                        'target_congestion': target_seq['congestion_label'].values,
                        'target_rush_hour': target_seq['is_rush_hour'].values,
                        'timestamps': target_seq['timestamp'].values,
                        'quality_scores': target_seq['quality_score'].values
                    }
                    sequences.append(sequence_data)
        
        print(f"Created {len(sequences)} temporal sequences")
        return sequences
    
    def create_pyg_data(self, sequences: List[Dict], 
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15) -> Dict[str, DataLoader]:
        """
        Create PyTorch Geometric data loaders
        
        Args:
            sequences: List of temporal sequences
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
        """
        print("Creating PyTorch Geometric data...")
        
        if self.adjacency_matrix is None:
            self.create_adjacency_matrix()
        
        # Convert to PyG format
        pyg_data_list = []
        
        for seq in sequences:
            edge_id = seq['edge_id']
            
            # Find corresponding node (intersection) for this edge
            # For simplicity, we'll use the edge_id as node_id if mapping exists
            if edge_id in self.edge_mapping:
                node_id = self.edge_mapping[edge_id]
            else:
                continue
            
            # Create node features (for all nodes, but focus on current edge)
            num_nodes = len(self.node_mapping)
            x = torch.zeros((num_nodes, seq['input_features'].shape[1]))
            
            # Set features for current node
            if node_id < num_nodes:
                x[node_id] = torch.tensor(seq['input_features'][-1], dtype=torch.float)  # Use latest features
            
            # Create edge index from adjacency matrix
            edge_index = torch.tensor(
                np.array(np.where(self.adjacency_matrix > 0)), 
                dtype=torch.long
            )
            
            # Targets
            y_speed = torch.tensor(seq['target_speed'], dtype=torch.float)
            y_congestion = torch.tensor(seq['target_congestion'], dtype=torch.long)
            y_rush_hour = torch.tensor(seq['target_rush_hour'], dtype=torch.long)
            
            # Create PyG data object
            data = Data(
                x=x,
                edge_index=edge_index,
                y_speed=y_speed,
                y_congestion=y_congestion,
                y_rush_hour=y_rush_hour,
                edge_id=edge_id,
                node_id=node_id
            )
            
            pyg_data_list.append(data)
        
        # Split data
        n_total = len(pyg_data_list)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_data = pyg_data_list[:n_train]
        val_data = pyg_data_list[n_train:n_train + n_val]
        test_data = pyg_data_list[n_train + n_val:]
        
        print(f"Split data: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def save_graph_data(self, filepath: str):
        """Save graph construction data"""
        graph_data = {
            'graph': self.graph,
            'node_mapping': self.node_mapping,
            'edge_mapping': self.edge_mapping,
            'adjacency_matrix': self.adjacency_matrix
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(graph_data, f)
        
        print(f"Graph data saved to {filepath}")
    
    def load_graph_data(self, filepath: str):
        """Load graph construction data"""
        with open(filepath, 'rb') as f:
            graph_data = pickle.load(f)
        
        self.graph = graph_data['graph']
        self.node_mapping = graph_data['node_mapping']
        self.edge_mapping = graph_data['edge_mapping']
        self.adjacency_matrix = graph_data['adjacency_matrix']
        
        print(f"Graph data loaded from {filepath}")

def create_simple_adjacency_matrix(edge_ids: List[int], similarity_threshold: float = 0.1) -> np.ndarray:
    """
    Create a simple adjacency matrix based on edge similarity
    This is a fallback when road network graph is not available
    """
    n_edges = len(edge_ids)
    adj_matrix = np.eye(n_edges)  # Start with identity matrix
    
    # Add some random connections for demonstration
    np.random.seed(42)
    for i in range(n_edges):
        # Connect to a few random nearby edges
        n_connections = np.random.randint(1, min(5, n_edges))
        connections = np.random.choice(n_edges, n_connections, replace=False)
        for j in connections:
            if i != j:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
    
    # Normalize
    row_sums = adj_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1
    adj_matrix = adj_matrix / row_sums[:, np.newaxis]
    
    return adj_matrix

if __name__ == "__main__":
    # Example usage
    from data_processor import TrafficDataProcessor
    
    # Process data
    processor = TrafficDataProcessor()
    processed_data = processor.process_all_data()
    
    # Build graph
    constructor = GraphConstructor(processor.road_network)
    graph = constructor.build_road_graph()
    adj_matrix = constructor.create_adjacency_matrix()
    
    # Create sequences
    sequences = constructor.create_temporal_sequences(processed_data)
    
    # Create PyG data
    pyg_data = constructor.create_pyg_data(sequences)
    
    # Save graph data
    constructor.save_graph_data("d:/user/Data_project/Traffic_GNN_Classification/outputs/graph_data.pkl")