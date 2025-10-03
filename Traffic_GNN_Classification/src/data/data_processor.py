"""
Data Processing Module for Traffic GNN Classification
====================================================

This module handles:
1. Map-matching GPS probes to HOTOSM road network
2. Data aggregation (5-min bins)
3. Feature engineering (temporal, spatial, lag features)
4. Label creation for traffic congestion and rush hour classification
5. Graph construction for GNN
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import pyproj
from rtree import index
import datetime
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class TrafficDataProcessor:
    """Main class for processing traffic data for GNN classification"""
    
    def __init__(self, data_path: str = "d:/user/Data_project/GNN_fore/src/data/raw"):
        self.data_path = data_path
        self.road_network = None
        self.spatial_index = None
        self.processed_data = None
        
        # Classification thresholds
        self.speed_thresholds = {
            'gridlock': 10,      # Speed < 10 km/h
            'congested': 25,     # Speed 10-25 km/h  
            'moderate': 40,      # Speed 25-40 km/h
            'free_flow': 999     # Speed > 40 km/h
        }
        
        # Rush hour definitions
        self.rush_hours = [
            (7, 9),    # Morning rush: 7-9 AM
            (17, 19)   # Evening rush: 5-7 PM
        ]
        
    def load_probe_data(self, date_pattern: str = "2024*") -> pd.DataFrame:
        """Load and combine probe data from multiple files"""
        import glob
        import os
        
        probe_files = glob.glob(os.path.join(self.data_path, f"PROBE-{date_pattern}", "*.csv.out"))
        
        if not probe_files:
            # Generate synthetic probe data for demonstration
            return self._generate_synthetic_probe_data()
        
        probe_data_list = []
        
        for file in probe_files[:10]:  # Process first 10 files for demo
            try:
                # Assuming probe data format: timestamp, device_id, lat, lon, speed_kph, heading
                df = pd.read_csv(file, header=None, 
                               names=['timestamp', 'device_id', 'lat', 'lon', 'speed_kph', 'heading'])
                df['file_date'] = os.path.basename(file)[:8]
                probe_data_list.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
        
        if probe_data_list:
            combined_data = pd.concat(probe_data_list, ignore_index=True)
        else:
            combined_data = self._generate_synthetic_probe_data()
            
        # Convert timestamp
        combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'], errors='coerce')
        combined_data = combined_data.dropna(subset=['timestamp', 'lat', 'lon', 'speed_kph'])
        
        # Filter reasonable speed values
        combined_data = combined_data[
            (combined_data['speed_kph'] >= 0) & 
            (combined_data['speed_kph'] <= 150)
        ]
        
        print(f"Loaded {len(combined_data)} probe records")
        return combined_data
    
    def _generate_synthetic_probe_data(self) -> pd.DataFrame:
        """Generate synthetic probe data for demonstration"""
        np.random.seed(42)
        
        # Bangkok bounds
        lat_min, lat_max = 13.65, 13.85
        lon_min, lon_max = 100.45, 100.65
        
        n_records = 50000
        n_devices = 1000
        
        # Generate timestamps over 7 days
        start_date = datetime.datetime(2024, 1, 1)
        timestamps = []
        for i in range(n_records):
            random_hours = np.random.uniform(0, 7*24)  # 7 days
            timestamps.append(start_date + datetime.timedelta(hours=random_hours))
        
        data = {
            'timestamp': timestamps,
            'device_id': np.random.randint(1, n_devices, n_records),
            'lat': np.random.uniform(lat_min, lat_max, n_records),
            'lon': np.random.uniform(lon_min, lon_max, n_records),
            'speed_kph': np.random.gamma(2, 15, n_records),  # Realistic speed distribution
            'heading': np.random.uniform(0, 360, n_records)
        }
        
        df = pd.DataFrame(data)
        
        # Add rush hour effects
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        rush_mask = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 17) & (df['hour'] <= 19))
        df.loc[rush_mask, 'speed_kph'] *= 0.6  # Slower during rush hours
        
        # Clip speeds
        df['speed_kph'] = np.clip(df['speed_kph'], 0, 120)
        
        print(f"Generated {len(df)} synthetic probe records")
        return df
    
    def load_road_network(self) -> gpd.GeoDataFrame:
        """Load HOTOSM road network data"""
        try:
            # Try to load actual HOTOSM data
            hotosm_path = os.path.join(self.data_path, "hotosm_tha_roads_lines_geojson")
            if os.path.exists(hotosm_path):
                gdf = gpd.read_file(hotosm_path)
            else:
                # Generate synthetic road network
                gdf = self._generate_synthetic_road_network()
                
        except Exception as e:
            print(f"Error loading road network: {e}")
            gdf = self._generate_synthetic_road_network()
        
        # Ensure required columns exist
        required_cols = ['highway', 'lanes', 'oneway', 'length_km']
        for col in required_cols:
            if col not in gdf.columns:
                if col == 'highway':
                    gdf[col] = np.random.choice(['primary', 'secondary', 'tertiary', 'residential'], len(gdf))
                elif col == 'lanes':
                    gdf[col] = np.random.randint(1, 5, len(gdf))
                elif col == 'oneway':
                    gdf[col] = np.random.choice([True, False], len(gdf))
                elif col == 'length_km':
                    gdf[col] = gdf.geometry.length * 111  # Rough conversion to km
        
        # Create edge IDs
        gdf['edge_id'] = range(len(gdf))
        
        self.road_network = gdf
        self._build_spatial_index()
        
        print(f"Loaded {len(gdf)} road segments")
        return gdf
    
    def _generate_synthetic_road_network(self) -> gpd.GeoDataFrame:
        """Generate synthetic road network for Bangkok"""
        np.random.seed(42)
        
        # Bangkok bounds
        lat_min, lat_max = 13.65, 13.85
        lon_min, lon_max = 100.45, 100.65
        
        n_roads = 500
        geometries = []
        highway_types = []
        lanes = []
        oneways = []
        
        for i in range(n_roads):
            # Create random road segments
            start_lat = np.random.uniform(lat_min, lat_max)
            start_lon = np.random.uniform(lon_min, lon_max)
            
            # Random direction and length
            direction = np.random.uniform(0, 2*np.pi)
            length = np.random.uniform(0.001, 0.01)  # In degrees
            
            end_lat = start_lat + length * np.sin(direction)
            end_lon = start_lon + length * np.cos(direction)
            
            # Create LineString
            line = LineString([(start_lon, start_lat), (end_lon, end_lat)])
            geometries.append(line)
            
            # Random attributes
            highway_types.append(np.random.choice(['primary', 'secondary', 'tertiary', 'residential']))
            lanes.append(np.random.randint(1, 5))
            oneways.append(np.random.choice([True, False]))
        
        gdf = gpd.GeoDataFrame({
            'highway': highway_types,
            'lanes': lanes,
            'oneway': oneways,
            'geometry': geometries
        })
        
        gdf['length_km'] = gdf.geometry.length * 111  # Rough conversion
        
        return gdf
    
    def _build_spatial_index(self):
        """Build spatial index for efficient map-matching"""
        self.spatial_index = index.Index()
        
        for idx, row in self.road_network.iterrows():
            bounds = row.geometry.bounds
            self.spatial_index.insert(idx, bounds)
    
    def map_match_probes(self, probe_data: pd.DataFrame, max_distance: float = 50.0) -> pd.DataFrame:
        """Map-match GPS probes to road network"""
        print("Starting map-matching process...")
        
        matched_probes = []
        
        for idx, probe in probe_data.iterrows():
            if idx % 10000 == 0:
                print(f"Processed {idx}/{len(probe_data)} probes")
            
            probe_point = Point(probe['lon'], probe['lat'])
            
            # Find candidate roads using spatial index
            candidates = list(self.spatial_index.intersection(probe_point.bounds))
            
            if not candidates:
                continue
            
            # Find closest road
            min_distance = float('inf')
            best_match = None
            
            for candidate_idx in candidates:
                road_geom = self.road_network.iloc[candidate_idx].geometry
                
                # Calculate distance
                distance = probe_point.distance(road_geom) * 111000  # Convert to meters
                
                if distance < min_distance and distance <= max_distance:
                    min_distance = distance
                    best_match = candidate_idx
            
            if best_match is not None:
                matched_probe = probe.copy()
                matched_probe['edge_id'] = self.road_network.iloc[best_match]['edge_id']
                matched_probe['match_distance'] = min_distance
                matched_probe['highway'] = self.road_network.iloc[best_match]['highway']
                matched_probe['lanes'] = self.road_network.iloc[best_match]['lanes']
                matched_probes.append(matched_probe)
        
        matched_df = pd.DataFrame(matched_probes)
        print(f"Successfully matched {len(matched_df)} probes to road network")
        
        return matched_df
    
    def aggregate_data(self, matched_data: pd.DataFrame, bin_minutes: int = 5) -> pd.DataFrame:
        """Aggregate probe data into time bins"""
        print(f"Aggregating data into {bin_minutes}-minute bins...")
        
        # Create time bins
        matched_data['time_bin'] = matched_data['timestamp'].dt.floor(f'{bin_minutes}min')
        
        # Aggregate by edge_id and time_bin
        agg_functions = {
            'speed_kph': ['mean', 'median', 'std', 'count'],
            'match_distance': 'mean',
            'highway': 'first',
            'lanes': 'first'
        }
        
        aggregated = matched_data.groupby(['edge_id', 'time_bin']).agg(agg_functions).reset_index()
        
        # Flatten column names
        aggregated.columns = ['edge_id', 'timestamp'] + [
            f"{col[0]}_{col[1]}" if col[1] != '' else col[0] 
            for col in aggregated.columns[2:]
        ]
        
        # Rename columns for clarity
        column_mapping = {
            'speed_kph_mean': 'mean_speed',
            'speed_kph_median': 'median_speed', 
            'speed_kph_std': 'speed_std',
            'speed_kph_count': 'count_probes',
            'match_distance_mean': 'avg_match_distance',
            'highway_first': 'highway',
            'lanes_first': 'lanes'
        }
        
        aggregated = aggregated.rename(columns=column_mapping)
        
        # Calculate quality score
        k = 5  # Minimum probes for good quality
        aggregated['quality_score'] = np.minimum(1.0, aggregated['count_probes'] / k)
        
        # Fill NaN speed_std with 0
        aggregated['speed_std'] = aggregated['speed_std'].fillna(0)
        
        print(f"Created {len(aggregated)} aggregated records")
        return aggregated
    
    def create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features"""
        print("Creating temporal features...")
        
        df = data.copy()
        
        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        
        # Cyclical encoding for time
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # One-hot encode day of week
        dow_dummies = pd.get_dummies(df['day_of_week'], prefix='dow')
        df = pd.concat([df, dow_dummies], axis=1)
        
        # Holiday indicator (simplified)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    def create_lag_features(self, data: pd.DataFrame, lag_periods: List[int] = [1, 2, 3, 6, 12]) -> pd.DataFrame:
        """Create lag features for time series"""
        print("Creating lag features...")
        
        df = data.copy()
        df = df.sort_values(['edge_id', 'timestamp'])
        
        for lag in lag_periods:
            # Speed lags
            df[f'speed_lag_{lag}'] = df.groupby('edge_id')['mean_speed'].shift(lag)
            
            # Rolling statistics
            if lag >= 3:
                df[f'speed_roll_mean_{lag}'] = df.groupby('edge_id')['mean_speed'].rolling(lag, min_periods=1).mean().values
                df[f'speed_roll_std_{lag}'] = df.groupby('edge_id')['mean_speed'].rolling(lag, min_periods=1).std().values
        
        # Fill NaN values with forward fill, then backward fill
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df.groupby('edge_id')[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def create_spatial_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create spatial features from road network"""
        print("Creating spatial features...")
        
        df = data.copy()
        
        # One-hot encode highway types
        highway_dummies = pd.get_dummies(df['highway'], prefix='highway')
        df = pd.concat([df, highway_dummies], axis=1)
        
        # Lane categories
        df['lanes_cat'] = pd.cut(df['lanes'], bins=[0, 1, 2, 4, 10], 
                               labels=['single', 'double', 'multi', 'highway'])
        lane_dummies = pd.get_dummies(df['lanes_cat'], prefix='lanes')
        df = pd.concat([df, lane_dummies], axis=1)
        
        return df
    
    def create_classification_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create classification labels for traffic congestion and rush hour"""
        print("Creating classification labels...")
        
        df = data.copy()
        
        # Traffic congestion classification based on speed
        def classify_congestion(speed):
            if pd.isna(speed):
                return 'unknown'
            elif speed < self.speed_thresholds['gridlock']:
                return 'gridlock'
            elif speed < self.speed_thresholds['congested']:
                return 'congested' 
            elif speed < self.speed_thresholds['moderate']:
                return 'moderate'
            else:
                return 'free_flow'
        
        df['congestion_level'] = df['mean_speed'].apply(classify_congestion)
        
        # Rush hour classification
        def is_rush_hour(hour):
            for start, end in self.rush_hours:
                if start <= hour < end:
                    return 1
            return 0
        
        df['is_rush_hour'] = df['hour'].apply(is_rush_hour)
        
        # Create numeric labels for modeling
        congestion_mapping = {'gridlock': 0, 'congested': 1, 'moderate': 2, 'free_flow': 3, 'unknown': -1}
        df['congestion_label'] = df['congestion_level'].map(congestion_mapping)
        
        # Remove unknown labels
        df = df[df['congestion_label'] != -1]
        
        print(f"Congestion distribution:\n{df['congestion_level'].value_counts()}")
        print(f"Rush hour distribution:\n{df['is_rush_hour'].value_counts()}")
        
        return df
    
    def process_all_data(self) -> pd.DataFrame:
        """Main processing pipeline"""
        print("=== Starting complete data processing pipeline ===")
        
        # Load data
        probe_data = self.load_probe_data()
        self.load_road_network()
        
        # Map matching
        matched_data = self.map_match_probes(probe_data)
        
        if len(matched_data) == 0:
            raise ValueError("No probes could be matched to road network")
        
        # Aggregate data
        aggregated_data = self.aggregate_data(matched_data)
        
        # Feature engineering
        temporal_data = self.create_temporal_features(aggregated_data)
        lag_data = self.create_lag_features(temporal_data)
        spatial_data = self.create_spatial_features(lag_data)
        
        # Create labels
        final_data = self.create_classification_labels(spatial_data)
        
        # Store processed data
        self.processed_data = final_data
        
        print(f"=== Processing complete! Final dataset: {len(final_data)} records ===")
        
        return final_data
    
    def save_processed_data(self, filepath: str):
        """Save processed data to file"""
        if self.processed_data is not None:
            self.processed_data.to_pickle(filepath)
            print(f"Processed data saved to {filepath}")
        else:
            print("No processed data to save. Run process_all_data() first.")
    
    def load_processed_data(self, filepath: str) -> pd.DataFrame:
        """Load processed data from file"""
        try:
            self.processed_data = pd.read_pickle(filepath)
            print(f"Loaded processed data from {filepath}")
            return self.processed_data
        except Exception as e:
            print(f"Error loading processed data: {e}")
            return None
    
    # Helper methods for testing compatibility
    def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Helper method for testing - calls create_temporal_features"""
        return self.create_temporal_features(data)
    
    def _create_congestion_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Helper method for testing - creates congestion labels"""
        data = data.copy()
        
        def classify_congestion(speed):
            if speed < self.speed_thresholds['gridlock']:
                return 0  # Gridlock
            elif speed < self.speed_thresholds['congested']:
                return 1  # Congested
            elif speed < self.speed_thresholds['moderate']:
                return 2  # Moderate
            else:
                return 3  # Free flow
        
        if 'speed' in data.columns:
            data['congestion_level'] = data['speed'].apply(classify_congestion)
        elif 'mean_speed' in data.columns:
            data['congestion_level'] = data['mean_speed'].apply(classify_congestion)
        else:
            # Default to moderate if no speed data
            data['congestion_level'] = 2
            
        return data
    
    def _create_rush_hour_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Helper method for testing - creates rush hour labels"""
        data = data.copy()
        
        def is_rush_hour(hour):
            for start, end in self.rush_hours:
                if start <= hour < end:
                    return 1
            return 0
        
        if 'timestamp' in data.columns:
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            data['is_rush_hour'] = data['hour'].apply(is_rush_hour)
        elif 'hour_sin' in data.columns:
            # Reconstruct hour from sin/cos features
            hour_approx = np.arcsin(data['hour_sin']) * 24 / (2 * np.pi)
            hour_approx = hour_approx.clip(0, 23).round().astype(int)
            data['is_rush_hour'] = hour_approx.apply(is_rush_hour)
        else:
            # Default to non-rush hour if no time data
            data['is_rush_hour'] = 0
            
        return data

if __name__ == "__main__":
    # Example usage
    processor = TrafficDataProcessor()
    processed_data = processor.process_all_data()
    
    # Save processed data
    processor.save_processed_data("d:/user/Data_project/Traffic_GNN_Classification/outputs/processed_data.pkl")