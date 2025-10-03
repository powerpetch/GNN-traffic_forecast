"""
Real Road Data Loader
Loads actual Bangkok road names and data from HOTOSM and iTIC sources
"""

import json
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional

class BangkokRoadDataLoader:
    """Load and process real Bangkok road data"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.hotosm_file = os.path.join(data_dir, "hotosm_tha_roads_lines_geojson", "hotosm_tha_roads_lines_geojson.geojson")
        self.thailand_table_file = os.path.join(data_dir, "Thailand_T19_v3.2_flat_Thai.xlsx")
        self.itic_events_dir = os.path.join(data_dir, "iTIC-Longdo-Traffic-events-2022")
    
    def load_hotosm_roads(self, max_roads: int = 1000) -> pd.DataFrame:
        """Load real road data from HOTOSM GeoJSON"""
        
        if not os.path.exists(self.hotosm_file):
            print(f"HOTOSM file not found: {self.hotosm_file}")
            return self._get_fallback_roads()
        
        try:
            with open(self.hotosm_file, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
            
            roads = []
            road_count = 0
            
            for feature in geojson_data['features']:
                if road_count >= max_roads:
                    break
                
                props = feature['properties']
                geom = feature['geometry']
                
                # Skip roads without names or coordinates
                road_name = props.get('name', '')
                if not road_name or geom['type'] != 'LineString':
                    continue
                
                # Get coordinates (use middle point of the road)
                coords = geom['coordinates']
                if len(coords) < 2:
                    continue
                
                mid_point = coords[len(coords) // 2]
                lon, lat = mid_point[0], mid_point[1]
                
                # Filter to Bangkok area approximately
                if not (13.5 <= lat <= 14.0 and 100.3 <= lon <= 100.8):
                    continue
                
                # Extract road information
                highway_type = props.get('highway', 'tertiary')
                name_en = props.get('name:en', road_name)
                name_th = props.get('name:th', road_name)
                surface = props.get('surface', 'asphalt')
                maxspeed = props.get('maxspeed', None)
                
                # Determine speed limit based on highway type
                if maxspeed:
                    try:
                        speed_limit = int(maxspeed.split()[0]) if isinstance(maxspeed, str) else int(maxspeed)
                    except:
                        speed_limit = self._get_default_speed_limit(highway_type)
                else:
                    speed_limit = self._get_default_speed_limit(highway_type)
                
                roads.append({
                    'road_id': f"osm_{road_count:05d}",
                    'name': road_name,
                    'name_en': name_en,
                    'name_th': name_th,
                    'lat': lat,
                    'lon': lon,
                    'highway': highway_type,
                    'surface': surface,
                    'speed_limit': speed_limit,
                    'osm_id': props.get('id', ''),
                    'length_km': self._estimate_road_length(coords)
                })
                
                road_count += 1
            
            df = pd.DataFrame(roads)
            print(f"Loaded {len(df)} real roads from HOTOSM data")
            return df
            
        except Exception as e:
            print(f"Error loading HOTOSM data: {e}")
            return self._get_fallback_roads()
    
    def load_thailand_location_table(self) -> Optional[pd.DataFrame]:
        """Load Thailand location reference table"""
        
        if not os.path.exists(self.thailand_table_file):
            print(f"Thailand location table not found: {self.thailand_table_file}")
            return None
        
        try:
            df = pd.read_excel(self.thailand_table_file)
            print(f"Loaded Thailand location table with {len(df)} entries")
            return df
        except Exception as e:
            print(f"Error loading Thailand location table: {e}")
            return None
    
    def load_itic_traffic_events(self, month: str = "01") -> Optional[pd.DataFrame]:
        """Load iTIC traffic events for a specific month"""
        
        month_dir = os.path.join(self.itic_events_dir, month.zfill(2))
        
        if not os.path.exists(month_dir):
            print(f"iTIC events directory not found: {month_dir}")
            return None
        
        try:
            events = []
            files = [f for f in os.listdir(month_dir) if f.endswith('.json')][:100]  # Limit files
            
            for file in files:
                file_path = os.path.join(month_dir, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        event_data = json.load(f)
                    
                    if isinstance(event_data, list):
                        events.extend(event_data)
                    elif isinstance(event_data, dict):
                        events.append(event_data)
                        
                except Exception as e:
                    print(f"Error reading event file {file}: {e}")
                    continue
            
            if events:
                df = pd.DataFrame(events)
                print(f"Loaded {len(df)} traffic events from iTIC data")
                return df
            else:
                print("No traffic events found")
                return None
                
        except Exception as e:
            print(f"Error loading iTIC events: {e}")
            return None
    
    def _get_default_speed_limit(self, highway_type: str) -> int:
        """Get default speed limit based on highway type"""
        speed_map = {
            'motorway': 120,
            'trunk': 90,
            'primary': 70,
            'secondary': 60,
            'tertiary': 50,
            'residential': 40,
            'living_street': 30,
            'service': 20
        }
        return speed_map.get(highway_type, 50)
    
    def _estimate_road_length(self, coordinates: List) -> float:
        """Estimate road length from coordinates (simplified)"""
        if len(coordinates) < 2:
            return 0.5
        
        # Simple distance calculation between first and last point
        start = coordinates[0]
        end = coordinates[-1]
        
        # Rough distance calculation (not precise, but good enough for demo)
        lat_diff = abs(end[1] - start[1])
        lon_diff = abs(end[0] - start[0])
        
        # Convert to km (very rough approximation)
        distance_km = ((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 111
        
        return max(0.1, min(distance_km, 20))  # Clamp between 0.1 and 20 km
    
    def _get_fallback_roads(self) -> pd.DataFrame:
        """Fallback road data if HOTOSM data cannot be loaded"""
        
        fallback_roads = [
            {"road_id": "fb_001", "name": "Sukhumvit Road", "name_en": "Sukhumvit Road", 
             "name_th": "ถนนสุขุมวิท", "lat": 13.7563, "lon": 100.5018, "highway": "trunk", "speed_limit": 80},
            {"road_id": "fb_002", "name": "Phetchaburi Road", "name_en": "Phetchaburi Road", 
             "name_th": "ถนนเพชรบุรี", "lat": 13.7539, "lon": 100.5388, "highway": "primary", "speed_limit": 60},
            {"road_id": "fb_003", "name": "Rama IV Road", "name_en": "Rama IV Road", 
             "name_th": "ถนนพระราม 4", "lat": 13.7307, "lon": 100.5418, "highway": "trunk", "speed_limit": 80},
            {"road_id": "fb_004", "name": "Silom Road", "name_en": "Silom Road", 
             "name_th": "ถนนสีลม", "lat": 13.7307, "lon": 100.5338, "highway": "primary", "speed_limit": 50},
            {"road_id": "fb_005", "name": "Sathorn Road", "name_en": "Sathorn Road", 
             "name_th": "ถนนสาทร", "lat": 13.7209, "lon": 100.5234, "highway": "primary", "speed_limit": 60},
        ]
        
        for road in fallback_roads:
            road['surface'] = 'asphalt'
            road['length_km'] = np.random.uniform(1.0, 5.0)
            road['osm_id'] = ''
        
        print("Using fallback road data (limited set)")
        return pd.DataFrame(fallback_roads)

def create_enhanced_road_network(language_pref: str = "English") -> pd.DataFrame:
    """Create enhanced road network with real data"""
    
    loader = BangkokRoadDataLoader()
    
    # Load real road data
    road_df = loader.load_hotosm_roads(max_roads=500)  # Limit for performance
    
    # Set display names based on language preference
    for idx, row in road_df.iterrows():
        if language_pref == "English":
            road_df.at[idx, 'display_name'] = row['name_en'] if row['name_en'] else row['name']
        elif language_pref == "Thai":
            road_df.at[idx, 'display_name'] = row['name_th'] if row['name_th'] else row['name']
        else:  # Both
            name_en = row['name_en'] if row['name_en'] else row['name']
            name_th = row['name_th'] if row['name_th'] else ''
            if name_th and name_th != name_en:
                road_df.at[idx, 'display_name'] = f"{name_en} ({name_th})"
            else:
                road_df.at[idx, 'display_name'] = name_en
    
    return road_df

if __name__ == "__main__":
    # Test the loader
    loader = BangkokRoadDataLoader()
    
    print("Testing HOTOSM road loading...")
    roads = loader.load_hotosm_roads(max_roads=50)
    print(f"Loaded {len(roads)} roads")
    print(roads.head())
    
    print("\nTesting Thailand location table...")
    location_table = loader.load_thailand_location_table()
    if location_table is not None:
        print(f"Loaded {len(location_table)} location entries")
    
    print("\nTesting iTIC traffic events...")
    events = loader.load_itic_traffic_events()
    if events is not None:
        print(f"Loaded {len(events)} traffic events")
