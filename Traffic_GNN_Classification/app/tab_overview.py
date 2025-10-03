"""
Tab 1: Overview - Traffic forecast overview and main metrics
"""

import streamlit as st
import numpy as np
from datetime import datetime
from streamlit_folium import st_folium

from config import COLORS
from utils import create_metric_card, format_confidence, show_loading_spinner
from data_processing import generate_time_based_predictions

def render_overview_tab(data, settings, forecast_time, selected_model):
    """Render the overview tab with traffic forecast and metrics"""
    
    st.markdown("**GNN Traffic Forecast - Comprehensive Bangkok Coverage**")
    
    # Store forecast time for synchronization with GNN graph
    st.session_state.current_forecast_time = forecast_time
    
    # Import here to avoid circular imports
    from visualization import create_traffic_map
    
    # Calculate real metrics from DYNAMIC predictions based on forecast time
    try:
        current_time = datetime.now()
        forecast_hour = forecast_time.hour
        is_weekend = current_time.weekday() >= 5
        is_rush_hour = (7 <= forecast_hour <= 9 or 17 <= forecast_hour <= 19) and not is_weekend
        is_night = forecast_hour >= 22 or forecast_hour <= 6
        
        # **FIXED**: Generate predictions ONCE and cache for sharing with GNN Graph tab
        pred_cache_key = f"predictions_{selected_model}_{forecast_hour}"
        if pred_cache_key not in st.session_state:
            st.session_state[pred_cache_key] = generate_time_based_predictions(data, forecast_hour, is_weekend, is_rush_hour, is_night)
        dynamic_preds = st.session_state[pred_cache_key]
        
        # **FIXED**: Clear old map caches and generate map with cached predictions
        map_cache_key = f"traffic_map_{forecast_time.hour}_{selected_model}"
        keys_to_remove = [k for k in st.session_state.keys() if k.startswith('traffic_map_')]
        for key in keys_to_remove:
            if key != map_cache_key:
                del st.session_state[key]
        
        # Generate map with shared predictions
        if map_cache_key not in st.session_state:
            with show_loading_spinner(f"Loading Bangkok traffic map for {forecast_time.strftime('%H:%M')}..."):
                st.session_state[map_cache_key] = create_traffic_map(data, settings, dynamic_preds)
        
        if st.session_state[map_cache_key]:
            st_folium(st.session_state[map_cache_key], width=1400, height=600, key=f"traffic_map_{forecast_time.hour}")
        else:
            st.info("Map temporarily unavailable")
        
        col1, col2, col3 = st.columns(3)
        congested_count = np.sum(np.array(dynamic_preds['congestion']) <= 1)  # Gridlock + Congested
        avg_confidence = np.mean(dynamic_preds['confidence']) * 100
        
        # Show current forecast status
        time_status = "🌅 Morning Rush" if (7 <= forecast_hour <= 9) else "🌆 Evening Rush" if (17 <= forecast_hour <= 19) else "🌙 Night Hours" if is_night else "☀️ Normal Hours"
        st.info(f"⏰ **Forecasting for {forecast_time.strftime('%H:%M')}** - {time_status}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
        # Fallback values
        congested_count = 25
        avg_confidence = 85.0
        st.info("⏰ **Using fallback predictions**")
    
    with col1:
        create_metric_card(
            "Bangkok Locations",
            f"{len(data['locations'])}",
            color=COLORS['primary_blue']
        )
    
    with col2:
        create_metric_card(
            "GNN Confidence", 
            format_confidence(avg_confidence / 100),
            color=COLORS['primary_green']
        )
    
    with col3:
        create_metric_card(
            "Congested Areas",
            f"{congested_count}",
            color=COLORS['primary_red']
        )