"""
Tab 5: GNN Graph - Network visualization and graph analysis
"""

import streamlit as st
import numpy as np
from datetime import datetime

from config import COLORS, TRAFFIC_LEVELS
from utils import create_metric_card, create_status_card, show_loading_spinner
from data_processing import generate_time_based_predictions
from visualization import create_network_visualization

def render_gnn_graph_tab(data, settings, selected_model):
    """Render the GNN graph visualization tab"""
    
    # Professional header
    st.markdown(f"""
    <div style="background: {COLORS['primary_blue']}; padding: 2rem; border-radius: 8px; margin-bottom: 2rem; border: 1px solid rgba(0,0,0,0.1);">
        <h2 style="color: #ffffff; margin: 0; font-weight: 600;">GNN Network Graph Visualization</h2>
        <p style="color: #ffffff; margin: 0.5rem 0 0 0; opacity: 0.9;">Interactive Bangkok traffic network with real-time GNN predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # **FIXED**: Use forecast time from settings (same as Live Traffic Map)
    forecast_time = settings['forecast_time']
    forecast_hour = forecast_time.hour
    current_time = datetime.now()
    is_weekend = current_time.weekday() >= 5
    is_rush_hour = (7 <= forecast_hour <= 9 or 17 <= forecast_hour <= 19) and not is_weekend
    is_night = forecast_hour >= 22 or forecast_hour <= 6
    
    # **FIXED**: Use CACHED predictions from session state (generated in overview tab)
    # This ensures EXACT same predictions as Live Traffic Map!
    pred_cache_key = f"predictions_{selected_model}_{forecast_hour}"
    if pred_cache_key not in st.session_state:
        with show_loading_spinner("Generating network predictions..."):
            st.session_state[pred_cache_key] = generate_time_based_predictions(data, forecast_hour, is_weekend, is_rush_hour, is_night)
    dynamic_preds = st.session_state[pred_cache_key]
    
    # Debug info
    st.info(f"Using predictions: {pred_cache_key} | Sample: {dynamic_preds['congestion'][:5].tolist()}")
    
    # Status Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Model Status
        create_status_card(
            "Model Status",
            "Active and Synchronized",
            COLORS['primary_green']
        )
    
    with col2:
        # Synchronization Status  
        create_status_card(
            "Synchronization",
            f"Showing same {len(data['locations'])} locations as Live Traffic Map",
            COLORS['primary_blue']
        )
    
    with col3:
        # Traffic Status
        time_status = "Rush Hour Active" if is_rush_hour else "Normal Traffic Flow"
        status_color = COLORS['primary_red'] if is_rush_hour else COLORS['primary_green']
        create_status_card(
            "Traffic Status", 
            time_status,
            status_color
        )
    
    # Network Visualization
    st.markdown("### Bangkok Traffic Network")
    st.markdown(f"*Displaying {len(data['locations'])} locations - Same data as Live Traffic Map*")
    
    try:
        # **FIXED**: Use SAME cached predictions as Live Traffic Map
        # This ensures identical node colors between map and graph
        network_fig = create_network_visualization(data, forecast_hour, is_weekend, is_rush_hour, is_night, dynamic_preds)
        st.plotly_chart(network_fig, use_container_width=True, key=f"network_graph_{selected_model}_{forecast_hour}")
    except Exception as e:
        st.error(f"Network visualization error: {e}")
        st.info("Network graph temporarily unavailable")
    
    # Network Statistics and Traffic Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Professional traffic distribution display
        st.markdown("#### Traffic Congestion Levels")
        
        # Calculate congestion distribution
        congestion_counts = np.bincount(dynamic_preds['congestion'], minlength=4)
        labels = TRAFFIC_LEVELS['labels']
        colors = TRAFFIC_LEVELS['colors']
        
        for i, (label, count, color) in enumerate(zip(labels, congestion_counts, colors)):
            percentage = (count / len(dynamic_preds['congestion'])) * 100
            st.markdown(f"""
            <div style="background: {color}; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; display: flex; align-items: center; border: 1px solid rgba(0,0,0,0.1);">
                <div style="flex: 1;">
                    <div style="font-weight: 600; color: #ffffff; font-size: 1.1rem;">{label}</div>
                    <div style="color: #ffffff; margin-top: 0.2rem; opacity: 0.9;">{count} locations • {percentage:.1f}%</div>
                </div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #ffffff;">{percentage:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Congestion hotspots
        congested_locations = [i for i, pred in enumerate(dynamic_preds['congestion']) if pred <= 1]
        if congested_locations:
            st.markdown("**Current Congestion Hotspots:**")
            for idx in congested_locations[:3]:  # Show top 3
                if idx < len(data['location_names']):
                    name = data['location_names'][idx]
                    confidence = dynamic_preds['confidence'][idx]
                    st.markdown(f"• {name} (Confidence: {confidence:.1%})")
    
    with col2:
        # Professional summary cards
        st.markdown("#### Network Summary")
        
        district_counts = {}
        for district in data.get('location_districts', []):
            district_counts[district] = district_counts.get(district, 0) + 1
        
        # Districts coverage card
        st.markdown(f"""
        <div style="background: {COLORS['primary_purple']}; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid rgba(0,0,0,0.1);">
            <div style="color: #ffffff; font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">Bangkok Coverage</div>
            <div style="color: #ffffff; font-size: 0.9rem; opacity: 0.9;">{len(district_counts)} districts monitored</div>
            <div style="color: #ffffff; font-weight: 600; margin-top: 0.3rem;">{len(data['locations'])} total locations</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Rush hour status card
        rush_hour_count = np.sum(dynamic_preds['rush_hour'])
        rush_hour_percentage = (rush_hour_count / len(dynamic_preds['rush_hour'])) * 100
        time_status = "Rush Hour" if is_rush_hour else "Normal Traffic"
        rush_color = COLORS['primary_red'] if is_rush_hour else COLORS['primary_green']
        
        st.markdown(f"""
        <div style="background: {rush_color}; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid rgba(0,0,0,0.1);">
            <div style="color: #ffffff; font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">{time_status}</div>
            <div style="color: #ffffff; font-size: 0.9rem; opacity: 0.9;">{rush_hour_count} active locations</div>
            <div style="color: #ffffff; font-weight: 600; margin-top: 0.3rem;">{rush_hour_percentage:.1f}% activity</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Top districts mini list with professional styling
        st.markdown("""
        <h4 style="color: #2c3e50; font-weight: 600; margin-bottom: 1rem; font-size: 1.1rem;">
            Top Monitored Areas
        </h4>
        """, unsafe_allow_html=True)
        
        top_districts = sorted(district_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        colors = [COLORS['primary_blue'], COLORS['primary_green'], COLORS['primary_orange']]  # Blue, Green, Orange for top 3
        
        for i, (district, count) in enumerate(top_districts, 1):
            color = colors[i-1] if i <= 3 else '#95a5a6'
            st.markdown(f"""
            <div style="background: {color}; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border: 1px solid rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 600; color: #ffffff; font-size: 1rem;">{i}. {district}</div>
                        <div style="color: #ffffff; font-size: 0.85rem; margin-top: 0.2rem; opacity: 0.9;">{count} monitoring points</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.2); color: #ffffff; padding: 0.3rem 0.6rem; border-radius: 20px; font-weight: 600; font-size: 0.8rem;">
                        #{i}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # GNN Architecture Visualization
    st.markdown("---")
    st.markdown("### GNN Model Architecture")
    st.markdown("*Visualizing how the Graph Neural Network processes traffic data*")
    
    # Import visualization function
    from visualization import create_gnn_architecture_visualization
    
    try:
        arch_fig = create_gnn_architecture_visualization(selected_model)
        st.plotly_chart(arch_fig, use_container_width=True, key=f"gnn_architecture_{selected_model}")
        
        # Add explanation box below the graph
        st.info("**How GNN Predicts Traffic:**")
        st.markdown("""
        **1. Input Layer:** Location features (coordinates, time, day, weather)
        
        **2. Graph Conv 1:** Aggregates neighbor road information
        
        **3. Attention Layer:** Focuses on important connections
        
        **4. Graph Conv 2:** Refines predictions with weighted data
        
        **5. Output Layer:** Congestion (4 levels), Rush hour, Confidence
        
        ---
        *Edge opacity/thickness = connection weights • Hover over nodes for details*
        """)
        
    except Exception as e:
        st.error(f"Architecture visualization error: {e}")
    
    # Network Analysis
    st.markdown("---")
    st.markdown("### Network Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_confidence = np.mean(dynamic_preds['confidence'])
        create_metric_card(
            "Average Confidence",
            f"{avg_confidence:.1%}",
            color=COLORS['primary_green']
        )
    
    with col2:
        active_predictions = len(dynamic_preds['congestion'])
        create_metric_card(
            "Active Predictions", 
            f"{active_predictions}",
            color=COLORS['primary_blue']
        )
    
    with col3:
        network_coverage = len(district_counts)
        create_metric_card(
            "District Coverage",
            f"{network_coverage}",
            color=COLORS['primary_purple']
        )