"""
Utility functions for the GNN Traffic Dashboard
Contains shared helper functions used across multiple modules
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import COLORS, TRAFFIC_LEVELS

def get_time_status(hour):
    """Get time status based on hour"""
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 22:
        return "Evening"
    else:
        return "Night"

def is_rush_hour(hour):
    """Check if given hour is rush hour"""
    return (7 <= hour <= 9) or (17 <= hour <= 19)

def is_weekend(date=None):
    """Check if given date is weekend"""
    if date is None:
        date = datetime.now()
    return date.weekday() >= 5

def is_night_hours(hour):
    """Check if given hour is night time"""
    return hour >= 22 or hour <= 6

def create_sidebar(data):
    """Create the forecast controls sidebar"""
    with st.sidebar:
        st.markdown("## 🎛️ Forecast Controls")
        
        # Forecast time selector
        st.markdown("### ⏰ Forecast Time")
        
        # Time selection
        current_time = datetime.now()
        forecast_hours = st.slider(
            "Select forecast time",
            min_value=0,
            max_value=23,
            value=current_time.hour,
            help="Select the hour for traffic prediction"
        )
        
        forecast_time = current_time.replace(hour=forecast_hours, minute=0, second=0, microsecond=0)
        
        # Display current status
        time_status = get_time_status(forecast_hours)
        is_rush = is_rush_hour(forecast_hours)
        
        status_color = COLORS['primary_red'] if is_rush else COLORS['primary_green']
        status_icon = "🌙" if is_night_hours(forecast_hours) else "☀️"
        
        st.markdown(f"""
        <div style="background: {status_color}; padding: 1rem; border-radius: 8px; margin: 1rem 0; border: 1px solid rgba(0,0,0,0.1);">
            <div style="color: white; font-size: 1.1rem; font-weight: 600;">
                {status_icon} Current Status: {time_status}
            </div>
            <div style="color: white; font-size: 0.9rem; margin-top: 0.5rem;">
                Target time: {forecast_time.strftime('%H:%M')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    return {
        'forecast_time': forecast_time,
        'is_rush_hour': is_rush,
        'is_weekend': is_weekend(),
        'is_night': is_night_hours(forecast_time.hour),
        'show_confidence': True,
        'highlight_rush': True,
        'map_style': 'Default'
    }

def format_confidence(confidence):
    """Format confidence score as percentage"""
    return f"{confidence * 100:.1f}%"

def format_time(hour):
    """Format hour as time string"""
    return f"{hour:02d}:00"

def get_congestion_color(level):
    """Get color for congestion level (0-3)"""
    colors = [COLORS['free_flow'], COLORS['moderate'], COLORS['congested'], COLORS['gridlock']]
    return colors[min(max(int(level), 0), 3)]

def get_congestion_label(level):
    """Get label for congestion level (0-3)"""
    return TRAFFIC_LEVELS['labels'][min(max(int(level), 0), 3)]

def create_metric_card(title, value, delta=None, help_text=None, color=None):
    """Create a professional metric card"""
    if color is None:
        color = COLORS['primary_blue']
    
    delta_html = ""
    if delta is not None:
        # Convert delta to float if it's a string to enable comparison
        try:
            delta_val = float(str(delta).rstrip('%')) if isinstance(delta, str) else float(delta)
        except (ValueError, TypeError):
            delta_val = 0
        
        delta_color = "green" if delta_val > 0 else "red" if delta_val < 0 else "gray"
        delta_html = f'<div style="color: {delta_color}; font-size: 0.8rem; margin-top: 0.2rem;">{"+" if delta_val > 0 else ""}{delta}</div>'
    
    help_html = ""
    if help_text:
        help_html = f'<div style="color: rgba(255,255,255,0.8); font-size: 0.75rem; margin-top: 0.3rem;">{help_text}</div>'
    
    st.markdown(f"""
    <div style="background: {color}; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid rgba(0,0,0,0.1);">
        <div style="color: #ffffff; font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">{title}</div>
        <div style="color: #ffffff; font-size: 1.8rem; font-weight: 700;">{value}</div>
        {delta_html}
        {help_html}
    </div>
    """, unsafe_allow_html=True)

def create_status_card(title, status, color, icon=None):
    """Create a status card"""
    icon_html = f"<span style='margin-right: 0.5rem;'>{icon}</span>" if icon else ""
    
    st.markdown(f"""
    <div style="background: {color}; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid rgba(0,0,0,0.1);">
        <div style="color: #ffffff; font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">{icon_html}{title}</div>
        <div style="color: #ffffff; font-size: 0.9rem; opacity: 0.9;">{status}</div>
    </div>
    """, unsafe_allow_html=True)

def calculate_model_accuracy(predictions, actuals):
    """Calculate model accuracy metrics"""
    if len(predictions) == 0 or len(actuals) == 0:
        return {'accuracy': 0.0, 'mae': 0.0, 'rmse': 0.0}
    
    # Ensure arrays are same length
    min_len = min(len(predictions), len(actuals))
    pred_array = np.array(predictions[:min_len])
    actual_array = np.array(actuals[:min_len])
    
    # Calculate metrics
    accuracy = np.mean(pred_array == actual_array) if len(pred_array) > 0 else 0.0
    mae = np.mean(np.abs(pred_array - actual_array)) if len(pred_array) > 0 else 0.0
    rmse = np.sqrt(np.mean((pred_array - actual_array) ** 2)) if len(pred_array) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'mae': mae,
        'rmse': rmse
    }

def safe_divide(numerator, denominator, default=0.0):
    """Safely divide two numbers, returning default if denominator is 0"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (ZeroDivisionError, TypeError):
        return default

def generate_time_series(hours=24, base_traffic=50):
    """Generate synthetic time series data for testing"""
    times = []
    traffic_levels = []
    
    for hour in range(hours):
        times.append(hour)
        
        # Rush hour pattern
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            traffic = base_traffic + np.random.normal(30, 10)
        elif 22 <= hour or hour <= 6:
            traffic = base_traffic + np.random.normal(-20, 5)
        else:
            traffic = base_traffic + np.random.normal(0, 8)
        
        traffic_levels.append(max(0, min(100, traffic)))
    
    return times, traffic_levels

def validate_data(data, required_fields):
    """Validate that data contains required fields"""
    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)
    
    if missing_fields:
        st.error(f"Missing required data fields: {missing_fields}")
        return False
    return True

def cache_key_generator(*args, **kwargs):
    """Generate a cache key from arguments"""
    import hashlib
    key_string = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_string.encode()).hexdigest()[:8]

def format_number(number, decimals=1):
    """Format number with appropriate decimals and units"""
    if number >= 1000000:
        return f"{number/1000000:.{decimals}f}M"
    elif number >= 1000:
        return f"{number/1000:.{decimals}f}K"
    else:
        return f"{number:.{decimals}f}"

def create_download_button(data, filename, label="Download Data"):
    """Create a download button for data"""
    if isinstance(data, pd.DataFrame):
        csv = data.to_csv(index=False)
        st.download_button(
            label=label,
            data=csv,
            file_name=filename,
            mime='text/csv'
        )
    elif isinstance(data, dict):
        import json
        json_str = json.dumps(data, indent=2)
        st.download_button(
            label=label,
            data=json_str,
            file_name=filename,
            mime='application/json'
        )

def show_loading_spinner(message="Loading..."):
    """Show a loading spinner with message"""
    return st.spinner(message)

def create_info_box(message, type="info"):
    """Create an info box with different types"""
    if type == "success":
        st.success(message)
    elif type == "warning":
        st.warning(message)
    elif type == "error":
        st.error(message)
    else:
        st.info(message)
