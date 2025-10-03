"""
Tab 3: Route Predictions - Smart route optimization and departure planning
"""

import streamlit as st
from datetime import datetime
from streamlit_folium import st_folium

from config import COLORS, CSS_STYLES
from utils import show_loading_spinner

def render_predictions_tab():
    """Render the route predictions tab with smart route optimization"""
    
    st.markdown(f"""
    <div style="background: {COLORS['primary_blue']}; padding: 2rem; border-radius: 8px; margin-bottom: 2rem; border: 1px solid rgba(0,0,0,0.1);">
        <h2 style="color: white; margin: 0; font-weight: 600;">Smart Route Optimization</h2>
        <p style="color: white; margin: 0.5rem 0 0 0; opacity: 0.9;">Plan your journey with AI-powered traffic predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Departure Planning Section
    st.markdown("## Departure Planning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Departure Date**")
        departure_date = st.date_input(
            "Choose your departure date",
            value=datetime.now().date(),
            help="Select your planned departure date"
        )
        
    with col2:
        st.markdown("🕐 **Departure Time**")
        departure_time = st.time_input(
            "Choose your departure time",
            value=datetime.now().time(),
            help="Select your preferred departure time"
        )
    
    # Combine date and time
    departure_datetime = datetime.combine(departure_date, departure_time)
    
    # Show departure summary
    st.markdown(f"""
    <div style="background: {COLORS['primary_green']}; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        <h4 style="color: white; margin: 0;">📅 Planned Departure</h4>
        <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.1rem;">{departure_datetime.strftime('%A, %B %d, %Y at %H:%M')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Route Selection
    st.markdown("## Route Selection")
    
    # Get all location names from data
    from data_processing import load_model_and_data
    try:
        data = load_model_and_data()
        all_locations = ["-- Select Location --"] + data.get('location_names', [])
    except:
        # Fallback to basic list if data loading fails
        all_locations = ["-- Select Location --", "Siam Square", "MBK Center", "CentralWorld"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Origin**")
        origin = st.selectbox(
            "Starting location",
            options=all_locations,
            index=0,
            key="origin_select"
        )
    
    with col2:
        st.markdown("🅱️ **Destination**")
        destination = st.selectbox(
            "Destination location",
            options=all_locations,
            index=0,
            key="destination_select"
        )
    
    # Only show route analysis if user has made valid selections
    if origin and destination and origin != "-- Select Location --" and destination != "-- Select Location --":
        st.info(f"**Selected Route:** {origin} → {destination}")
        
        # Calculate route metrics based on selected locations
        import math
        
        # Create location coordinates mapping from data
        location_coords = {}
        for i, name in enumerate(data['location_names']):
            if i < len(data['locations']):
                lat, lon = data['locations'][i]
                location_coords[name] = (lat, lon)
        
        # Get coordinates
        origin_coords = location_coords.get(origin, (13.7463, 100.5348))
        dest_coords = location_coords.get(destination, (13.7421, 100.5488))
        
        # Calculate distance using Haversine formula
        lat1, lon1 = origin_coords
        lat2, lon2 = dest_coords
        
        R = 6371  # Earth's radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        # Add realistic factor for road distance (not straight line)
        distance = distance * 1.3  # Roads are ~30% longer than straight line
        
        # Calculate travel time based on departure time and distance
        hour = departure_time.hour
        
        # Base speed varies by time
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            avg_speed = 22  # km/h in rush hour
            traffic_factor = "Heavy"
            color = COLORS['primary_red']
        elif 22 <= hour or hour <= 6:
            avg_speed = 45  # km/h at night
            traffic_factor = "Light"
            color = COLORS['primary_green']
        elif 10 <= hour <= 16:
            avg_speed = 32  # km/h during midday
            traffic_factor = "Moderate"
            color = COLORS['primary_orange']
        else:
            avg_speed = 35  # km/h normal hours
            traffic_factor = "Moderate"
            color = COLORS['primary_orange']
        
        # Calculate time in minutes
        base_time = (distance / avg_speed) * 60
        
        # Route Analysis
        st.markdown("## Route Analysis")
        
        # Import here to avoid circular imports
        from visualization import create_route_map
        
        # Generate route map with actual coordinates
        with show_loading_spinner("Generating optimized route..."):
            try:
                route_map = create_route_map(departure_time, origin_coords, dest_coords, origin, destination)
                if route_map:
                    st_folium(route_map, width=1400, height=500, key="route_map")
                else:
                    st.info("Route map temporarily unavailable")
            except Exception as e:
                st.error(f"Route map error: {e}")
        
        # Route metrics (only show when route is selected)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            
            st.markdown(f"""
            <div style="background: {color}; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid rgba(0,0,0,0.1);">
                <div style="color: #ffffff; font-size: 1.8rem; font-weight: 700;">{int(base_time)} min</div>
                <div style="color: #ffffff; font-size: 1.1rem; font-weight: 600;">Estimated Time</div>
                <div style="color: #ffffff; font-size: 0.9rem; margin-top: 0.2rem; opacity: 0.9;">{traffic_factor} traffic</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            
            st.markdown(f"""
            <div style="background: {COLORS['primary_blue']}; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid rgba(0,0,0,0.1);">
                <div style="color: #ffffff; font-size: 1.8rem; font-weight: 700;">{distance:.1f} km</div>
                <div style="color: #ffffff; font-size: 1.1rem; font-weight: 600;">Total Distance</div>
                <div style="color: #ffffff; font-size: 0.9rem; margin-top: 0.2rem; opacity: 0.9;">Via optimal route</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Calculate fuel cost based on actual distance
            fuel_cost = (distance / 12) * 35  # 12km/L fuel efficiency, 35 THB/L
            
            st.markdown(f"""
            <div style="background: {COLORS['primary_purple']}; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid rgba(0,0,0,0.1);">
                <div style="color: #ffffff; font-size: 1.8rem; font-weight: 700;">฿{fuel_cost:.0f}</div>
                <div style="color: #ffffff; font-size: 1.1rem; font-weight: 600;">Fuel Cost</div>
                <div style="color: #ffffff; font-size: 0.9rem; margin-top: 0.2rem; opacity: 0.9;">Estimated</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Alternative routes suggestion
        st.markdown("## Alternative Routes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Route Option A (Recommended)")
            st.success(f"""
            **Via**: Direct route via main roads
            
            **Time**: {int(base_time)} minutes
            
            **Distance**: {distance:.1f} km
            
            **Traffic**: {traffic_factor}
            
            **Advantages**: Fastest route, uses highways when available
            """)
        
        with col2:
            st.markdown("### Route Option B (Alternative)")
            alt_time = int(base_time * 1.25)  # 25% longer
            alt_distance = distance * 1.15  # 15% longer
            st.warning(f"""
            **Via**: Alternative via side roads
            
            **Time**: {alt_time} minutes
            
            **Distance**: {alt_distance:.1f} km
            
            **Traffic**: Light to Moderate
            
            **Advantages**: Less congestion, more scenic route
            """)
        
        # Traffic predictions for route
        st.markdown("## Traffic Predictions Along Route")
        
        # Show hourly predictions
        hours = list(range(6, 23))
        predictions = []
        
        for hour in hours:
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                traffic_level = "Heavy"
                color = "🔴"
            elif 10 <= hour <= 16:
                traffic_level = "Moderate"
                color = "🟡"
            else:
                traffic_level = "Light"
                color = "🟢"
            
            predictions.append(f"{hour:02d}:00 {color} {traffic_level}")
        
        # Display in columns
        col1, col2, col3 = st.columns(3)
        
        chunk_size = len(predictions) // 3
        
        with col1:
            st.markdown("**Morning Hours**")
            for pred in predictions[:chunk_size]:
                st.text(pred)
        
        with col2:
            st.markdown("**Afternoon Hours**")
            for pred in predictions[chunk_size:chunk_size*2]:
                st.text(pred)
        
        with col3:
            st.markdown("**Evening Hours**")
            for pred in predictions[chunk_size*2:]:
                st.text(pred)
    else:
        # Show message when no route is selected
        st.info("**Please select both Origin and Destination above to view route analysis and traffic predictions.**")