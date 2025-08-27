import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import GoogleV3
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
import googlemaps
import os
# OR-Tools removed for Python 3.13 compatibility - using simple optimization instead
import tempfile
import json
import time
from datetime import datetime
import base64

# Utility functions
def geocode_addresses(df, api_key):
    """Geocode addresses using Google Maps API"""
    gmaps = googlemaps.Client(key=api_key)
    geocoded_data = []

    for idx, row in df.iterrows():
        try:
            # Geocode the address
            geocode_result = gmaps.geocode(row['Address'])

            if geocode_result:
                location = geocode_result[0]['geometry']['location']
                lat, lng = location['lat'], location['lng']

                geocoded_data.append({
                    'index': idx,
                    'name': row['Name'],
                    'address': row['Address'],
                    'wheelchair': row['Wheelchair'],
                    'latitude': lat,
                    'longitude': lng
                })
            else:
                st.warning(f"Could not geocode address: {row['Address']}")

        except Exception as e:
            st.error(f"Error geocoding {row['Address']}: {str(e)}")
            continue

    return pd.DataFrame(geocoded_data)

def group_by_address(df):
    """Group individuals by address to optimize van assignments"""
    # Group by address
    grouped = df.groupby('address').agg({
        'name': list,
        'wheelchair': list,
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()

    # Calculate group size and wheelchair count
    grouped['group_size'] = grouped['name'].apply(len)
    grouped['wheelchair_count'] = grouped['wheelchair'].apply(sum)
    grouped['total_wheelchair'] = grouped['wheelchair'].apply(sum)
    grouped['total_regular'] = grouped['group_size'] - grouped['total_wheelchair']

    return grouped

def create_distance_matrix(locations, api_key):
    """Create distance matrix using Google Maps Distance Matrix API"""
    gmaps = googlemaps.Client(key=api_key)
    n = len(locations)
    distance_matrix = [[0] * n for _ in range(n)]

    # Use Google Maps Distance Matrix API for accurate distances
    origins = [(loc['latitude'], loc['longitude']) for loc in locations]
    destinations = origins.copy()

    # Split into chunks to avoid API limits
    chunk_size = 25
    for i in range(0, n, chunk_size):
        for j in range(0, n, chunk_size):
            origin_chunk = origins[i:i+chunk_size]
            dest_chunk = destinations[j:j+chunk_size]

            if origin_chunk and dest_chunk:
                result = gmaps.distance_matrix(
                    origin_chunk,
                    dest_chunk,
                    mode="driving",
                    units="metric"
                )

                # Extract distances in meters
                for x in range(len(origin_chunk)):
                    for y in range(len(dest_chunk)):
                        if result['rows'][x]['elements'][y]['status'] == 'OK':
                            distance = result['rows'][x]['elements'][y]['distance']['value']
                            distance_matrix[i+x][j+y] = distance
                        else:
                            # Fallback to straight-line distance if API fails
                            coord1 = origin_chunk[x]
                            coord2 = dest_chunk[y]
                            straight_distance = geodesic(coord1, coord2).meters
                            distance_matrix[i+x][j+y] = int(straight_distance)

    return distance_matrix

def optimize_routes(distance_matrix, num_vehicles, vehicle_capacity, wheelchair_capacity):
    """Optimize routes using simple greedy algorithm (nearest neighbor)"""
    n = len(distance_matrix)
    if n == 0:
        return [], 0

    # Simple nearest neighbor assignment
    unassigned = list(range(n))
    routes = []
    total_distance = 0

    for vehicle_id in range(num_vehicles):
        if not unassigned:
            break

        route = []
        current_distance = 0

        # Start with the first available location
        current = unassigned.pop(0)
        route.append(current)

        # Add nearest neighbors until capacity is reached or no more locations
        while len(route) < vehicle_capacity and unassigned:
            # Find nearest unassigned location
            nearest = min(unassigned, key=lambda x: distance_matrix[current][x])
            current_distance += distance_matrix[current][nearest]
            current = nearest
            route.append(current)
            unassigned.remove(current)

        routes.append({
            'vehicle_id': vehicle_id + 1,
            'stops': route,
            'distance': current_distance
        })
        total_distance += current_distance

    # If there are still unassigned locations, add them to the last route
    if unassigned and routes:
        last_route = routes[-1]
        for location in unassigned:
            if len(last_route['stops']) < vehicle_capacity:
                last_distance = distance_matrix[last_route['stops'][-1]][location]
                last_route['stops'].append(location)
                last_route['distance'] += last_distance
                total_distance += last_distance

    return routes, total_distance

def create_route_map(routes, locations, grouped_data):
    """Create a folium map with the optimized routes"""
    if not routes:
        return None

    # Center map on average location
    avg_lat = sum(loc['latitude'] for loc in locations) / len(locations)
    avg_lng = sum(loc['longitude'] for loc in locations) / len(locations)

    m = folium.Map(location=[avg_lat, avg_lng], zoom_start=12)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige',
              'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue']

    for route in routes:
        color = colors[route['vehicle_id'] % len(colors)]

        # Add route line
        route_coords = []
        for stop_idx in route['stops']:
            if stop_idx < len(locations):
                location = locations[stop_idx]
                route_coords.append([location['latitude'], location['longitude']])

                # Add marker with popup
                if stop_idx < len(grouped_data):
                    stop_data = grouped_data.iloc[stop_idx]
                    names = ', '.join(stop_data['name'])
                    wheelchair_count = stop_data['wheelchair_count']

                    popup_text = f"""
                    <b>Van {route['vehicle_id']}</b><br>
                    Stop {len(route_coords)}<br>
                    Address: {stop_data['address']}<br>
                    Passengers: {names}<br>
                    Wheelchair: {wheelchair_count}
                    """

                    folium.Marker(
                        [location['latitude'], location['longitude']],
                        popup=popup_text,
                        icon=folium.Icon(color=color)
                    ).add_to(m)

        if len(route_coords) > 1:
            folium.PolyLine(route_coords, color=color, weight=3, opacity=0.8).add_to(m)

    return m

def generate_manifest(routes, grouped_data):
    """Generate driver manifests"""
    manifests = []

    for route in routes:
        manifest = {
            'vehicle_id': route['vehicle_id'],
            'stops': [],
            'total_passengers': 0,
            'total_wheelchair': 0
        }

        for stop_idx in route['stops']:
            if stop_idx < len(grouped_data):
                stop_data = grouped_data.iloc[stop_idx]
                manifest['stops'].append({
                    'address': stop_data['address'],
                    'passengers': stop_data['name'],
                    'wheelchair_count': stop_data['wheelchair_count'],
                    'total_count': stop_data['group_size']
                })
                manifest['total_passengers'] += stop_data['group_size']
                manifest['total_wheelchair'] += stop_data['wheelchair_count']

        manifests.append(manifest)

    return manifests

def create_download_link(df, filename):
    """Create a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Page configuration
st.set_page_config(
    page_title="NES Route Optimizer",
    page_icon="🚐",
    layout="wide"
)

# Title and description
st.title("🚐 NES Route Optimizer")
st.markdown("""
Upload a CSV file with your daily pickup list to generate optimized routes for NES vans.

**Required CSV format:**
- Column 1: Name (Individual's name)
- Column 2: Address (Full address including city, state, zip)
- Column 3: Wheelchair (Yes/No)
""")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'routes' not in st.session_state:
    st.session_state.routes = None
if 'geocoded_data' not in st.session_state:
    st.session_state.geocoded_data = None

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Google Maps API Key input
    api_key = st.text_input("Google Maps API Key", type="password")
    
    # Vehicle configuration
    st.subheader("Vehicle Settings")
    num_vehicles = st.number_input("Number of Vans", min_value=1, max_value=10, value=4)
    
    # Vehicle capacities
    st.subheader("Vehicle Capacities")
    vehicle_capacity = st.number_input("Seats per Van", min_value=1, max_value=20, value=10)
    wheelchair_spots = st.number_input("Wheelchair Spots per Van", min_value=0, max_value=5, value=2)
    
    # Optimization settings
    st.subheader("Optimization")
    objective = st.selectbox(
        "Optimization Objective",
        ["Minimize Total Distance", "Minimize Total Time", "Balance Workload"]
    )

# File upload section
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        
        # Validate CSV format
        if len(df.columns) != 3:
            st.error("CSV must have exactly 3 columns: Name, Address, Wheelchair")
        else:
            # Rename columns for consistency
            df.columns = ['Name', 'Address', 'Wheelchair']
            
            # Clean data
            df['Name'] = df['Name'].astype(str).str.strip()
            df['Address'] = df['Address'].astype(str).str.strip()
            df['Wheelchair'] = df['Wheelchair'].astype(str).str.strip().str.lower()
            
            # Convert wheelchair to boolean
            df['Wheelchair'] = df['Wheelchair'].isin(['yes', 'y', 'true', '1'])
            
            # Remove empty rows
            df = df[df['Name'].notna() & df['Address'].notna() & (df['Name'] != '') & (df['Address'] != '')]
            
            st.session_state.data = df
            
            st.success(f"Successfully loaded {len(df)} individuals")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df)
            
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    if st.session_state.data is not None:
        st.subheader("Route Optimization")
        
        if st.button("Generate Optimized Routes", type="primary"):
            if not api_key:
                st.error("Please enter your Google Maps API Key in the sidebar")
            else:
                with st.spinner("Geocoding addresses..."):
                    # Geocode addresses
                    geocoded_df = geocode_addresses(st.session_state.data, api_key)

                    if len(geocoded_df) == 0:
                        st.error("No addresses could be geocoded. Please check your addresses and API key.")
                    else:
                        st.session_state.geocoded_data = geocoded_df

                        # Group by address
                        grouped_data = group_by_address(geocoded_df)

                        with st.spinner("Creating distance matrix..."):
                            # Create distance matrix
                            locations = grouped_data[['latitude', 'longitude']].to_dict('records')
                            distance_matrix = create_distance_matrix(locations, api_key)

                        with st.spinner("Optimizing routes..."):
                            # Optimize routes
                            routes, total_distance = optimize_routes(
                                distance_matrix,
                                num_vehicles,
                                vehicle_capacity,
                                wheelchair_spots
                            )

                            if routes:
                                st.session_state.routes = routes
                                st.session_state.grouped_data = grouped_data
                                st.success("Routes optimized successfully!")

                                # Display results
                                col1, col2 = st.columns([2, 1])

                                with col1:
                                    st.subheader("Route Map")
                                    route_map = create_route_map(routes, locations, grouped_data)
                                    if route_map:
                                        st_folium(route_map, width=800, height=600)

                                with col2:
                                    st.subheader("Route Summary")

                                    # Total statistics
                                    total_passengers = sum(route['passengers'] for route in generate_manifest(routes, grouped_data))
                                    total_wheelchair = sum(route['wheelchair'] for route in generate_manifest(routes, grouped_data))

                                    st.metric("Total Distance", ".1f")
                                    st.metric("Total Passengers", total_passengers)
                                    st.metric("Total Wheelchair", total_wheelchair)

                                    # Route details
                                    for route in routes:
                                        with st.expander(f"Van {route['vehicle_id']} - {len(route['stops'])} stops"):
                                            manifest = generate_manifest([route], grouped_data)[0]

                                            st.write(f"**Passengers:** {manifest['total_passengers']}")
                                            st.write(f"**Wheelchair:** {manifest['total_wheelchair']}")
                                            st.write(".1f")

                                            for i, stop in enumerate(manifest['stops'], 1):
                                                st.write(f"{i}. {stop['address']}")
                                                st.write(f"   Passengers: {', '.join(stop['passengers'])}")
                                                if stop['wheelchair_count'] > 0:
                                                    st.write(f"   🚗 Wheelchair: {stop['wheelchair_count']}")

                                # Export options
                                st.subheader("Export Options")

                                # Generate manifests
                                manifests = generate_manifest(routes, grouped_data)

                                # Create summary CSV
                                summary_data = []
                                for route in routes:
                                    manifest = generate_manifest([route], grouped_data)[0]
                                    summary_data.append({
                                        'Van': f"Van {route['vehicle_id']}",
                                        'Stops': len(route['stops']),
                                        'Passengers': manifest['total_passengers'],
                                        'Wheelchair': manifest['total_wheelchair'],
                                        'Distance (km)': route['distance'] / 1000
                                    })

                                summary_df = pd.DataFrame(summary_data)

                                # Download buttons
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown(create_download_link(summary_df, "route_summary.csv"), unsafe_allow_html=True)

                                with col2:
                                    # Create detailed manifest
                                    detailed_manifests = []
                                    for route in routes:
                                        manifest = generate_manifest([route], grouped_data)[0]
                                        for i, stop in enumerate(manifest['stops'], 1):
                                            detailed_manifests.append({
                                                'Van': f"Van {route['vehicle_id']}",
                                                'Stop_Number': i,
                                                'Address': stop['address'],
                                                'Passengers': ', '.join(stop['passengers']),
                                                'Wheelchair_Count': stop['wheelchair_count'],
                                                'Total_Count': stop['total_count']
                                            })

                                    detailed_df = pd.DataFrame(detailed_manifests)
                                    st.markdown(create_download_link(detailed_df, "detailed_manifest.csv"), unsafe_allow_html=True)

                            else:
                                st.error("Could not find optimal routes. Try adjusting vehicle capacities or number of vehicles.")

with col2:
    if st.session_state.data is not None:
        st.subheader("Summary Statistics")
        st.metric("Total Individuals", len(st.session_state.data))
        st.metric("Wheelchair Users", st.session_state.data['Wheelchair'].sum())
        st.metric("Unique Addresses", st.session_state.data['Address'].nunique())

# Footer
st.markdown("---")
st.markdown("Built for NES - Van Route Optimization System")
