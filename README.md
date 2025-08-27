# NES Route Optimizer

A Streamlit application for optimizing van routes for NES transportation services.

## Features

- **CSV Upload**: Upload CSV files with individual pickup information
- **Address Geocoding**: Automatically convert addresses to coordinates using Google Maps API
- **Automatic Grouping**: Groups individuals by address for efficient van assignments
- **Route Optimization**: Uses OR-Tools to optimize routes for multiple vans
- **Map Visualization**: Interactive map showing optimized routes
- **Export Functionality**: Download route manifests and summaries as CSV files

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Get a Google Maps API key:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Enable the Maps JavaScript API and Distance Matrix API
   - Create an API key

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## CSV Format

Your CSV file must have exactly 3 columns:
- **Name**: Individual's name
- **Address**: Full address (street, city, state, zip)
- **Wheelchair**: "Yes" or "No"

Example:
```csv
Name,Address,Wheelchair
John Smith,123 Main St, No
Jane Doe,456 Oak Ave, Yes
Bob Johnson,789 Pine St, No
```

## Usage

1. Enter your Google Maps API key in the sidebar
2. Configure vehicle settings (number of vans, capacity, wheelchair spots)
3. Upload your CSV file
4. Click "Generate Optimized Routes"
5. View the optimized routes on the map
6. Download manifests and summaries

## Configuration

- **Number of Vans**: 1-10 vans
- **Seats per Van**: Maximum passenger capacity
- **Wheelchair Spots per Van**: Dedicated wheelchair capacity
- **Optimization Objective**: Choose between distance, time, or workload balancing

## Sample Data

A sample CSV file (`sample_data.csv`) is included for testing purposes.
