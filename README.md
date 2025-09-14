# Airfield Weather Prediction System

This project implements an AI/ML-based system for predicting thunderstorms and gale force winds over airfields. The system provides real-time weather monitoring, prediction, and alerts to ensure airfield safety and operational efficiency.

## Features

1. **Data Ingestion & Processing**
   - Multiple data source integration (Radar, Satellite, Weather Stations)
   - Real-time data processing and cleaning
   - Time series alignment and synchronization

2. **Weather Prediction Models**
   - Short-term forecasting (0-3 hours) using LSTM and CNN models
   - Medium-term forecasting (up to 24 hours) using ensemble methods
   - Continuous model retraining capabilities

3. **Alert System**
   - Real-time condition monitoring
   - Configurable alert thresholds
   - Detailed explanations for alerts
   - Alert logging and audit trail

4. **Interactive Dashboard**
   - Real-time weather maps
   - Risk level indicators
   - Forecast visualizations
   - Alert display and management

## Installation

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
```bash
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

3. Install required packages:
```bash
pip install numpy pandas scikit-learn tensorflow dash plotly netCDF4 xarray metpy
```

## Project Structure

```
airfield-weather-predictor/
├── data/               # Data storage directory
├── models/            # Trained model storage
├── src/              # Source code
│   ├── data_ingestion.py   # Data processing
│   ├── weather_predictor.py # ML models
│   ├── alert_system.py     # Alert management
│   └── dashboard.py        # Web interface
└── main.py           # Application entry point
```

## Usage

1. Start the application:
```bash
python main.py
```

2. Access the dashboard at `http://localhost:8050`

## Configuration

- Alert thresholds can be configured in `src/alert_system.py`
- Model parameters can be adjusted in `src/weather_predictor.py`
- Dashboard settings can be modified in `src/dashboard.py`

## Data Sources

The system expects the following data sources:
- Radar (DWR) data for storm cells and wind velocity
- Satellite imagery for cloud movements
- AWS (Automated Weather Station) data for ground conditions
- Historical weather records for model training

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
