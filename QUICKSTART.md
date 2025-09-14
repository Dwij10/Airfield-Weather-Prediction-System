# Quick Start Guide

This guide will help you get the Airfield Weather Prediction System up and running.

## 1. Installation

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install requirements
pip install -r requirements.txt
pip install -e .
```

## 2. Configuration

1. Find your nearest weather stations:
```bash
python src/utils/location_finder.py
```
Enter your latitude and longitude when prompted.

2. Update configuration in `download_data.py`:
- Set `RADAR_STATION` to your nearest NEXRAD station ID
- Set `AIRPORT_CODE` to your nearest airport's ICAO code

## 3. Data Collection

Start the automated data collection:
```bash
.\start_data_collection.bat  # Windows
./start_data_collection.sh   # Linux/Mac
```

This will:
- Download radar data from NEXRAD
- Download satellite data from GOES-16
- Collect METAR data from your local airport
- Update every hour automatically

## 4. Running the System

1. Start the dashboard:
```bash
python main.py
```

2. Access the dashboard at: http://localhost:8050

## 5. System Components

- **Data Collection**: Automated collection from multiple sources
- **Weather Prediction**: 
  - Short-term (0-3 hours)
  - Medium-term (up to 24 hours)
- **Alert System**: Automated alerts for:
  - Strong winds
  - Thunderstorms
  - Poor visibility
- **Dashboard**: Real-time visualization and monitoring

## 6. Testing

Run the system tests:
```bash
python -m pytest tests/
```

## 7. Logs

- Data collection logs: `data_collection.log`
- Alert system logs: Located in `logs/alerts.log`

## 8. Maintenance

- Data is automatically archived after 30 days
- Models are retrained weekly with new data
- Alert thresholds can be adjusted in `src/alert_system.py`

## Support

For issues or questions:
1. Check the logs in the `logs` directory
2. Run the system tests
3. Verify your weather station IDs
4. Check your internet connection
