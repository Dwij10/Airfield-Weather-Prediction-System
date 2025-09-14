from src.data_downloaders.radar_downloader import RadarDownloader
from src.data_downloaders.satellite_downloader import SatelliteDownloader
from src.data_downloaders.metar_downloader import MetarDownloader
import os
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configuration
RADAR_STATION = "VABJ"  # Replace with your nearest NEXRAD radar station
AIRPORT_CODE = "VAAH"   # Replace with your airport's ICAO code

# Create data directories
base_dir = "data/raw"
for subdir in ['radar', 'satellite', 'metar']:
    os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

# Initialize downloaders
radar_dl = RadarDownloader(os.path.join(base_dir, "radar"))
satellite_dl = SatelliteDownloader(os.path.join(base_dir, "satellite"))
metar_dl = MetarDownloader(os.path.join(base_dir, "metar"))

def download_historical_data(start_date, end_date):
    """
    Download historical data from all sources
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    """
    print(f"Downloading data from {start_date} to {end_date}")
    
    # Download radar data
    print("\nDownloading radar data...")
    radar_dl.download_radar_data(RADAR_STATION, start_date, end_date)
    
    # Download satellite data
    print("\nDownloading satellite data...")
    satellite_dl.download_goes_data(start_date, end_date)
    
    # Download METAR data
    print("\nDownloading METAR data...")
    metar_dl.download_metar_data(AIRPORT_CODE, start_date, end_date)

def download_latest_data():
    """
    Download the most recent data (last 24 hours)
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    download_historical_data(start_date, end_date)

if __name__ == "__main__":
    # Download the last 30 days of historical data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    download_historical_data(start_date, end_date)
    
    # You can also schedule this script to run periodically to keep data updated
