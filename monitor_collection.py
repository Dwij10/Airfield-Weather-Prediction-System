import os
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitor.log'),
        logging.StreamHandler()
    ]
)

def check_data_freshness():
    """
    Check if the collected data is up to date
    """
    now = datetime.now()
    data_root = Path('data/raw')
    
    # Check METAR data
    metar_path = data_root / 'metar' / now.strftime('%Y') / now.strftime('%m')
    if metar_path.exists():
        metar_files = list(metar_path.glob('*.csv'))
        if metar_files:
            latest_file = max(metar_files, key=os.path.getmtime)
            last_modified = datetime.fromtimestamp(os.path.getmtime(latest_file))
            if now - last_modified > timedelta(hours=1):
                logging.warning(f"METAR data is outdated. Last update: {last_modified}")
            else:
                logging.info(f"METAR data is current. Last update: {last_modified}")
        else:
            logging.error("No METAR files found")
    else:
        logging.error("METAR directory not found")
    
    # Check Radar data
    radar_path = data_root / 'radar'
    if radar_path.exists():
        radar_files = list(radar_path.rglob('*.nc'))
        if radar_files:
            latest_file = max(radar_files, key=os.path.getmtime)
            last_modified = datetime.fromtimestamp(os.path.getmtime(latest_file))
            if now - last_modified > timedelta(hours=1):
                logging.warning(f"Radar data is outdated. Last update: {last_modified}")
            else:
                logging.info(f"Radar data is current. Last update: {last_modified}")
        else:
            logging.error("No radar files found")
    else:
        logging.error("Radar directory not found")
    
    # Check Satellite data
    satellite_path = data_root / 'satellite'
    if satellite_path.exists():
        satellite_files = list(satellite_path.rglob('*.nc'))
        if satellite_files:
            latest_file = max(satellite_files, key=os.path.getmtime)
            last_modified = datetime.fromtimestamp(os.path.getmtime(latest_file))
            if now - last_modified > timedelta(hours=1):
                logging.warning(f"Satellite data is outdated. Last update: {last_modified}")
            else:
                logging.info(f"Satellite data is current. Last update: {last_modified}")
        else:
            logging.error("No satellite files found")
    else:
        logging.error("Satellite directory not found")

def check_disk_space():
    """
    Check available disk space
    """
    data_path = Path('data')
    total_size = 0
    for path in data_path.rglob('*'):
        if path.is_file():
            total_size += path.stat().st_size
    
    # Convert to GB
    total_size_gb = total_size / (1024 * 1024 * 1024)
    
    if total_size_gb > 50:  # Alert if more than 50GB
        logging.warning(f"Data directory size is large: {total_size_gb:.2f} GB")
    else:
        logging.info(f"Current data directory size: {total_size_gb:.2f} GB")

def main():
    """
    Main monitoring function
    """
    try:
        logging.info("Starting data collection monitoring")
        check_data_freshness()
        check_disk_space()
        logging.info("Monitoring check completed")
    except Exception as e:
        logging.error(f"Error during monitoring: {str(e)}")

if __name__ == "__main__":
    main()
