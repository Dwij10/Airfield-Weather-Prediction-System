import requests
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.metar_utils import validate_metar_data, clean_metar_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetarDownloader:
    def __init__(self, save_dir):
        """
        Initialize the METAR data downloader
        Args:
            save_dir: Directory to save downloaded METAR files
        """
        self.save_dir = save_dir
        self.iowa_base_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
        
    def download_metar_data(self, station_id, start_date, end_date=None):
        """
        Download METAR data for a specific airport
        Args:
            station_id: Four-letter ICAO airport code (e.g., 'KJFK' for JFK airport)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (optional, defaults to start_date)
        Returns:
            pandas.DataFrame or None: Processed METAR data if successful, None if failed
        """
        logger.info(f"Downloading METAR data for station {station_id}")
        
        if end_date is None:
            end_date = start_date

        # Parameters for the request
        params = {
            'station': station_id,
            'data': 'all',
            'year1': start_date.split('-')[0],
            'month1': start_date.split('-')[1],
            'day1': start_date.split('-')[2],
            'year2': end_date.split('-')[0],
            'month2': end_date.split('-')[1],
            'day2': end_date.split('-')[2],
            'tz': 'Etc/UTC',
            'format': 'comma',
            'latlon': 'yes',
            'direct': 'yes',
            'report_type': '1,2'
        }

        try:
            # Make the request
            response = requests.get(self.iowa_base_url, params=params)
            
            if response.status_code != 200:
                logger.error(f"Failed to download data: HTTP {response.status_code}")
                return None

            # Create directory structure
            year = start_date.split('-')[0]
            month = start_date.split('-')[1]
            save_path = os.path.join(self.save_dir, year, month)
            os.makedirs(save_path, exist_ok=True)
            
            # Check if we got valid data
            if "No data found for" in response.text:
                logger.warning(f"No data available for station {station_id} in the specified date range")
                return None
                
            # Save raw data
            filename = f"{station_id}_{start_date}_{end_date}.csv"
            file_path = os.path.join(save_path, filename)
            
            # Save raw response
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            logger.info(f"Raw METAR data saved to {file_path}")
            
            try:
                # Read the raw CSV data
                df = pd.read_csv(
                    file_path,
                    skiprows=lambda x: x < 5,  # Skip header comments
                    na_values=['M', 'null', ''],
                    parse_dates=['valid'],
                    delimiter=',',
                    encoding='utf-8'
                )
                
                # Validate the data
                is_valid, error_msg = validate_metar_data(df)
                if not is_valid:
                    logger.error(f"Invalid METAR data: {error_msg}")
                    return None
                
                # Clean and process the data
                df = clean_metar_data(df)
                
                # Save processed data
                processed_file = os.path.join(save_path, f"processed_{filename}")
                df.to_csv(processed_file, index=False)
                
                logger.info(f"Successfully processed METAR data for {station_id}")
                return df
                
            except Exception as e:
                logger.error(f"Error processing METAR data: {str(e)}")
                # Save the problematic data for inspection
                error_file = os.path.join(save_path, f"error_{filename}")
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                logger.error(f"Saved problematic data to {error_file}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading METAR data: {str(e)}")
            return None
