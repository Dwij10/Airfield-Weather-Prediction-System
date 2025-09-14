import boto3
import os
from datetime import datetime, timedelta
import pytz
from botocore import UNSIGNED
from botocore.config import Config

class RadarDownloader:
    def __init__(self, save_dir):
        """
        Initialize the radar downloader
        Args:
            save_dir: Directory to save downloaded radar files
        """
        self.save_dir = save_dir
        self.s3_client = boto3.client(
            's3',
            config=Config(signature_version=UNSIGNED)
        )
        self.bucket_name = 'noaa-nexrad-level2'

    def download_radar_data(self, station_id, start_date, end_date=None):
        """
        Download NEXRAD radar data for a specific station and date range
        Args:
            station_id: Four-letter radar station identifier (e.g., 'KJFK' for JFK airport)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (optional, defaults to start_date)
        """
        if end_date is None:
            end_date = start_date

        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        current_date = start

        while current_date <= end:
            year = current_date.strftime('%Y')
            month = current_date.strftime('%m')
            day = current_date.strftime('%d')
            
            # Create directory structure
            save_path = os.path.join(self.save_dir, year, month, day)
            os.makedirs(save_path, exist_ok=True)

            # List available files for the station on this date
            prefix = f"{year}/{month}/{day}/{station_id}"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    filename = os.path.basename(key)
                    local_file = os.path.join(save_path, filename)
                    
                    # Download file if it doesn't exist
                    if not os.path.exists(local_file):
                        print(f"Downloading {filename}...")
                        self.s3_client.download_file(
                            self.bucket_name,
                            key,
                            local_file
                        )

            current_date += timedelta(days=1)

# Example usage:
if __name__ == "__main__":
    downloader = RadarDownloader("data/raw/radar")
    # Replace KXXX with your nearest NEXRAD radar station ID
    downloader.download_radar_data("KXXX", "2025-09-13", "2025-09-14")
