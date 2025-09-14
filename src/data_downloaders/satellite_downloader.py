import requests
import os
from datetime import datetime, timedelta

class SatelliteDownloader:
    def __init__(self, save_dir):
        """
        Initialize the satellite data downloader
        Args:
            save_dir: Directory to save downloaded satellite files
        """
        self.save_dir = save_dir
        self.goes_base_url = "https://noaa-goes16.s3.amazonaws.com"
        
    def download_goes_data(self, start_date, end_date=None, product="ABI-L2-MCMIPC"):
        """
        Download GOES-16 satellite data
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (optional, defaults to start_date)
            product: GOES product name (default: ABI-L2-MCMIPC for multi-band cloud and moisture imagery)
        """
        if end_date is None:
            end_date = start_date

        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        current_date = start

        while current_date <= end:
            year = current_date.strftime('%Y')
            day_of_year = current_date.strftime('%j')
            
            # Create directory structure
            save_path = os.path.join(self.save_dir, year, current_date.strftime('%m-%d'))
            os.makedirs(save_path, exist_ok=True)

            # GOES-16 data is organized by hour
            for hour in range(24):
                url = f"{self.goes_base_url}/{product}/{year}/{day_of_year}/{hour:02d}"
                
                try:
                    # List files in the bucket for this hour
                    response = requests.get(f"{url}/index.html")
                    if response.status_code == 200:
                        # Parse the HTML to find .nc files
                        files = [line for line in response.text.split('\n') if '.nc' in line]
                        
                        for file in files:
                            filename = file.split('"')[1]  # Extract filename from HTML
                            file_url = f"{url}/{filename}"
                            local_file = os.path.join(save_path, filename)
                            
                            if not os.path.exists(local_file):
                                print(f"Downloading {filename}...")
                                response = requests.get(file_url, stream=True)
                                if response.status_code == 200:
                                    with open(local_file, 'wb') as f:
                                        for chunk in response.iter_content(chunk_size=8192):
                                            f.write(chunk)
                
                except Exception as e:
                    print(f"Error downloading data for {current_date} hour {hour}: {str(e)}")
                    continue

            current_date += timedelta(days=1)

# Example usage:
if __name__ == "__main__":
    downloader = SatelliteDownloader("data/raw/satellite")
    downloader.download_goes_data("2025-09-13", "2025-09-14")
