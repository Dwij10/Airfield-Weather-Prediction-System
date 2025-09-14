import requests
import os
from datetime import datetime, timedelta
import logging
import xarray as xr
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class RadarDataFetcher:
    """Fetches radar data from IMD's DWR (Doppler Weather Radar) network"""
    
    def __init__(self, radar_id="VAAH", data_dir="data/raw/radar"):
        self.radar_id = radar_id  # VAAH for Ahmedabad radar
        self.data_dir = Path(data_dir)
        self.base_url = "https://mausam.imd.gov.in/radar/dwr"  # Base URL for IMD radar data
        
    def fetch_latest_data(self):
        """
        Fetches the latest radar data from IMD's DWR network
        Returns:
            Path to downloaded file or None if unsuccessful
        """
        try:
            current_time = datetime.now()
            # Round down to nearest 10 minutes as that's typically the radar scan interval
            current_time = current_time.replace(minute=current_time.minute // 10 * 10, second=0, microsecond=0)
            
            # Create directory structure
            save_dir = self.data_dir / str(current_time.year) / f"{current_time.month:02d}" / f"{current_time.day:02d}"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Construct URL for the latest radar data
            timestamp = current_time.strftime("%Y%m%d_%H%M")
            url = f"{self.base_url}/{self.radar_id}/MAX_{timestamp}.nc"
            
            # Attempt to download the file
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                save_path = save_dir / f"radar_{timestamp}.nc"
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Successfully downloaded radar data to {save_path}")
                return save_path
            else:
                # If latest not available, try the previous 10-minute interval
                previous_time = current_time - timedelta(minutes=10)
                timestamp = previous_time.strftime("%Y%m%d_%H%M")
                url = f"{self.base_url}/{self.radar_id}/MAX_{timestamp}.nc"
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    save_path = save_dir / f"radar_{timestamp}.nc"
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Successfully downloaded radar data to {save_path}")
                    return save_path
                else:
                    logger.error(f"Failed to fetch radar data. Status code: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching radar data: {str(e)}")
            return None
            
    def get_latest_data(self):
        """
        Gets the latest radar data, either by downloading new data or using cached data
        Returns:
            xarray.Dataset: The radar data
            None: If no data is available
        """
        try:
            # Try to fetch new data
            file_path = self.fetch_latest_data()
            if file_path is not None and file_path.exists():
                return xr.open_dataset(file_path)
            
            # If fetch fails, generate sample data
            current_time = datetime.now()
            base_path = self.data_dir / str(current_time.year) / f"{current_time.month:02d}" / f"{current_time.day:02d}"
            base_path.mkdir(parents=True, exist_ok=True)
            
            # Generate sample radar data
            logger.info("Generating sample radar data for testing")
            sample_file = base_path / f"radar_{current_time.strftime('%Y%m%d_%H%M')}.nc"
            
            # Create sample data
            nx, ny = 100, 100
            reflectivity = np.random.normal(10, 5, size=(nx, ny))  # Base reflectivity
            velocity = np.random.normal(0, 10, size=(nx, ny))      # Radial velocity
            
            # Add some simulated storm cells
            for _ in range(3):
                x = np.random.randint(20, 80)
                y = np.random.randint(20, 80)
                reflectivity[x-10:x+10, y-10:y+10] += np.random.normal(40, 10)
                velocity[x-10:x+10, y-10:y+10] += np.random.normal(20, 5)
            
            # Create xarray dataset
            ds = xr.Dataset(
                {
                    "reflectivity": (["y", "x"], reflectivity),
                    "velocity": (["y", "x"], velocity),
                },
                coords={
                    "x": np.arange(nx),
                    "y": np.arange(ny),
                    "time": current_time
                },
            )
            
            # Save sample data
            ds.to_netcdf(sample_file)
            logger.info(f"Generated sample radar data: {sample_file}")
            return ds
            
        except Exception as e:
            logger.error(f"Error getting latest radar data: {str(e)}")
            return None
