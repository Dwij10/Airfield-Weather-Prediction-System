import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import os

def generate_sample_radar_data(date, directory):
    """Generate sample radar data for a given date"""
    # Create a sample 100x100 reflectivity field
    data = np.random.normal(10, 5, size=(100, 100))
    # Add some realistic storm cells
    for _ in range(3):
        x = np.random.randint(20, 80)
        y = np.random.randint(20, 80)
        data[x-10:x+10, y-10:y+10] += np.random.normal(40, 10)
    
    # Create xarray dataset
    ds = xr.Dataset(
        {
            "Reflectivity": (["y", "x"], data),
        },
        coords={
            "x": np.arange(100),
            "y": np.arange(100),
        },
    )
    
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Save to netCDF file
    filename = f"radar_{date.strftime('%Y%m%d_%H%M')}.nc"
    ds.to_netcdf(os.path.join(directory, filename))

def main():
    base_dir = "data/raw/radar"
    # Generate data for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    current_date = start_date
    
    while current_date <= end_date:
        # Generate data every 30 minutes
        for hour in range(24):
            for minute in [0, 30]:
                date = current_date.replace(hour=hour, minute=minute)
                directory = os.path.join(
                    base_dir,
                    str(date.year),
                    f"{date.month:02d}",
                    f"{date.day:02d}"
                )
                generate_sample_radar_data(date, directory)
        
        current_date += timedelta(days=1)

if __name__ == "__main__":
    main()
