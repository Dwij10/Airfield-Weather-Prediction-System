import pandas as pd
import numpy as np
import xarray as xr
import metpy.calc
from metpy.units import units
from pathlib import Path
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
from datetime import datetime
from .data_fetchers.radar_fetcher import RadarDataFetcher
import logging

logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self):
        self.radar_data = None
        self.satellite_data = None
        self.aws_data = None
        self.metar_data = None
        self.data_root = Path('data/raw')
        self.radar_fetcher = RadarDataFetcher(radar_id="VAAH")
        
    def __del__(self):
        if hasattr(self, 'satellite_data') and self.satellite_data is not None:
            try:
                self.satellite_data.close()
            except:
                pass
        
    def ingest_metar_data(self):
        try:
            now = datetime.now()
            
            metar_dir = self.data_root / 'metar' / str(now.year) / f"{now.month:02d}"
            if not metar_dir.exists():
                logger.warning(f"METAR directory does not exist: {metar_dir}")
                return False
                
            metar_files = list(metar_dir.glob("VAAH_*.csv"))
            if not metar_files:
                logger.warning("No METAR files found")
                return False
                
            latest_file = max(metar_files, key=lambda f: f.stat().st_mtime)
            
            self.metar_data = pd.read_csv(latest_file, comment='#')
            logger.info(f"METAR columns available: {list(self.metar_data.columns)}")
            
            self.metar_data['timestamp'] = pd.to_datetime(self.metar_data['valid'])
            
            column_mapping = {
                'sknt': 'wind_speed',      
                'drct': 'wind_direction',  
                'tmpf': 'temperature',     
                'dwpf': 'dewpoint',       
                'mslp': 'pressure',       
                'relh': 'humidity',       
                'vsby': 'visibility'      
            }
            
            existing_columns = {k: v for k, v in column_mapping.items() if k in self.metar_data.columns}
            self.metar_data = self.metar_data.rename(columns=existing_columns)
            
            # Handling missing data and replacing missing values with M
            numeric_columns = ['sknt', 'drct', 'tmpf', 'dwpf', 'mslp', 'relh', 'vsby']
            for col in numeric_columns:
                if col in self.metar_data.columns:
                    self.metar_data[col] = pd.to_numeric(self.metar_data[col], errors='coerce')
            
            # Filling out missing values with forward fill and backward fill
            self.metar_data = self.metar_data.fillna(method='ffill').fillna(method='bfill')
            
            # Convert into standard format
            if 'wind_speed' in self.metar_data.columns:
                self.metar_data['wind_speed'] = self.metar_data['wind_speed'] * 0.514444  # Convert knots to m/s
            
            if 'temperature' in self.metar_data.columns:
                self.metar_data['temperature'] = (self.metar_data['temperature'] - 32) * 5/9  # Convert °F to °C
            
            if 'dewpoint' in self.metar_data.columns:
                self.metar_data['dewpoint'] = (self.metar_data['dewpoint'] - 32) * 5/9  # Convert °F to °C
                
            if 'mslp' in self.metar_data.columns:
                self.metar_data['mslp'] = self.metar_data['mslp'].astype(float)  # Ensure numeric
            
            # Calculate relative humidity
            if 'humidity' not in self.metar_data.columns and all(col in self.metar_data.columns for col in ['temperature', 'dewpoint']):
                self.metar_data['humidity'] = metpy.calc.relative_humidity_from_dewpoint(
                    self.metar_data['temperature'] * units.degC,
                    self.metar_data['dewpoint'] * units.degC
                )
            
            logger.info(f"Successfully ingested METAR data from {latest_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting METAR data: {str(e)}")
            return False

    def ingest_radar_data(self):
        
        try:
            dataset = self.radar_fetcher.get_latest_data()
            if dataset is None:
                logger.error("Failed to fetch radar data")
                return False
                
            self.radar_data = dataset
            
            reflectivity = self.radar_data['reflectivity']
            velocity = self.radar_data['velocity']
            
            # Remove values below noise threshold
            reflectivity = reflectivity.where(reflectivity > -20)
            velocity = velocity.where(abs(velocity) < 100)  # Filter unrealistic velocities
            
            self.radar_data['storm_intensity'] = reflectivity.where(reflectivity > 35)  # Strong storm threshold
            
            logger.info("Successfully ingested radar data")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting radar data: {str(e)}")
            return False
        
    def ingest_satellite_data(self):
        
        try:
            now = datetime.now()
            
            satellite_dir = self.data_root / 'satellite' / str(now.year) / f"{now.month:02d}" / f"{now.day:02d}"
            satellite_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensuring satellite directory exists: {satellite_dir}")
                
            satellite_dir.mkdir(parents=True, exist_ok=True)
            
            current_time = datetime.now()
            sample_file = satellite_dir / f"satellite_{current_time.strftime('%Y%m%d_%H%M')}.nc"
            
            logger.info("Generating sample satellite data")
            # sample data
            nx, ny = 256, 256
            ir_data = np.random.normal(273, 20, size=(nx, ny))  # IR temperatures around 0°C
            wv_data = np.random.normal(260, 15, size=(nx, ny))  # Water vapor
            vis_data = np.random.normal(0.5, 0.2, size=(nx, ny)).clip(0, 1)  # Visible albedo
            
            # Add cloud features
            for _ in range(5):
                x = np.random.randint(50, 200)
                y = np.random.randint(50, 200)
                size = np.random.randint(20, 50)
                ir_data[x-size//2:x+size//2, y-size//2:y+size//2] -= np.random.normal(40, 10)
                vis_data[x-size//2:x+size//2, y-size//2:y+size//2] += np.random.normal(0.3, 0.1)
            
            # dataset
            ds = xr.Dataset(
                {
                    "IR": (["y", "x"], ir_data),
                    "WV": (["y", "x"], wv_data),
                    "VIS": (["y", "x"], vis_data),
                    "brightness_temperature": (["y", "x"], ir_data)  # IR data is brightness temperature
                },
                coords={
                    "x": np.arange(nx),
                    "y": np.arange(ny),
                    "time": current_time
                }
            )
            
            # Close existing dataset
            if self.satellite_data is not None:
                try:
                    self.satellite_data.close()
                except:
                    pass
                    
            # Generate new file with unique timestamp
            timestamp = current_time.strftime('%Y%m%d_%H%M%S_%f')[:19]  # Include milliseconds
            sample_file = satellite_dir / f"satellite_{timestamp}.nc"
            
            ds.to_netcdf(sample_file)
            logger.info(f"Generated sample satellite data: {sample_file}")
            
            try:
                # satellite data closing handling
                self.satellite_data = xr.open_dataset(sample_file, cache=False, engine='netcdf4')
                
                # Calculate derived fields
                self.satellite_data['cloud_top_temp'] = self.satellite_data['IR'] - 273.15  # Convert to Celsius
            except ImportError as e:
                logger.error(f"Missing required dependencies for netCDF files: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Error opening satellite data file: {str(e)}")
                raise
            self.satellite_data['storm_clouds'] = xr.where(
                self.satellite_data['cloud_top_temp'] < -40,
                True, False
            )
            
            # getting relevant channels
            channels = self.satellite_data.data_vars
            if 'IR' in channels:
                ir_channel = self.satellite_data['IR']
                self.satellite_data['cloud_top_temp'] = ir_channel - 273.15  # Convert to Celsius
                
                # Identify  storm cells
                self.satellite_data['storm_clouds'] = xr.where(
                    self.satellite_data['cloud_top_temp'] < -40,
                    True, False
                )
            
            logger.info(f"Successfully ingested satellite data from {sample_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting satellite data: {str(e)}")
            return False
        
    def ingest_aws_data(self, filepath):
        
        try:
            self.aws_data = pd.read_csv(filepath)
            
            # Convert timestamp to datetime
            self.aws_data['timestamp'] = pd.to_datetime(self.aws_data['timestamp'])
            self.aws_data.set_index('timestamp', inplace=True)
            
            # Handle missing values
            self.aws_data = self.aws_data.interpolate(method='time')  # Time-based interpolation
            
            # Calculate derived features
            if all(param in self.aws_data.columns for param in ['temperature', 'dewpoint']):
                # Calculate relative humidity
                self.aws_data['relative_humidity'] = metpy.calc.relative_humidity_from_dewpoint(
                    self.aws_data['temperature'] * units.degC,
                    self.aws_data['dewpoint'] * units.degC
                )
            
            # Remove impossible values
            self.aws_data = self.aws_data[
                (self.aws_data['temperature'] > -50) & 
                (self.aws_data['temperature'] < 50) &  
                (self.aws_data['wind_speed'] >= 0) &
                (self.aws_data['wind_speed'] < 200) &  
                (self.aws_data['pressure'] > 900) &
                (self.aws_data['pressure'] < 1100) 
            ]
            
            return True
        except Exception as e:
            print(f"Error ingesting AWS data: {str(e)}")
            return False
        
    def align_timestamps(self):
        
        try:
            common_freq = '5min'
            
            if self.aws_data is not None:
                time_index = self.aws_data.index
            elif self.radar_data is not None:
                time_index = pd.date_range(
                    start=self.radar_data.time.min().values,
                    end=self.radar_data.time.max().values,
                    freq=common_freq
                )
            else:
                return False
            
            
            if self.aws_data is not None:
                self.aws_data = self.aws_data.resample(common_freq).mean()
            
            
            if self.radar_data is not None:
                self.radar_data = self.radar_data.resample(time=common_freq).mean()
            
            
            if self.satellite_data is not None:
                self.satellite_data = self.satellite_data.resample(time=common_freq).mean()
            
            return True
        except Exception as e:
            print(f"Error aligning timestamps: {str(e)}")
            return False
        
    def get_current_data(self):
        
        return {
            'metar': self.metar_data,
            'radar': self.radar_data,
            'satellite': self.satellite_data,
            'aws': self.aws_data
        }

    def preprocess_data(self):
        """
        Clean and preprocess all data sources
        """
        # TODO: Implement data preprocessing
        pass
