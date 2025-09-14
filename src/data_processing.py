import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import xarray as xr
from datetime import datetime, timedelta
import cv2
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherDataProcessor:
    def __init__(self, data_root: str = 'data/raw'):
        self.data_root = Path(data_root)
        self.scaler = StandardScaler()
        
    def load_metar_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Load and process METAR data for the specified date range
        """
        try:
            metar_files = []
            current_date = start_date
            while current_date <= end_date:
                file_path = (self.data_root / 'metar' / 
                           str(current_date.year) / 
                           f"{current_date.month:02d}" /
                           f"VAAH_{current_date.strftime('%Y-%m-%d')}*.csv")
                metar_files.extend(list(Path().glob(str(file_path))))
                current_date += timedelta(days=1)
            
            if not metar_files:
                raise FileNotFoundError(f"No METAR data found between {start_date} and {end_date}")
            
            # Read and concatenate all files
            dfs = []
            for file in metar_files:
                # Skip comment lines starting with #
                df = pd.read_csv(file, comment='#')
                # Rename 'valid' column to 'timestamp' and convert to datetime
                df['timestamp'] = pd.to_datetime(df['valid'])
                dfs.append(df)
            
            metar_data = pd.concat(dfs, ignore_index=True)
            metar_data = metar_data.sort_values('timestamp')
            
            return metar_data
            
        except Exception as e:
            logger.error(f"Error loading METAR data: {str(e)}")
            raise
    
    def load_radar_data(self, start_date: datetime, end_date: datetime) -> List[np.ndarray]:
        """
        Load and process radar data into image arrays
        """
        try:
            radar_files = []
            current_date = start_date
            while current_date <= end_date:
                file_path = (self.data_root / 'radar' / 
                           str(current_date.year) /
                           f"{current_date.month:02d}" /
                           f"{current_date.day:02d}" /
                           '*.nc')
                radar_files.extend(list(Path().glob(str(file_path))))
                current_date += timedelta(days=1)
            
            if not radar_files:
                raise FileNotFoundError(f"No radar data found between {start_date} and {end_date}")
            
            # Process radar files into normalized images
            radar_images = []
            for file in radar_files:
                with xr.open_dataset(file) as ds:
                    # Extract reflectivity data and convert to dBZ
                    refl = ds['reflectivity'].values
                    # Normalize to 0-255 range for image processing
                    refl_norm = ((refl - refl.min()) * (255 / (refl.max() - refl.min()))).astype(np.uint8)
                    # Resize to standard size (256x256)
                    refl_resized = cv2.resize(refl_norm, (256, 256))
                    radar_images.append(refl_resized)
            
            return radar_images
            
        except Exception as e:
            logger.error(f"Error loading radar data: {str(e)}")
            raise
    
    def load_satellite_data(self, start_date: datetime, end_date: datetime) -> List[np.ndarray]:
        """
        Load and process satellite data into image arrays
        """
        try:
            satellite_files = []
            current_date = start_date
            while current_date <= end_date:
                file_path = (self.data_root / 'satellite' / 
                           current_date.strftime('%Y/%m/%d') /
                           '*.nc')
                satellite_files.extend(list(Path().glob(str(file_path))))
                current_date += timedelta(days=1)
            
            if not satellite_files:
                raise FileNotFoundError(f"No satellite data found between {start_date} and {end_date}")
            
            # Process satellite files into normalized images
            satellite_images = []
            for file in satellite_files:
                with xr.open_dataset(file) as ds:
                    # Extract cloud top temperature/brightness
                    # Try both new and legacy variable names
                    try:
                        data = ds['brightness_temperature'].values
                    except KeyError:
                        # Fallback to IR data if brightness_temperature is not available
                        data = ds['IR'].values
                    # Normalize to 0-255 range
                    data_norm = ((data - data.min()) * (255 / (data.max() - data.min()))).astype(np.uint8)
                    # Resize to standard size (256x256)
                    data_resized = cv2.resize(data_norm, (256, 256))
                    satellite_images.append(data_resized)
            
            return satellite_images
            
        except Exception as e:
            logger.error(f"Error loading satellite data: {str(e)}")
            raise
    
    def prepare_sequence_data(self, df: pd.DataFrame, 
                            sequence_length: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequential data for LSTM model
        """
        logger.info(f"Available columns in data: {list(df.columns)}")
        
        # Define feature mappings with fallbacks
        wind_speed_cols = ['wind_speed', 'sknt']  # sknt is the METAR column for wind speed in knots
        wind_dir_cols = ['wind_direction', 'drct']  # drct is the METAR column for wind direction
        temp_cols = ['temperature', 'tmpf']  # tmpf is the METAR column for temperature in Fahrenheit
        pressure_cols = ['pressure', 'mslp']  # mslp is mean sea level pressure
        humidity_cols = ['humidity', 'relh']  # relh is relative humidity
        
        # Find available features
        features = []
        
        # Find wind speed (required)
        wind_speed_col = next((col for col in wind_speed_cols if col in df.columns), None)
        if wind_speed_col is None:
            raise ValueError(f"No wind speed column found. Looked for: {wind_speed_cols}")
        features.append(wind_speed_col)
        
        # Add optional features if available
        for feat_cols in [wind_dir_cols, temp_cols, pressure_cols]:
            col = next((col for col in feat_cols if col in df.columns), None)
            if col is not None:
                features.append(col)
        
        logger.info(f"Using features for sequence data: {features}")
        
        # Convert data to numeric and handle any remaining missing values
        data = df[features].apply(pd.to_numeric, errors='coerce')
        
        # Fill any remaining missing values with forward fill then backward fill (avoid deprecated method argument)
        data = data.ffill().bfill()
        
        # Convert to numpy array
        data = data.values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i+sequence_length])
            y.append(scaled_data[i+sequence_length, 0])  # wind_speed is the target
        
        return np.array(X), np.array(y)
    
    def prepare_storm_data(self, 
                         metar_df: pd.DataFrame,
                         radar_images: List[np.ndarray],
                         satellite_images: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Prepare data for storm prediction, robust to missing columns and variable names
        """
        # Define feature mappings with fallbacks (harmonize with prepare_sequence_data)
        wind_speed_cols = ['wind_speed', 'sknt']
        wind_dir_cols = ['wind_direction', 'drct']
        temp_cols = ['temperature', 'tmpf']
        pressure_cols = ['pressure', 'mslp']
        humidity_cols = ['humidity', 'relh']
        visibility_cols = ['visibility', 'vsby']
        precipitation_cols = ['precipitation', 'p01i']

        # Find available features
        features = []
        feature_names = []
        # Required: wind speed
        wind_speed_col = next((col for col in wind_speed_cols if col in metar_df.columns), None)
        if wind_speed_col is None:
            raise ValueError(f"No wind speed column found for storm data. Looked for: {wind_speed_cols}")
        features.append(wind_speed_col)
        feature_names.append('wind_speed')
        # Optional features
        for feat_cols, name in zip(
            [wind_dir_cols, temp_cols, pressure_cols, humidity_cols, visibility_cols, precipitation_cols],
            ['wind_direction', 'temperature', 'pressure', 'humidity', 'visibility', 'precipitation']):
            col = next((col for col in feat_cols if col in metar_df.columns), None)
            if col is not None:
                features.append(col)
                feature_names.append(name)

        logger.info(f"Using features for storm data: {features}")

        # Convert data to numeric and handle missing values
        data = metar_df[features].apply(pd.to_numeric, errors='coerce')
        data = data.ffill().bfill()
        # Fill any remaining NaNs with column means (for sklearn compatibility)
        data = data.fillna(data.mean())
        # Drop columns that are all NaN (mean is still NaN)
        data = data.dropna(axis=1, how='all')
        X_numerical = self.scaler.fit_transform(data.values)
        # Fill any remaining NaNs in scaled data with zero
        X_numerical = np.nan_to_num(X_numerical)

        # Combine radar and satellite images (handle length mismatch)
        min_len = min(len(radar_images), len(satellite_images), len(X_numerical))
        X_images = np.array([np.dstack([r, s]) for r, s in zip(radar_images[:min_len], satellite_images[:min_len])])
        X_numerical = X_numerical[:min_len]
        # Target: storm occurred if wind speed > 50 (use the actual wind speed column)
        y = (data[wind_speed_col][:min_len] > 50).astype(int).values

        return {
            'X_numerical': X_numerical,
            'X_images': X_images,
            'y': y
        }
    
    def get_training_data(self, days: int = 30) -> Dict:
        """
        Get training data for the last n days including radar data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Load all data sources
        metar_data = self.load_metar_data(start_date, end_date)
        radar_images = self.load_radar_data(start_date, end_date)
        satellite_images = self.load_satellite_data(start_date, end_date)
        
        # Prepare data for both models
        X_seq, y_seq = self.prepare_sequence_data(metar_data)
        storm_data = self.prepare_storm_data(metar_data, radar_images, satellite_images)
        
        return {
            'wind_data': (X_seq, y_seq),
            'storm_data': storm_data
        }
        
    def get_training_data_no_radar(self, days: int = 30) -> Dict:
        """
        Get training data for the last n days without requiring radar data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Load METAR data only
        metar_data = self.load_metar_data(start_date, end_date)
        
        # Prepare wind prediction data
        X_seq, y_seq = self.prepare_sequence_data(metar_data)
        
        # Create minimal storm data using METAR only, robust to missing columns
        wind_speed_cols = ['wind_speed', 'sknt']
        wind_dir_cols = ['wind_direction', 'drct']
        temp_cols = ['temperature', 'tmpf']
        pressure_cols = ['pressure', 'mslp']
        humidity_cols = ['humidity', 'relh']
        visibility_cols = ['visibility', 'vsby']

        features = []
        feature_names = []
        wind_speed_col = next((col for col in wind_speed_cols if col in metar_data.columns), None)
        if wind_speed_col is None:
            raise ValueError(f"No wind speed column found for storm data. Looked for: {wind_speed_cols}")
        features.append(wind_speed_col)
        feature_names.append('wind_speed')
        for feat_cols, name in zip(
            [wind_dir_cols, temp_cols, pressure_cols, humidity_cols, visibility_cols],
            ['wind_direction', 'temperature', 'pressure', 'humidity', 'visibility']):
            col = next((col for col in feat_cols if col in metar_data.columns), None)
            if col is not None:
                features.append(col)
                feature_names.append(name)

        data = metar_data[features].apply(pd.to_numeric, errors='coerce')
        data = data.ffill().bfill()
        # Fill any remaining NaNs with column means (for sklearn compatibility)
        data = data.fillna(data.mean())
        # Drop columns that are all NaN (mean is still NaN)
        data = data.dropna(axis=1, how='all')
        X_numerical = self.scaler.fit_transform(data.values)
        # Fill any remaining NaNs in scaled data with zero
        X_numerical = np.nan_to_num(X_numerical)
        y = (data[wind_speed_col] > 50).astype(int).values

        storm_data = {
            'X_numerical': X_numerical,
            'X_images': np.zeros((len(X_numerical), 256, 256, 2)),  # Dummy image data
            'y': y
        }

        return {
            'wind_data': (X_seq, y_seq),
            'storm_data': storm_data
        }
