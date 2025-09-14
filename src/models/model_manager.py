import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import joblib
import tensorflow as tf
from typing import Dict, Tuple, Optional
from src.models.weather_models import WindPredictor, StormPredictor
from src.data_processing import WeatherDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = Path(model_dir)
        self.data_processor = WeatherDataProcessor()
        self.wind_predictor = None
        self.storm_predictor = None
        self.last_training_time = None
        self.model_version = None
        self.load_latest_models()
        
    def load_latest_models(self):
        """
        Load the most recent versions of trained models
        """
        try:
            # Find the most recent model versions
            wind_models = list(self.model_dir.glob('wind_model_*'))
            storm_models = list(self.model_dir.glob('storm_model_*'))
            
            if wind_models and storm_models:
                latest_wind = max(wind_models, key=lambda x: x.stat().st_mtime)
                latest_storm = max(storm_models, key=lambda x: x.stat().st_mtime)
                
                # Load wind prediction model
                self.wind_predictor = WindPredictor()
                self.wind_predictor.model = tf.keras.models.load_model(
                    latest_wind / 'model.keras'
                )
                
                # Load storm prediction models
                self.storm_predictor = StormPredictor()
                self.storm_predictor.cnn_model = tf.keras.models.load_model(
                    latest_storm / 'cnn_model.keras'
                )
                self.storm_predictor.rf = joblib.load(
                    latest_storm / 'random_forest.joblib'
                )
                self.storm_predictor.gbm = joblib.load(
                    latest_storm / 'gradient_boosting.joblib'
                )
                
                # Get model version from directory name
                self.model_version = latest_wind.name.split('_')[2]
                self.last_training_time = datetime.fromtimestamp(
                    latest_wind.stat().st_mtime
                )
                
                logger.info(f"Loaded models version {self.model_version}")
                return True
            else:
                logger.warning("No trained models found. Using new models.")
                self.wind_predictor = WindPredictor()
                self.storm_predictor = StormPredictor()
                return False
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def predict_weather(self, 
                       metar_data: Dict,
                       radar_image: np.ndarray,
                       satellite_image: np.ndarray) -> Dict:
        """
        Make weather predictions using current conditions
        """
        try:
            # Prepare data for wind prediction
            wind_sequence = self.data_processor.prepare_sequence_data(
                metar_data
            )
            
            # Prepare data for storm prediction
            storm_data = self.data_processor.prepare_storm_data(
                metar_data,
                [radar_image],
                [satellite_image]
            )
            
            # Make predictions
            wind_prediction = self.wind_predictor.predict(wind_sequence)
            storm_prediction = self.storm_predictor.predict(
                storm_data['X_images'],
                storm_data['X_numerical']
            )
            
            return {
                'wind_speed_prediction': wind_prediction[0],
                'storm_prediction': storm_prediction,
                'model_version': self.model_version,
                'prediction_time': datetime.now(),
                'confidence_scores': {
                    'wind': 1.0 - np.std(wind_prediction) / np.mean(wind_prediction),
                    'storm': storm_prediction['confidence_score']
                }
            }
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def needs_retraining(self, max_age_days: int = 7) -> bool:
        """
        Check if models need retraining based on age
        """
        if not self.last_training_time:
            return True
            
        age = datetime.now() - self.last_training_time
        return age.days >= max_age_days
    
    def backup_models(self):
        """
        Create a backup of current models
        """
        try:
            backup_dir = self.model_dir / 'backups' / datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            if self.model_version:
                # Copy current models to backup
                wind_dir = self.model_dir / f'wind_model_{self.model_version}'
                storm_dir = self.model_dir / f'storm_model_{self.model_version}'
                
                if wind_dir.exists() and storm_dir.exists():
                    import shutil
                    shutil.copytree(wind_dir, backup_dir / 'wind_model')
                    shutil.copytree(storm_dir, backup_dir / 'storm_model')
                    
                    logger.info(f"Models backed up to {backup_dir}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error backing up models: {str(e)}")
            raise
    
    def get_model_metrics(self) -> Dict:
        """
        Get current model performance metrics
        """
        return {
            'version': self.model_version,
            'last_training': self.last_training_time,
            'wind_model': {
                'mse': float(self.wind_predictor.model.evaluate(
                    self.data_processor.get_validation_data()['wind_data'][0],
                    self.data_processor.get_validation_data()['wind_data'][1]
                )[0])
            },
            'storm_model': {
                'accuracy': float(self.storm_predictor.cnn_model.evaluate(
                    self.data_processor.get_validation_data()['storm_data']['X_images'],
                    self.data_processor.get_validation_data()['storm_data']['y']
                )[1])
            }
        }
