import os
import logging
from datetime import datetime
from pathlib import Path
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import json
from src.data_processing import WeatherDataProcessor
from src.models.weather_models import WindPredictor, StormPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.data_processor = WeatherDataProcessor()
        self.wind_predictor = None  # Will be initialized after knowing num_features
        self.storm_predictor = StormPredictor()
        
    def train_models(self, training_days: int = 30, validation_split: float = 0.2, use_radar: bool = True):
        """
        Train both wind and storm prediction models using recent historical data
        Args:
            training_days: Number of days of historical data to use
            validation_split: Fraction of data to use for validation
            use_radar: Whether to train storm prediction models using radar data
        """
        try:
            logger.info(f"Starting model training using {training_days} days of data...")
            
            # Get training data
            if use_radar:
                training_data = self.data_processor.get_training_data(days=training_days)
            else:
                training_data = self.data_processor.get_training_data_no_radar(days=training_days)
            
            # Train wind prediction model
            logger.info("Training wind prediction model...")
            X_wind, y_wind = training_data['wind_data']

            # Dynamically initialize WindPredictor with correct num_features
            num_features = X_wind.shape[-1]
            self.wind_predictor = WindPredictor(num_features=num_features)

            # Split data and train
            X_train, X_val, y_train, y_val = train_test_split(
                X_wind, y_wind, 
                test_size=validation_split,
                random_state=42
            )

            wind_history = self.wind_predictor.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.2,
                        patience=3
                    )
                ]
            )
            
            # Train storm prediction models
            logger.info("Training storm prediction models...")
            storm_data = training_data['storm_data']
            
            # Split data for storm prediction
            indices = np.arange(len(storm_data['y']))
            train_idx, val_idx = train_test_split(
                indices,
                test_size=validation_split,
                random_state=42
            )
            
            # Train CNN model
            storm_cnn_history = self.storm_predictor.cnn_model.fit(
                storm_data['X_images'][train_idx],
                storm_data['y'][train_idx],
                validation_data=(
                    storm_data['X_images'][val_idx],
                    storm_data['y'][val_idx]
                ),
                epochs=50,
                batch_size=32,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.2,
                        patience=3
                    )
                ]
            )
            
            # Train ensemble models
            logger.info("Training ensemble models...")
            self.storm_predictor.train_ensemble(
                storm_data['X_numerical'][train_idx],
                storm_data['y'][train_idx]
            )
            
            # Save models and training history
            self._save_models(
                wind_history.history,
                storm_cnn_history.history
            )
            
            logger.info("Model training completed successfully!")
            return {
                'wind_history': wind_history.history,
                'storm_history': storm_cnn_history.history
            }
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    def _save_models(self, wind_history: dict, storm_history: dict):
        """Save trained models and their training history"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save wind prediction model
        wind_dir = self.model_dir / f'wind_model_{timestamp}'
        wind_dir.mkdir(exist_ok=True)
        self.wind_predictor.model.save(wind_dir / 'model.keras')

        # Save storm prediction models
        storm_dir = self.model_dir / f'storm_model_{timestamp}'
        storm_dir.mkdir(exist_ok=True)
        self.storm_predictor.cnn_model.save(storm_dir / 'cnn_model.keras')

        # Save ensemble models
        import joblib
        joblib.dump(self.storm_predictor.rf, storm_dir / 'random_forest.joblib')
        joblib.dump(self.storm_predictor.gbm, storm_dir / 'gradient_boosting.joblib')

        # Save training histories
        with open(wind_dir / 'training_history.json', 'w') as f:
            json.dump(wind_history, f)
        with open(storm_dir / 'training_history.json', 'w') as f:
            json.dump(storm_history, f)

        logger.info(f"Models saved in {self.model_dir}")
    
    def validate_models(self, validation_days: int = 5):
        """Validate models on recent data"""
        try:
            logger.info(f"Starting model validation using {validation_days} days of data...")
            
            # Get validation data
            validation_data = self.data_processor.get_training_data(days=validation_days)
            
            # Validate wind prediction
            X_wind, y_wind = validation_data['wind_data']
            wind_loss, wind_mae = self.wind_predictor.model.evaluate(X_wind, y_wind)
            
            # Validate storm prediction
            storm_data = validation_data['storm_data']
            storm_loss, storm_acc = self.storm_predictor.cnn_model.evaluate(
                storm_data['X_images'],
                storm_data['y']
            )
            
            # Get ensemble predictions
            ensemble_predictions = self.storm_predictor.predict(
                storm_data['X_images'],
                storm_data['X_numerical']
            )
            
            validation_results = {
                'wind_model': {
                    'mae': float(wind_mae),
                    'mse': float(wind_loss)
                },
                'storm_model': {
                    'cnn_accuracy': float(storm_acc),
                    'ensemble_probability': float(np.mean(ensemble_predictions['storm_probability'])),
                    'confidence_score': float(np.mean(ensemble_predictions['confidence_score']))
                }
            }
            
            logger.info("Validation results:")
            logger.info(json.dumps(validation_results, indent=2))
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error during model validation: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        trainer = ModelTrainer()
        
        # Train models
        logger.info("Starting model training pipeline...")
        training_results = trainer.train_models(training_days=30)
        
        # Validate models
        logger.info("Starting model validation...")
        validation_results = trainer.validate_models(validation_days=5)
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise
