import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv2D, MaxPooling2D, Flatten
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from typing import Tuple, List, Dict
import pandas as pd

class WindPredictor:
    def __init__(self, sequence_length: int = 24, num_features: int = 5):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = self._build_lstm_model()

    def _build_lstm_model(self) -> Sequential:
        """
        Build LSTM model for wind prediction
        """
        model = Sequential([
            LSTM(64, input_shape=(self.sequence_length, self.num_features), return_sequences=True),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(1)  # Predict wind speed
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model
        Features: wind_speed, wind_direction, pressure, temperature, humidity
        """
        # Implementation will process historical weather data
        pass
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50):
        """Train the wind prediction model"""
        self.model.fit(X, y, epochs=epochs, validation_split=0.2)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict wind speeds"""
        return self.model.predict(X)

class StormPredictor:
    def __init__(self, image_size: Tuple[int, int] = (256, 256)):
        self.image_size = image_size
        self.cnn_model = self._build_cnn_model()
        self.ensemble_model = self._build_ensemble_model()
        
    def _build_cnn_model(self) -> Sequential:
        """
        Build CNN model for radar/satellite image analysis (2-channel input)
        """
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 2)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')  # Storm probability
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def _build_ensemble_model(self):
        """
        Build ensemble model (Random Forest + Gradient Boosting)
        """
        self.rf = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gbm = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    def prepare_image_data(self, radar_images: List[np.ndarray], 
                          satellite_images: List[np.ndarray]) -> np.ndarray:
        """Prepare image data for CNN model"""
        # Implementation will process radar and satellite images
        pass
    
    def prepare_feature_data(self, weather_data: pd.DataFrame) -> np.ndarray:
        """Prepare feature data for ensemble models"""
        # Implementation will process numerical weather features
        pass
    
    def train_cnn(self, X_img: np.ndarray, y: np.ndarray, epochs: int = 50):
        """Train the CNN model"""
        self.cnn_model.fit(X_img, y, epochs=epochs, validation_split=0.2)
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray):
        """Train the ensemble models"""
        self.rf.fit(X, y)
        self.gbm.fit(X, y)
    
    def predict(self, X_img: np.ndarray, X_features: np.ndarray) -> Dict[str, float]:
        """
        Make storm predictions using both CNN and ensemble models
        Returns:
            Dictionary with probabilities and confidence scores
        """
        cnn_prob = self.cnn_model.predict(X_img)[0][0]
        rf_prob = self.rf.predict_proba(X_features)[0][1]
        gbm_prob = self.gbm.predict_proba(X_features)[0][1]
        
        # Weighted ensemble prediction
        ensemble_prob = 0.4 * cnn_prob + 0.3 * rf_prob + 0.3 * gbm_prob
        
        # Calculate confidence score based on model agreement
        predictions = [cnn_prob, rf_prob, gbm_prob]
        confidence = 1.0 - np.std(predictions)  # Higher agreement = higher confidence
        
        return {
            'storm_probability': ensemble_prob,
            'confidence_score': confidence,
            'cnn_prob': cnn_prob,
            'rf_prob': rf_prob,
            'gbm_prob': gbm_prob
        }
