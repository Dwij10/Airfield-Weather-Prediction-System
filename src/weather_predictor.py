import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np

class WeatherPredictor:
    def __init__(self):
        self.nowcast_wind_model = None
        self.nowcast_storm_model = None
        self.medium_term_model = None
        
    def build_wind_model(self, input_shape):
        """
        Build LSTM model for wind gust prediction
        """
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(1)  # Predict wind speed
        ])
        model.compile(optimizer='adam', loss='mse')
        self.nowcast_wind_model = model
        
    def build_storm_model(self, input_shape):
        """
        Build CNN model for storm cell prediction
        """
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')  # Storm probability
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        self.nowcast_storm_model = model
        
    def build_medium_term_model(self):
        """
        Build ensemble model for medium-term prediction
        """
        self.medium_term_model = {
            'rf': RandomForestRegressor(n_estimators=100),
            'gbm': GradientBoostingRegressor(n_estimators=100)
        }
        
    def train_models(self, X_wind, y_wind, X_storm, y_storm, X_medium, y_medium):
        """
        Train all weather prediction models
        """
        # Train wind model
        self.nowcast_wind_model.fit(X_wind, y_wind, epochs=50, batch_size=32)
        
        # Train storm model
        self.nowcast_storm_model.fit(X_storm, y_storm, epochs=50, batch_size=32)
        
        # Train medium-term models
        for model in self.medium_term_model.values():
            model.fit(X_medium, y_medium)
            
    def predict_nowcast(self, wind_data, radar_data):
        """
        Make short-term predictions (0-3 hours)
        """
        wind_pred = self.nowcast_wind_model.predict(wind_data)
        storm_pred = self.nowcast_storm_model.predict(radar_data)
        return wind_pred, storm_pred
        
    def predict_medium_term(self, data):
        """
        Make medium-term predictions (up to 24 hours)
        """
        predictions = {}
        for name, model in self.medium_term_model.items():
            predictions[name] = model.predict(data)
        return np.mean([pred for pred in predictions.values()], axis=0)  # Ensemble average
