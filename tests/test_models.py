import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.data_processing import WeatherDataProcessor
from src.models.weather_models import WindPredictor, StormPredictor
import tensorflow as tf

class TestWeatherPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        # Create sample data
        cls.dates = pd.date_range(
            start=datetime.now() - timedelta(days=2),
            end=datetime.now(),
            freq='30min'
        )
        
        cls.metar_data = pd.DataFrame({
            'timestamp': cls.dates,
            'wind_speed': np.random.normal(15, 5, len(cls.dates)),
            'wind_direction': np.random.uniform(0, 360, len(cls.dates)),
            'pressure': np.random.normal(1013, 2, len(cls.dates)),
            'temperature': np.random.normal(25, 5, len(cls.dates)),
            'humidity': np.random.uniform(40, 90, len(cls.dates)),
            'visibility': np.random.normal(10, 2, len(cls.dates)),
            'precipitation': np.random.uniform(0, 5, len(cls.dates))
        })
        
        # Create sample radar and satellite images
        cls.radar_images = [
            np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            for _ in range(10)
        ]
        cls.satellite_images = [
            np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            for _ in range(10)
        ]
    
    def test_wind_predictor(self):
        """Test wind prediction model"""
        predictor = WindPredictor()
        
        # Create sample sequence data
        X = np.random.normal(0, 1, (100, 24, 5))  # 100 sequences of 24 timesteps with 5 features
        y = np.random.normal(0, 1, 100)
        
        # Test model compilation
        self.assertIsInstance(predictor.model, tf.keras.Model)
        
        # Test training
        history = predictor.model.fit(X, y, epochs=1, verbose=0)
        self.assertIn('loss', history.history)
        
        # Test prediction
        pred = predictor.predict(X[:1])
        self.assertEqual(pred.shape, (1,))
    
    def test_storm_predictor(self):
        """Test storm prediction model"""
        predictor = StormPredictor()
        
        # Create sample data
        X_img = np.random.normal(0, 1, (100, 256, 256, 1))
        X_features = np.random.normal(0, 1, (100, 10))
        y = np.random.randint(0, 2, 100)
        
        # Test CNN model
        self.assertIsInstance(predictor.cnn_model, tf.keras.Model)
        history = predictor.cnn_model.fit(X_img, y, epochs=1, verbose=0)
        self.assertIn('loss', history.history)
        
        # Test ensemble training
        predictor.train_ensemble(X_features, y)
        
        # Test prediction
        pred = predictor.predict(X_img[:1], X_features[:1])
        self.assertIn('storm_probability', pred)
        self.assertIn('confidence_score', pred)
    
    def test_data_processor(self):
        """Test data processing pipeline"""
        processor = WeatherDataProcessor()
        
        # Test sequence data preparation
        X_seq, y_seq = processor.prepare_sequence_data(self.metar_data)
        self.assertEqual(len(X_seq.shape), 3)  # (samples, sequence_length, features)
        self.assertEqual(len(y_seq.shape), 1)  # (samples,)
        
        # Test storm data preparation
        storm_data = processor.prepare_storm_data(
            self.metar_data,
            self.radar_images[:len(self.metar_data)],
            self.satellite_images[:len(self.metar_data)]
        )
        
        self.assertIn('X_numerical', storm_data)
        self.assertIn('X_images', storm_data)
        self.assertIn('y', storm_data)
        
        # Check shapes
        self.assertEqual(len(storm_data['X_images'].shape), 4)  # (samples, height, width, channels)
        self.assertEqual(len(storm_data['X_numerical'].shape), 2)  # (samples, features)
        self.assertEqual(len(storm_data['y'].shape), 1)  # (samples,)

if __name__ == '__main__':
    unittest.main()
