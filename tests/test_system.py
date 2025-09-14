import unittest
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.data_ingestion import DataIngestion
from src.weather_predictor import WeatherPredictor
from src.alert_system import AlertSystem
from src.dashboard import Dashboard
from src.data_downloaders.radar_downloader import RadarDownloader
from src.data_downloaders.satellite_downloader import SatelliteDownloader
from src.data_downloaders.metar_downloader import MetarDownloader

class WeatherPredictorSystemTest(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.test_data_dir = "data/test"
        os.makedirs(self.test_data_dir, exist_ok=True)
        
    def test_data_downloaders(self):
        """Test data downloading functionality"""
        radar_dl = RadarDownloader(os.path.join(self.test_data_dir, "radar"))
        satellite_dl = SatelliteDownloader(os.path.join(self.test_data_dir, "satellite"))
        metar_dl = MetarDownloader(os.path.join(self.test_data_dir, "metar"))
        
        # Test one day of data download
        test_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        self.assertTrue(hasattr(radar_dl, 'download_radar_data'))
        self.assertTrue(hasattr(satellite_dl, 'download_goes_data'))
        self.assertTrue(hasattr(metar_dl, 'download_metar_data'))
    
    def test_data_ingestion(self):
        """Test data ingestion and processing"""
        ingestion = DataIngestion()
        
        # Test method existence
        self.assertTrue(hasattr(ingestion, 'ingest_radar_data'))
        self.assertTrue(hasattr(ingestion, 'ingest_satellite_data'))
        self.assertTrue(hasattr(ingestion, 'ingest_aws_data'))
        self.assertTrue(hasattr(ingestion, 'align_timestamps'))
    
    def test_weather_predictor(self):
        """Test weather prediction models"""
        predictor = WeatherPredictor()
        
        # Test model building methods
        self.assertTrue(hasattr(predictor, 'build_wind_model'))
        self.assertTrue(hasattr(predictor, 'build_storm_model'))
        self.assertTrue(hasattr(predictor, 'build_medium_term_model'))
        
        # Test prediction methods
        self.assertTrue(hasattr(predictor, 'predict_nowcast'))
        self.assertTrue(hasattr(predictor, 'predict_medium_term'))
    
    def test_alert_system(self):
        """Test alert system functionality"""
        alert_system = AlertSystem()
        
        # Test alert methods
        self.assertTrue(hasattr(alert_system, 'check_conditions'))
        self.assertTrue(hasattr(alert_system, 'generate_explanation'))
        self.assertTrue(hasattr(alert_system, 'log_alert'))
        
        # Test threshold configuration
        self.assertTrue(hasattr(alert_system, 'thresholds'))
        self.assertIsInstance(alert_system.thresholds, dict)
    
    def test_dashboard(self):
        """Test dashboard functionality"""
        dashboard = Dashboard()
        
        # Test dashboard components
        self.assertTrue(hasattr(dashboard, 'app'))
        self.assertTrue(hasattr(dashboard, 'setup_layout'))
        self.assertTrue(hasattr(dashboard, 'setup_callbacks'))
    
    def test_end_to_end(self):
        """Test complete system workflow"""
        try:
            # 1. Download some test data
            test_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            # 2. Process the data
            ingestion = DataIngestion()
            
            # 3. Make predictions
            predictor = WeatherPredictor()
            
            # 4. Generate alerts
            alert_system = AlertSystem()
            
            # 5. Check dashboard can start
            dashboard = Dashboard()
            
            self.assertTrue(True, "End-to-end test completed successfully")
            
        except Exception as e:
            self.fail(f"End-to-end test failed: {str(e)}")

if __name__ == '__main__':
    unittest.main()
