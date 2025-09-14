import logging
import time
from src.data_ingestion import DataIngestion
from src.models.model_manager import ModelManager
from src.alert_system import AlertSystem
from src.dashboard import Dashboard
from apscheduler.schedulers.background import BackgroundScheduler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weather_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    scheduler = None
    try:
        # Initialize components
        logger.info("Initializing system components...")
        from src.data_collection.data_collector import DataCollector
        data_collector = DataCollector()
        data_collector.start_collection(interval_minutes=1)  # Collect data every minute
        
        data_ingester = DataIngestion()
        model_manager = ModelManager()
        alert_system = AlertSystem()
        dashboard = Dashboard()
        
        # Wait for first data point (max 60 seconds)
        logger.info("Waiting for initial data...")
        max_attempts = 60
        attempts = 0
        while attempts < max_attempts:
            data_status = data_collector.get_collected_data_status()
            if data_status['available_data']['metar']:
                logger.info("Initial data collected successfully")
                break
            time.sleep(1)  # Check every second
            attempts += 1
            if attempts % 10 == 0:  # Log every 10 seconds
                logger.info(f"Still waiting for data... ({attempts} seconds)")
        
        if not data_status['available_data']['metar']:
            raise RuntimeError("No METAR data available after 60 seconds. Cannot proceed without weather data.")
        
        # Check if models need retraining
        if model_manager.needs_retraining():
            logger.info("Models need retraining. Starting training process...")
            from train_models import ModelTrainer
            trainer = ModelTrainer()
            # Train with available data
            trainer.train_models(use_radar=data_status['available_data']['radar'])
            model_manager.load_latest_models()
        
        # Initialize real-time prediction loop
        def update_predictions():
            try:
                # Get latest weather data
                current_data = data_ingester.get_current_data()
                
                # Make predictions
                predictions = model_manager.predict_weather(
                    current_data['metar'],
                    current_data['radar'],
                    current_data['satellite']
                )
                
                # Check conditions and generate alerts
                alerts = alert_system.check_conditions(
                    predictions['wind_speed_prediction'],
                    predictions['storm_prediction']['storm_probability']
                )
                
                # Update dashboard data
                dashboard.update_data(predictions, alerts)
                
            except Exception as e:
                logger.error(f"Error in prediction loop: {str(e)}")
        
        # Set up periodic updates
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler()
        scheduler.add_job(update_predictions, 'interval', seconds=30)  # Update every 30 seconds
        scheduler.start()
        
        # Start the dashboard
        logger.info("Starting dashboard server...")
        dashboard.run_server(debug=True)  # disable reloader for better performance
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
    finally:
        if scheduler is not None:
            scheduler.shutdown()

if __name__ == "__main__":
    main()
