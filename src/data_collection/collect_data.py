import logging
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data_collection.data_collector import DataCollector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main data collection service"""
    try:
        # Create data directories if they don't exist
        data_root = Path('data/raw')
        for subdir in ['radar', 'satellite', 'metar']:
            (data_root / subdir).mkdir(parents=True, exist_ok=True)
        
        # Initialize data collector
        collector = DataCollector()
        
        # Start collection with 10-minute intervals
        logger.info("Starting data collection service...")
        collector.start_collection(interval_minutes=10)
        
        try:
            # Keep the script running
            while True:
                # Print status every hour
                status = collector.get_collected_data_status()
                logger.info(f"Data collection status: {status}")
                import time
                time.sleep(3600)  # Sleep for an hour
                
        except KeyboardInterrupt:
            logger.info("Received stop signal. Shutting down...")
            collector.stop_collection()
            
    except Exception as e:
        logger.error(f"Error in data collection service: {str(e)}")
        raise

if __name__ == "__main__":
    main()
