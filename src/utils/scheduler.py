import schedule
import time
from datetime import datetime, timedelta
import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from download_data import download_latest_data
except ImportError as e:
    logging.error(f"Import error: {e}")
    logging.error(f"Python path: {sys.path}")
    logging.error(f"Current directory: {os.getcwd()}")
    raise

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)

def scheduled_download():
    """
    Function to run the scheduled download
    """
    try:
        logging.info("Starting scheduled data download")
        download_latest_data()
        logging.info("Successfully completed data download")
    except Exception as e:
        logging.error(f"Error during data download: {str(e)}")

def setup_scheduler(interval_hours=1):
    """
    Set up the scheduler to run at specified intervals
    Args:
        interval_hours: How often to download data (in hours)
    """
    # Schedule the job
    schedule.every(interval_hours).hours.do(scheduled_download)
    
    logging.info(f"Scheduler set up to run every {interval_hours} hours")
    
    # Run the first download immediately
    scheduled_download()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Wait one minute before checking again

if __name__ == "__main__":
    # Create data directories if they don't exist
    for subdir in ['radar', 'satellite', 'metar']:
        os.makedirs(os.path.join('data', 'raw', subdir), exist_ok=True)
    
    # Start the scheduler
    setup_scheduler(interval_hours=1)  # Update data every hour
