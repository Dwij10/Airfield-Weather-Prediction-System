import logging
from datetime import datetime, timedelta
import time
import threading
from pathlib import Path
from ..data_ingestion import DataIngestion

logger = logging.getLogger(__name__)

class DataCollector:
    """
    Responsible for collecting and storing weather data continuously
    """
    def __init__(self, storage_path: str = "data/raw"):
        self.storage_path = Path(storage_path)
        self.data_ingestion = DataIngestion()
        self.running = False
        self.collection_thread = None
        
    def start_collection(self, interval_minutes: int = 1):
        """Start continuous data collection"""
        if self.running:
            logger.warning("Data collection is already running")
            return
            
        self.running = True
        # Start immediate data collection
        self._collect_current_data()
        
        # Start background collection
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            args=(interval_minutes,),
            daemon=True
        )
        self.collection_thread.start()
        logger.info(f"Started data collection with {interval_minutes} minute interval")
        
    def stop_collection(self):
        """Stop data collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join()
            logger.info("Stopped data collection")
            
    def _collection_loop(self, interval_minutes: int):
        """Main data collection loop"""
        while self.running:
            try:
                start_time = time.time()
                
                # Collect all data types
                self._collect_current_data()
                
                # Calculate precise wait time
                elapsed = time.time() - start_time
                wait_time = max(0, (interval_minutes * 60) - elapsed)
                
                # Wait in small intervals to be responsive to shutdown
                while self.running and wait_time > 0:
                    sleep_time = min(1.0, wait_time)  # Sleep max 1 second at a time
                    time.sleep(sleep_time)
                    wait_time -= sleep_time
                
            except Exception as e:
                logger.error(f"Error in data collection loop: {str(e)}")
                # Wait 5 seconds before retrying on error
                time.sleep(5)
                
    def _collect_current_data(self):
        """Collect current weather data from all sources"""
        timestamp = datetime.now()
        
        try:
            # Collect METAR data
            metar_success = self.data_ingestion.ingest_metar_data()
            if metar_success:
                logger.info("Successfully collected METAR data")
            
            # Try to collect radar data
            radar_success = self.data_ingestion.ingest_radar_data()
            if radar_success:
                logger.info("Successfully collected radar data")
            else:
                logger.warning("Failed to collect radar data")
            
            # Try to collect satellite data
            satellite_success = self.data_ingestion.ingest_satellite_data()
            if satellite_success:
                logger.info("Successfully collected satellite data")
            else:
                logger.warning("Failed to collect satellite data")
                
        except Exception as e:
            logger.error(f"Error collecting data: {str(e)}")
            
    def get_collected_data_status(self):
        """Get status of collected data"""
        status = {
            'last_collection': None,
            'available_data': {
                'metar': False,
                'radar': False,
                'satellite': False
            }
        }
        
        try:
            # Check METAR data
            metar_path = self.storage_path / 'metar'
            if metar_path.exists():
                status['available_data']['metar'] = True
                
            # Check radar data
            radar_path = self.storage_path / 'radar'
            if radar_path.exists():
                status['available_data']['radar'] = True
                
            # Check satellite data
            satellite_path = self.storage_path / 'satellite'
            if satellite_path.exists():
                status['available_data']['satellite'] = True
                
            # Get last collection time
            all_paths = [metar_path, radar_path, satellite_path]
            last_modified = []
            
            for path in all_paths:
                if path.exists():
                    files = list(path.rglob('*.*'))
                    if files:
                        last_modified.extend([f.stat().st_mtime for f in files])
                        
            if last_modified:
                status['last_collection'] = datetime.fromtimestamp(max(last_modified))
                
        except Exception as e:
            logger.error(f"Error getting data status: {str(e)}")
            
        return status
