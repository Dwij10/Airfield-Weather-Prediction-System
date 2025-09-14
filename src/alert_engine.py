from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class AlertEngine:
    def __init__(self):
        self.thresholds = {
            'wind_speed': 65.0,  # km/h
            'storm_probability': 0.7,  # 70%
            'visibility': 3.0,  # km
            'wind_gust': 80.0,  # km/h
            'confidence_threshold': 0.8  # 80%
        }
        self.alerts_log = []
        
    def set_custom_thresholds(self, airfield_id: str, thresholds: Dict[str, float]):
        """Set custom thresholds for a specific airfield"""
        # TODO: Store in database
        self.thresholds.update(thresholds)
    
    def generate_alert_explanation(self, conditions: Dict[str, float],
                                 predictions: Dict[str, float]) -> str:
        """
        Generate human-readable explanation for alert
        """
        explanations = []
        
        # Current conditions analysis
        if conditions.get('wind_speed', 0) > self.thresholds['wind_speed'] * 0.8:
            explanations.append(f"High winds currently at {conditions['wind_speed']:.1f} km/h")
        
        if conditions.get('visibility', 10) < self.thresholds['visibility'] * 1.2:
            explanations.append(f"Reduced visibility at {conditions['visibility']:.1f} km")
        
        # Prediction analysis
        if predictions.get('storm_probability', 0) > self.thresholds['storm_probability']:
            conf = predictions.get('confidence_score', 0)
            exp = (f"Storm predicted with {predictions['storm_probability']*100:.0f}% probability "
                  f"(confidence: {conf*100:.0f}%)")
            
            # Add model-specific insights
            if predictions.get('cnn_prob', 0) > 0.8:
                exp += "\n• Radar/satellite patterns indicate strong storm formation"
            if predictions.get('rf_prob', 0) > 0.8:
                exp += "\n• Historical conditions match severe weather patterns"
            
            explanations.append(exp)
        
        if predictions.get('wind_gust_prediction', 0) > self.thresholds['wind_gust']:
            explanations.append(f"Gale force winds expected: {predictions['wind_gust_prediction']:.1f} km/h")
        
        return "\n".join(explanations) if explanations else "No significant weather threats detected"
    
    def check_conditions(self, 
                        current_conditions: Dict[str, float],
                        predictions: Dict[str, float]) -> Optional[Dict]:
        """
        Check if current conditions and predictions warrant an alert
        """
        alert_level = "green"
        explanations = []
        
        # Check wind conditions
        if (current_conditions.get('wind_speed', 0) > self.thresholds['wind_speed'] or
            predictions.get('wind_gust_prediction', 0) > self.thresholds['wind_gust']):
            alert_level = "red"
            explanations.append("Severe wind conditions")
        
        # Check storm probability
        storm_prob = predictions.get('storm_probability', 0)
        confidence = predictions.get('confidence_score', 0)
        if (storm_prob > self.thresholds['storm_probability'] and 
            confidence > self.thresholds['confidence_threshold']):
            alert_level = "red"
            explanations.append(f"High probability of storm: {storm_prob*100:.0f}%")
        elif storm_prob > self.thresholds['storm_probability'] * 0.7:
            alert_level = max(alert_level, "yellow")
            explanations.append(f"Moderate storm risk: {storm_prob*100:.0f}%")
        
        # Generate alert if conditions warrant
        if alert_level != "green":
            alert = {
                'timestamp': datetime.now(),
                'level': alert_level,
                'conditions': current_conditions,
                'predictions': predictions,
                'explanation': self.generate_alert_explanation(current_conditions, predictions)
            }
            self.log_alert(alert)
            return alert
        return None
    
    def log_alert(self, alert: Dict):
        """Log alert for auditing and system improvement"""
        self.alerts_log.append(alert)
        # TODO: Store in database
        
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get alerts from the last n hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts_log 
                if alert['timestamp'] > cutoff_time]
