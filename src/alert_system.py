import numpy as np
from datetime import datetime

class AlertSystem:
    def __init__(self):
        self.alerts = []
        self.thresholds = {
            'wind_speed': {
                'warning': 35,    # km/h - Initial warning level
                'moderate': 50,   # km/h - Moderate risk level
                'severe': 65,     # km/h - Severe risk level
                'critical': 80    # km/h - Critical risk level
            },
            'storm_probability': {
                'warning': 0.4,   # 40% chance of storm
                'moderate': 0.6,  # 60% chance of storm
                'severe': 0.8,    # 80% chance of storm
                'critical': 0.9   # 90% chance of storm
            },
            'pressure_drop': {
                'warning': 3,     # hPa/hour
                'severe': 5       # hPa/hour
            },
            'visibility': {
                'warning': 5000,  # meters
                'severe': 1500    # meters
            }
        }
        
    def check_conditions(self, wind_speed, storm_prob):
        """
        Check weather conditions against thresholds
        """
        alerts = []
        confidence_scores = {}
        
        # Check wind conditions
        if wind_speed > self.thresholds['wind_speed']:
            alerts.append({
                'type': 'WIND_ALERT',
                'severity': 'HIGH' if wind_speed > 65 else 'MODERATE',
                'message': f'Gale force winds expected: {wind_speed} km/h',
                'timestamp': datetime.now()
            })
            confidence_scores['wind'] = min(wind_speed / 100, 0.95)
            
        # Check storm conditions
        if storm_prob > self.thresholds['storm_probability']:
            alerts.append({
                'type': 'STORM_ALERT',
                'severity': 'HIGH' if storm_prob > 0.85 else 'MODERATE',
                'message': f'Thunderstorm probability: {storm_prob*100}%',
                'timestamp': datetime.now()
            })
            confidence_scores['storm'] = storm_prob
            
        return alerts, confidence_scores
        
    def generate_explanation(self, conditions):
        """
        Generate human-readable explanation for alerts
        """
        explanation = []
        
        if 'pressure_drop' in conditions:
            explanation.append(f"Rapid pressure drop of {conditions['pressure_drop']} hPa")
            
        if 'radar_echoes' in conditions:
            explanation.append("Strong radar echoes detected")
            
        if 'wind_gust' in conditions:
            explanation.append(f"Wind gusts of {conditions['wind_gust']} km/h recorded")
            
        return " | ".join(explanation)
        
    def log_alert(self, alert):
        """
        Log alert for audit purposes
        """
        self.alerts.append(alert)
        # TODO: Implement persistent storage of alerts
        
    def get_active_alerts(self):
        """
        Get currently active alerts
        """
        # Filter alerts from last hour
        current_time = datetime.now()
        recent_alerts = [
            alert for alert in self.alerts
            if (current_time - alert['timestamp']).total_seconds() < 3600
        ]
        return recent_alerts
        
    def update_thresholds(self, new_thresholds):
        """
        Update alert thresholds
        Args:
            new_thresholds: dict with new threshold values
        """
        self.thresholds.update(new_thresholds)
