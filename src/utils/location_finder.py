import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    r = 6371  # Radius of earth in kilometers
    
    return c * r

class LocationFinder:
    def __init__(self):
        # Indian Doppler Weather Radar (DWR) stations data
        self.nexrad_stations = pd.DataFrame([
            # Format: ID, Name, State/UT, Latitude, Longitude
            ['VABJ', 'Bhuj', 'Gujarat', 23.27, 69.67], # Closest DWR to Ahmedabad
            ['VANP', 'Nagpur', 'Maharashtra', 21.05, 79.03], # Another major DWR
            ['VIDP', 'Delhi', 'Delhi', 28.56, 77.10], # Delhi DWR, also major airport
            ['VABB', 'Mumbai', 'Maharashtra', 19.09, 72.86], # Mumbai DWR, also major airport
            ['VOMM', 'Chennai', 'Tamil Nadu', 12.98, 80.17] # Chennai DWR, also major airport
        ], columns=['ID', 'Name', 'State', 'Latitude', 'Longitude'])

        # Major Indian airports data
        self.airports = pd.DataFrame([
            # Format: ICAO, Name, Latitude, Longitude
            ['VAAH', 'Sardar Vallabhbhai Patel Intl', 23.0772, 72.6347],
            ['VIDP', 'Indira Gandhi Intl (Delhi)', 28.5665, 77.1031],
            ['VABB', 'Chhatrapati Shivaji Maharaj Intl (Mumbai)', 19.0886, 72.8681],
            ['VOBL', 'Kempegowda Intl (Bengaluru)', 13.1979, 77.7061],
            ['VOMM', 'Chennai Intl', 12.9880, 80.1764]
        ], columns=['ICAO', 'Name', 'Latitude', 'Longitude'])

    def find_nearest_stations(self, latitude, longitude, max_distance=500):
        """
        Find nearest DWR stations and airports within max_distance km
        """
        # Calculate distances to DWR stations
        nexrad_distances = self.nexrad_stations.apply(
            lambda row: haversine_distance(latitude, longitude, 
                                           row['Latitude'], row['Longitude']),
            axis=1
        )
        
        # Calculate distances to airports
        airport_distances = self.airports.apply(
            lambda row: haversine_distance(latitude, longitude, 
                                           row['Latitude'], row['Longitude']),
            axis=1
        )
        
        # Filter and sort DWR stations
        nearby_nexrad = pd.DataFrame({
            'Station_ID': self.nexrad_stations['ID'],
            'Name': self.nexrad_stations['Name'],
            'Distance_km': nexrad_distances
        })
        nearby_nexrad = nearby_nexrad[nearby_nexrad['Distance_km'] <= max_distance]
        nearby_nexrad = nearby_nexrad.sort_values('Distance_km')
        
        # Filter and sort airports
        nearby_airports = pd.DataFrame({
            'ICAO': self.airports['ICAO'],
            'Name': self.airports['Name'],
            'Distance_km': airport_distances
        })
        nearby_airports = nearby_airports[nearby_airports['Distance_km'] <= max_distance]
        nearby_airports = nearby_airports.sort_values('Distance_km')
        
        return nearby_nexrad, nearby_airports

# Example usage
if __name__ == "__main__":
    finder = LocationFinder()
    
    # Get user's location (example for Ahmedabad city center)
    print("Please enter your location coordinates:")
    latitude = float(input("Latitude (e.g., 23.0225 for Ahmedabad): "))
    longitude = float(input("Longitude (e.g., 72.5714 for Ahmedabad): "))
    
    # Find nearest stations
    nexrad_stations, airports = finder.find_nearest_stations(latitude, longitude)
    
    print("\nNearest Indian Doppler Weather Radar Stations:")
    print(nexrad_stations.to_string(index=False))
    
    print("\nNearest Airports:")
    print(airports.to_string(index=False))