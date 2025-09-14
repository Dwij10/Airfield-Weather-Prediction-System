import pandas as pd
import numpy as np

def validate_metar_data(df):
    """
    Validate METAR data format and content
    Args:
        df: pandas DataFrame containing METAR data
    Returns:
        bool: True if data is valid, False otherwise
        str: Error message if data is invalid, None otherwise
    """
    expected_columns = [
        'station', 'valid', 'tmpf', 'dwpf', 'relh', 'sknt',
        'alti', 'vsby', 'skyc1', 'skyl1', 'wxcodes'
    ]
    
    # Check required columns
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        return False, f"Missing expected columns: {missing_cols}"
    
    # Check for empty dataset
    if len(df) == 0:
        return False, "Dataset is empty"
    
    # Verify station consistency
    if len(df['station'].unique()) > 1:
        return False, "Multiple stations found in single file"
    
    # Check value ranges
    if not (df['tmpf'].between(-100, 150, inclusive='both').all()):
        return False, "Temperature values out of reasonable range"
    
    if not (df['vsby'].between(0, 100, inclusive='both').all()):
        return False, "Visibility values out of reasonable range"
    
    if not (df['sknt'].between(0, 200, inclusive='both').all()):
        return False, "Wind speed values out of reasonable range"
    
    return True, None

def clean_metar_data(df):
    """
    Clean and process METAR data
    Args:
        df: pandas DataFrame containing METAR data
    Returns:
        pandas.DataFrame: Cleaned data
    """
    # Create a copy to avoid modifying the original
    cleaned = df.copy()
    
    # Convert units and rename columns
    cleaned['temperature_c'] = (cleaned['tmpf'] - 32) * 5/9  # Fahrenheit to Celsius
    cleaned['dewpoint_c'] = (cleaned['dwpf'] - 32) * 5/9    # Fahrenheit to Celsius
    cleaned['wind_speed_kmh'] = cleaned['sknt'] * 1.852     # Knots to km/h
    cleaned['visibility_km'] = cleaned['vsby'] * 1.60934    # Miles to km
    cleaned['pressure_hpa'] = cleaned['alti'] * 33.8639     # inHg to hPa
    
    # Convert timestamp to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(cleaned['valid']):
        cleaned['valid'] = pd.to_datetime(cleaned['valid'])
    
    # Process sky conditions
    def interpret_sky_condition(skyc, skyl):
        if pd.isna(skyc) or skyc == 'M':
            return None
        return f"{skyc}{int(skyl/100):03d}" if not pd.isna(skyl) else skyc
    
    # Combine sky condition layers
    sky_conditions = []
    for i in range(1, 5):
        skyc = f'skyc{i}'
        skyl = f'skyl{i}'
        if skyc in cleaned.columns and skyl in cleaned.columns:
            layer = cleaned.apply(lambda row: interpret_sky_condition(row[skyc], row[skyl]), axis=1)
            sky_conditions.append(layer)
    
    cleaned['sky_conditions'] = [' '.join(filter(None, layers)) for layers in zip(*sky_conditions)]
    
    # Select and rename final columns
    result = pd.DataFrame({
        'timestamp': cleaned['valid'],
        'station': cleaned['station'],
        'temperature_c': cleaned['temperature_c'],
        'dewpoint_c': cleaned['dewpoint_c'],
        'relative_humidity': cleaned['relh'],
        'wind_speed_kmh': cleaned['wind_speed_kmh'],
        'wind_direction': cleaned['drct'],
        'visibility_km': cleaned['visibility_km'],
        'pressure_hpa': cleaned['pressure_hpa'],
        'sky_conditions': cleaned['sky_conditions'],
        'weather_codes': cleaned['wxcodes'],
        'raw_metar': cleaned['metar']
    })
    
    # Sort by timestamp
    result = result.sort_values('timestamp')
    
    # Interpolate missing values where appropriate
    numeric_cols = ['temperature_c', 'dewpoint_c', 'relative_humidity', 
                   'wind_speed_kmh', 'wind_direction', 'visibility_km', 
                   'pressure_hpa']
    
    for col in numeric_cols:
        if col in result.columns:
            result[col] = result[col].interpolate(method='linear', 
                                                limit=3,  # Only interpolate gaps up to 3 points
                                                limit_direction='both')
    
    return result
