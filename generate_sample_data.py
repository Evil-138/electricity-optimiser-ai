"""
Generate realistic electricity consumption and weather data for the Energy Consumption Predictor project.
This script creates a full year of hourly data with realistic patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_electricity_data(start_date='2023-01-01', end_date='2024-01-01'):
    """Generate realistic electricity consumption data with seasonal and daily patterns."""
    
    # Create hourly datetime range
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')[:-1]  # Exclude last hour to avoid overlap
    
    # Base consumption patterns
    data = []
    
    for timestamp in date_range:
        hour = timestamp.hour
        month = timestamp.month
        day_of_week = timestamp.dayofweek
        
        # Base consumption by hour (typical daily pattern)
        if 0 <= hour < 6:  # Night
            base_consumption = np.random.normal(1.5, 0.3)
        elif 6 <= hour < 9:  # Morning rush
            base_consumption = np.random.normal(3.5, 0.5)
        elif 9 <= hour < 17:  # Daytime
            base_consumption = np.random.normal(4.2, 0.4)
        elif 17 <= hour < 22:  # Evening peak
            base_consumption = np.random.normal(7.5, 0.8)
        else:  # Late evening
            base_consumption = np.random.normal(4.0, 0.5)
        
        # Seasonal adjustments
        if month in [12, 1, 2]:  # Winter - higher consumption due to heating
            seasonal_multiplier = np.random.normal(1.4, 0.1)
        elif month in [6, 7, 8]:  # Summer - higher consumption due to cooling
            seasonal_multiplier = np.random.normal(1.3, 0.1)
        else:  # Spring/Fall
            seasonal_multiplier = np.random.normal(1.0, 0.05)
        
        # Weekend pattern (slightly different)
        if day_of_week >= 5:  # Weekend
            if 8 <= hour <= 10:  # Later morning peak on weekends
                weekend_adjustment = np.random.normal(0.8, 0.1)
            elif 10 <= hour <= 16:  # More consistent daytime usage
                weekend_adjustment = np.random.normal(1.1, 0.1)
            else:
                weekend_adjustment = np.random.normal(1.0, 0.05)
        else:
            weekend_adjustment = 1.0
        
        # Calculate final consumption
        consumption = base_consumption * seasonal_multiplier * weekend_adjustment
        
        # Add some random noise and ensure positive values
        consumption = max(0.1, consumption + np.random.normal(0, 0.2))
        
        # Occasionally add missing values (2% chance)
        if np.random.random() < 0.02:
            consumption = np.nan
        
        data.append({
            'timestamp': timestamp,
            'kwh': round(consumption, 2)
        })
    
    return pd.DataFrame(data)

def generate_weather_data(start_date='2023-01-01', end_date='2024-01-01'):
    """Generate realistic weather data with seasonal patterns."""
    
    # Create hourly datetime range
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')[:-1]
    
    data = []
    
    for timestamp in date_range:
        hour = timestamp.hour
        month = timestamp.month
        day_of_year = timestamp.dayofyear
        
        # Base temperature with seasonal variation
        base_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Seasonal cycle
        
        # Daily temperature variation
        daily_variation = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        # Add random noise
        temperature = base_temp + daily_variation + np.random.normal(0, 2)
        
        # Humidity (inversely related to temperature with noise)
        base_humidity = 70 - 0.5 * temperature + np.random.normal(0, 10)
        humidity = max(20, min(100, base_humidity))
        
        # Wind speed (more variable)
        wind_speed = max(0, np.random.exponential(8))
        
        # Precipitation (20% chance of rain)
        precipitation = np.random.exponential(2) if np.random.random() < 0.2 else 0
        
        # Cloud cover (0-100%)
        if precipitation > 0:
            cloud_cover = np.random.normal(85, 10)
        else:
            cloud_cover = np.random.normal(40, 20)
        cloud_cover = max(0, min(100, cloud_cover))
        
        # Solar irradiance (depends on time of day, season, and cloud cover)
        if 6 <= hour <= 18:  # Daylight hours
            max_irradiance = 800 + 200 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            hour_factor = np.sin(np.pi * (hour - 6) / 12)
            cloud_factor = 1 - (cloud_cover / 100) * 0.8
            solar_irradiance = max(0, max_irradiance * hour_factor * cloud_factor + np.random.normal(0, 50))
        else:
            solar_irradiance = 0
        
        # Occasionally add missing values (1% chance)
        if np.random.random() < 0.01:
            temperature = np.nan
            humidity = np.nan
        
        data.append({
            'timestamp': timestamp,
            'temperature_c': round(temperature, 1),
            'humidity_percent': round(humidity, 1),
            'wind_speed_kmh': round(wind_speed, 1),
            'precipitation_mm': round(precipitation, 1),
            'cloud_cover_percent': round(cloud_cover, 1),
            'solar_irradiance_wm2': round(solar_irradiance, 1)
        })
    
    return pd.DataFrame(data)

def main():
    """Generate and save the datasets."""
    print("Generating electricity consumption data...")
    electricity_df = generate_electricity_data()
    
    print("Generating weather data...")
    weather_df = generate_weather_data()
    
    # Create output directory if it doesn't exist
    output_dir = 'data/raw'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    electricity_df.to_csv(f'{output_dir}/electricity_consumption.csv', index=False)
    weather_df.to_csv(f'{output_dir}/weather_data.csv', index=False)
    
    print(f"Generated {len(electricity_df)} electricity records")
    print(f"Generated {len(weather_df)} weather records")
    print(f"Data saved to {output_dir}/")
    
    # Show sample data
    print("\nSample electricity data:")
    print(electricity_df.head())
    print(f"\nElectricity data shape: {electricity_df.shape}")
    print(f"Date range: {electricity_df['timestamp'].min()} to {electricity_df['timestamp'].max()}")
    
    print("\nSample weather data:")
    print(weather_df.head())
    print(f"\nWeather data shape: {weather_df.shape}")
    print(f"Date range: {weather_df['timestamp'].min()} to {weather_df['timestamp'].max()}")

if __name__ == "__main__":
    main()