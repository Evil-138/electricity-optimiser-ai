"""
Feature engineering module for Energy Consumption Predictor.
Creates lag features, rolling statistics, calendar features, and holiday flags.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    print("Warning: holidays package not available. Install with: pip install holidays")

class FeatureEngineer:
    """
    Handles feature engineering tasks including:
    - Lag features (t-1, t-24, t-168 hours)
    - Rolling window statistics
    - Calendar and holiday features
    - Weather interaction features
    - Seasonal decomposition features
    """
    
    def __init__(self, target_col: str = 'kwh', country: str = 'US'):
        """
        Initialize the feature engineer.
        
        Args:
            target_col: Name of the target variable column
            country: Country code for holiday calendar
        """
        self.target_col = target_col
        self.country = country
        self.feature_names = []
        
        # Initialize holidays calendar if available
        if HOLIDAYS_AVAILABLE:
            self.holidays_calendar = holidays.country_holidays(country)
        else:
            self.holidays_calendar = None
    
    def create_lag_features(self, df: pd.DataFrame, 
                           lag_hours: List[int] = [1, 2, 3, 6, 12, 24, 48, 72, 168]) -> pd.DataFrame:
        """
        Create lag features for the target variable.
        
        Args:
            df: Input DataFrame with datetime index
            lag_hours: List of lag periods in hours
            
        Returns:
            DataFrame with lag features added
        """
        print(f"Creating lag features for periods: {lag_hours}")
        
        df_copy = df.copy()
        
        for lag in lag_hours:
            feature_name = f'{self.target_col}_lag_{lag}h'
            df_copy[feature_name] = df_copy[self.target_col].shift(lag)
            self.feature_names.append(feature_name)
        
        print(f"Created {len(lag_hours)} lag features")
        return df_copy
    
    def create_rolling_features(self, df: pd.DataFrame, 
                               windows: List[int] = [6, 12, 24, 48, 168],
                               stats: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
        """
        Create rolling window statistical features.
        
        Args:
            df: Input DataFrame with datetime index
            windows: List of window sizes in hours
            stats: List of statistical functions to apply
            
        Returns:
            DataFrame with rolling features added
        """
        print(f"Creating rolling features for windows: {windows}, stats: {stats}")
        
        df_copy = df.copy()
        
        for window in windows:
            for stat in stats:
                feature_name = f'{self.target_col}_rolling_{window}h_{stat}'
                
                if stat == 'mean':
                    df_copy[feature_name] = df_copy[self.target_col].rolling(window=window).mean()
                elif stat == 'std':
                    df_copy[feature_name] = df_copy[self.target_col].rolling(window=window).std()
                elif stat == 'min':
                    df_copy[feature_name] = df_copy[self.target_col].rolling(window=window).min()
                elif stat == 'max':
                    df_copy[feature_name] = df_copy[self.target_col].rolling(window=window).max()
                elif stat == 'median':
                    df_copy[feature_name] = df_copy[self.target_col].rolling(window=window).median()
                elif stat == 'q25':
                    df_copy[feature_name] = df_copy[self.target_col].rolling(window=window).quantile(0.25)
                elif stat == 'q75':
                    df_copy[feature_name] = df_copy[self.target_col].rolling(window=window).quantile(0.75)
                
                self.feature_names.append(feature_name)
        
        print(f"Created {len(windows) * len(stats)} rolling features")
        return df_copy
    
    def create_exponential_smoothing_features(self, df: pd.DataFrame,
                                            alphas: List[float] = [0.1, 0.3, 0.5, 0.7]) -> pd.DataFrame:
        """
        Create exponentially weighted moving average features.
        
        Args:
            df: Input DataFrame with datetime index
            alphas: List of smoothing parameters
            
        Returns:
            DataFrame with EMA features added
        """
        print(f"Creating exponential smoothing features for alphas: {alphas}")
        
        df_copy = df.copy()
        
        for alpha in alphas:
            feature_name = f'{self.target_col}_ema_alpha_{alpha}'
            df_copy[feature_name] = df_copy[self.target_col].ewm(alpha=alpha).mean()
            self.feature_names.append(feature_name)
        
        print(f"Created {len(alphas)} exponential smoothing features")
        return df_copy
    
    def create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced calendar-based features.
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            DataFrame with calendar features added
        """
        print("Creating calendar features...")
        
        df_copy = df.copy()
        
        # Business day features
        df_copy['is_business_day'] = df_copy.index.to_series().dt.dayofweek < 5
        df_copy['is_business_day'] = df_copy['is_business_day'].astype(int)
        
        # Season features
        df_copy['season'] = df_copy.index.month % 12 // 3 + 1
        df_copy['is_winter'] = (df_copy['season'] == 1).astype(int)
        df_copy['is_spring'] = (df_copy['season'] == 2).astype(int)
        df_copy['is_summer'] = (df_copy['season'] == 3).astype(int)
        df_copy['is_fall'] = (df_copy['season'] == 4).astype(int)
        
        # Time of day features
        df_copy['is_morning'] = ((df_copy.index.hour >= 6) & (df_copy.index.hour < 12)).astype(int)
        df_copy['is_afternoon'] = ((df_copy.index.hour >= 12) & (df_copy.index.hour < 18)).astype(int)
        df_copy['is_evening'] = ((df_copy.index.hour >= 18) & (df_copy.index.hour < 22)).astype(int)
        df_copy['is_night'] = ((df_copy.index.hour >= 22) | (df_copy.index.hour < 6)).astype(int)
        
        # Peak hours (typical high energy usage times)
        df_copy['is_peak_morning'] = ((df_copy.index.hour >= 7) & (df_copy.index.hour <= 9)).astype(int)
        df_copy['is_peak_evening'] = ((df_copy.index.hour >= 17) & (df_copy.index.hour <= 21)).astype(int)
        df_copy['is_peak_hours'] = (df_copy['is_peak_morning'] | df_copy['is_peak_evening']).astype(int)
        
        # Week number
        df_copy['week_of_year'] = df_copy.index.isocalendar().week
        
        # Days since/until weekend
        df_copy['days_to_weekend'] = 4 - df_copy.index.dayofweek
        df_copy['days_to_weekend'] = df_copy['days_to_weekend'].where(df_copy['days_to_weekend'] >= 0, 
                                                                     df_copy['days_to_weekend'] + 7)
        
        calendar_features = ['is_business_day', 'is_winter', 'is_spring', 'is_summer', 'is_fall',
                           'is_morning', 'is_afternoon', 'is_evening', 'is_night',
                           'is_peak_morning', 'is_peak_evening', 'is_peak_hours',
                           'week_of_year', 'days_to_weekend']
        
        self.feature_names.extend(calendar_features)
        print(f"Created {len(calendar_features)} calendar features")
        
        return df_copy
    
    def create_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create holiday-based features.
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            DataFrame with holiday features added
        """
        if not self.holidays_calendar:
            print("Holiday calendar not available, skipping holiday features")
            return df
            
        print("Creating holiday features...")
        
        df_copy = df.copy()
        
        # Check if date is a holiday
        df_copy['is_holiday'] = df_copy.index.date.map(lambda x: x in self.holidays_calendar).astype(int)
        
        # Days before/after holiday
        df_copy['days_to_holiday'] = 0
        df_copy['days_from_holiday'] = 0
        
        for i, date in enumerate(df_copy.index.date):
            # Find nearest holiday
            min_days_to = float('inf')
            min_days_from = float('inf')
            
            for holiday_date in self.holidays_calendar.keys():
                if isinstance(holiday_date, pd.Timestamp):
                    holiday_date = holiday_date.date()
                    
                days_diff = (holiday_date - date).days
                
                if days_diff > 0 and days_diff < min_days_to:
                    min_days_to = days_diff
                elif days_diff < 0 and abs(days_diff) < min_days_from:
                    min_days_from = abs(days_diff)
            
            if min_days_to != float('inf') and min_days_to <= 7:
                df_copy.iloc[i, df_copy.columns.get_loc('days_to_holiday')] = min_days_to
            if min_days_from != float('inf') and min_days_from <= 7:
                df_copy.iloc[i, df_copy.columns.get_loc('days_from_holiday')] = min_days_from
        
        # Holiday period features
        df_copy['is_holiday_period'] = ((df_copy['is_holiday'] == 1) | 
                                       (df_copy['days_to_holiday'] <= 1) |
                                       (df_copy['days_from_holiday'] <= 1)).astype(int)
        
        holiday_features = ['is_holiday', 'days_to_holiday', 'days_from_holiday', 'is_holiday_period']
        self.feature_names.extend(holiday_features)
        print(f"Created {len(holiday_features)} holiday features")
        
        return df_copy
    
    def create_weather_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between weather variables.
        
        Args:
            df: Input DataFrame with weather columns
            
        Returns:
            DataFrame with weather interaction features added
        """
        print("Creating weather interaction features...")
        
        df_copy = df.copy()
        weather_features = []
        
        # Temperature-based features
        if 'temperature_c' in df_copy.columns:
            # Heating/cooling degree days (base 18°C)
            df_copy['heating_degree_hours'] = np.maximum(0, 18 - df_copy['temperature_c'])
            df_copy['cooling_degree_hours'] = np.maximum(0, df_copy['temperature_c'] - 22)
            weather_features.extend(['heating_degree_hours', 'cooling_degree_hours'])
            
            # Temperature categories
            df_copy['temp_very_cold'] = (df_copy['temperature_c'] < 0).astype(int)
            df_copy['temp_cold'] = ((df_copy['temperature_c'] >= 0) & (df_copy['temperature_c'] < 10)).astype(int)
            df_copy['temp_mild'] = ((df_copy['temperature_c'] >= 10) & (df_copy['temperature_c'] < 25)).astype(int)
            df_copy['temp_warm'] = ((df_copy['temperature_c'] >= 25) & (df_copy['temperature_c'] < 35)).astype(int)
            df_copy['temp_hot'] = (df_copy['temperature_c'] >= 35).astype(int)
            weather_features.extend(['temp_very_cold', 'temp_cold', 'temp_mild', 'temp_warm', 'temp_hot'])
        
        # Humidity and temperature interaction
        if all(col in df_copy.columns for col in ['temperature_c', 'humidity_percent']):
            # Heat index (feels-like temperature)
            df_copy['apparent_temperature'] = df_copy['temperature_c'] + 0.33 * (df_copy['humidity_percent'] / 100) - 0.70 * df_copy.get('wind_speed_kmh', 0) - 4.00
            weather_features.append('apparent_temperature')
            
            # Comfort index
            df_copy['comfort_index'] = 100 - ((df_copy['temperature_c'] - 22) ** 2 + (df_copy['humidity_percent'] - 50) ** 2) / 50
            weather_features.append('comfort_index')
        
        # Weather severity index
        if 'precipitation_mm' in df_copy.columns and 'wind_speed_kmh' in df_copy.columns:
            df_copy['weather_severity'] = (df_copy['precipitation_mm'] * 0.5 + 
                                         np.maximum(0, df_copy['wind_speed_kmh'] - 20) * 0.1)
            weather_features.append('weather_severity')
        
        # Solar efficiency (solar irradiance vs cloud cover)
        if all(col in df_copy.columns for col in ['solar_irradiance_wm2', 'cloud_cover_percent']):
            df_copy['solar_efficiency'] = df_copy['solar_irradiance_wm2'] * (100 - df_copy['cloud_cover_percent']) / 100
            weather_features.append('solar_efficiency')
        
        self.feature_names.extend(weather_features)
        print(f"Created {len(weather_features)} weather interaction features")
        
        return df_copy
    
    def create_time_based_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between time and other variables.
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            DataFrame with time interaction features added
        """
        print("Creating time-based interaction features...")
        
        df_copy = df.copy()
        interaction_features = []
        
        # Hour-day interactions
        if 'hour' in df_copy.columns and 'day_of_week' in df_copy.columns:
            df_copy['hour_weekday_interaction'] = df_copy['hour'] * (df_copy['day_of_week'] < 5).astype(int)
            df_copy['hour_weekend_interaction'] = df_copy['hour'] * (df_copy['day_of_week'] >= 5).astype(int)
            interaction_features.extend(['hour_weekday_interaction', 'hour_weekend_interaction'])
        
        # Season-weather interactions
        if all(col in df_copy.columns for col in ['temperature_c', 'month']):
            df_copy['temp_month_interaction'] = df_copy['temperature_c'] * df_copy['month']
            interaction_features.append('temp_month_interaction')
        
        # Weekend pattern indicators
        if 'day_of_week' in df_copy.columns and 'hour' in df_copy.columns:
            # Weekend morning (later wake-up)
            df_copy['weekend_morning'] = ((df_copy['day_of_week'] >= 5) & 
                                        (df_copy['hour'] >= 8) & 
                                        (df_copy['hour'] <= 11)).astype(int)
            
            # Weekday rush hours
            df_copy['weekday_rush'] = ((df_copy['day_of_week'] < 5) & 
                                     (((df_copy['hour'] >= 7) & (df_copy['hour'] <= 9)) |
                                      ((df_copy['hour'] >= 17) & (df_copy['hour'] <= 19)))).astype(int)
            
            interaction_features.extend(['weekend_morning', 'weekday_rush'])
        
        self.feature_names.extend(interaction_features)
        print(f"Created {len(interaction_features)} time-based interaction features")
        
        return df_copy
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features based on historical patterns.
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            DataFrame with statistical features added
        """
        print("Creating statistical features...")
        
        df_copy = df.copy()
        stat_features = []
        
        # Same hour last week comparison
        if len(df_copy) >= 168:  # Need at least a week of data
            df_copy[f'{self.target_col}_vs_same_hour_last_week'] = (
                df_copy[self.target_col] / df_copy[self.target_col].shift(168) - 1
            )
            stat_features.append(f'{self.target_col}_vs_same_hour_last_week')
        
        # Percentage change features
        for period in [1, 24, 168]:
            if len(df_copy) > period:
                feature_name = f'{self.target_col}_pct_change_{period}h'
                df_copy[feature_name] = df_copy[self.target_col].pct_change(period)
                stat_features.append(feature_name)
        
        # Z-score relative to recent period
        if len(df_copy) >= 168:
            rolling_mean = df_copy[self.target_col].rolling(168).mean()
            rolling_std = df_copy[self.target_col].rolling(168).std()
            df_copy[f'{self.target_col}_zscore_168h'] = (df_copy[self.target_col] - rolling_mean) / rolling_std
            stat_features.append(f'{self.target_col}_zscore_168h')
        
        self.feature_names.extend(stat_features)
        print(f"Created {len(stat_features)} statistical features")
        
        return df_copy
    
    def engineer_all_features(self, df: pd.DataFrame,
                             lag_hours: List[int] = [1, 2, 3, 6, 12, 24, 48, 72, 168],
                             rolling_windows: List[int] = [6, 12, 24, 48, 168],
                             rolling_stats: List[str] = ['mean', 'std', 'min', 'max'],
                             ema_alphas: List[float] = [0.1, 0.3, 0.5, 0.7]) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df: Input DataFrame with datetime index
            lag_hours: List of lag periods in hours
            rolling_windows: List of rolling window sizes
            rolling_stats: List of statistical functions for rolling windows
            ema_alphas: List of smoothing parameters for EMA
            
        Returns:
            DataFrame with all engineered features
        """
        print("Starting complete feature engineering pipeline...")
        
        # Reset feature names list
        self.feature_names = []
        
        # Start with input dataframe
        result_df = df.copy()
        
        # 1. Create lag features
        result_df = self.create_lag_features(result_df, lag_hours)
        
        # 2. Create rolling features
        result_df = self.create_rolling_features(result_df, rolling_windows, rolling_stats)
        
        # 3. Create exponential smoothing features
        result_df = self.create_exponential_smoothing_features(result_df, ema_alphas)
        
        # 4. Create calendar features
        result_df = self.create_calendar_features(result_df)
        
        # 5. Create holiday features
        result_df = self.create_holiday_features(result_df)
        
        # 6. Create weather interaction features
        result_df = self.create_weather_interaction_features(result_df)
        
        # 7. Create time-based interactions
        result_df = self.create_time_based_interactions(result_df)
        
        # 8. Create statistical features
        result_df = self.create_statistical_features(result_df)
        
        print(f"Feature engineering completed!")
        print(f"Total features created: {len(self.feature_names)}")
        print(f"Final dataset shape: {result_df.shape}")
        
        return result_df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all created feature names."""
        return self.feature_names.copy()
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Group features by type for analysis.
        
        Returns:
            Dictionary with feature groups
        """
        groups = {
            'lag_features': [f for f in self.feature_names if 'lag_' in f],
            'rolling_features': [f for f in self.feature_names if 'rolling_' in f],
            'ema_features': [f for f in self.feature_names if 'ema_' in f],
            'calendar_features': [f for f in self.feature_names if any(x in f for x in ['is_', 'week_', 'days_'])],
            'weather_features': [f for f in self.feature_names if any(x in f for x in ['temp_', 'heating_', 'cooling_', 'weather_', 'solar_', 'comfort_', 'apparent_'])],
            'interaction_features': [f for f in self.feature_names if 'interaction' in f],
            'statistical_features': [f for f in self.feature_names if any(x in f for x in ['pct_change', 'zscore', 'vs_same'])]
        }
        
        return groups

def main():
    """Example usage of the FeatureEngineer class."""
    # Load processed data (assuming it exists)
    try:
        df = pd.read_csv('data/processed/processed_data.csv', index_col='timestamp', parse_dates=['timestamp'])
        print(f"Loaded data with shape: {df.shape}")
        
        # Initialize feature engineer
        engineer = FeatureEngineer(target_col='kwh')
        
        # Create all features
        engineered_df = engineer.engineer_all_features(df)
        
        # Save engineered features
        engineered_df.to_csv('data/processed/engineered_features.csv')
        
        # Print feature summary
        feature_groups = engineer.get_feature_groups()
        print("\nFeature Groups Summary:")
        for group, features in feature_groups.items():
            print(f"{group}: {len(features)} features")
        
        print(f"\nTotal features: {len(engineer.get_feature_names())}")
        print(f"Final dataset shape: {engineered_df.shape}")
        
    except FileNotFoundError:
        print("Processed data not found. Please run data preprocessing first.")

if __name__ == "__main__":
    main()