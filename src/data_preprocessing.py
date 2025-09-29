"""
Data preprocessing module for Energy Consumption Predictor.
Handles data loading, cleaning, missing value imputation, resampling, and outlier detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Handles all data preprocessing tasks including:
    - Loading electricity consumption and weather data
    - Handling missing values
    - Resampling to consistent frequency
    - Outlier detection and removal
    - Data merging and alignment
    """
    
    def __init__(self, electricity_path: str, weather_path: str):
        """
        Initialize the preprocessor with data file paths.
        
        Args:
            electricity_path: Path to electricity consumption CSV
            weather_path: Path to weather data CSV
        """
        self.electricity_path = electricity_path
        self.weather_path = weather_path
        self.electricity_data = None
        self.weather_data = None
        self.merged_data = None
        self.scaler = StandardScaler()
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load electricity and weather data from CSV files.
        
        Returns:
            Tuple of (electricity_df, weather_df)
        """
        print("Loading electricity consumption data...")
        self.electricity_data = pd.read_csv(self.electricity_path)
        self.electricity_data['timestamp'] = pd.to_datetime(self.electricity_data['timestamp'])
        self.electricity_data.set_index('timestamp', inplace=True)
        self.electricity_data.sort_index(inplace=True)
        
        print("Loading weather data...")
        self.weather_data = pd.read_csv(self.weather_path)
        self.weather_data['timestamp'] = pd.to_datetime(self.weather_data['timestamp'])
        self.weather_data.set_index('timestamp', inplace=True)
        self.weather_data.sort_index(inplace=True)
        
        print(f"Electricity data shape: {self.electricity_data.shape}")
        print(f"Weather data shape: {self.weather_data.shape}")
        print(f"Electricity date range: {self.electricity_data.index.min()} to {self.electricity_data.index.max()}")
        print(f"Weather date range: {self.weather_data.index.min()} to {self.weather_data.index.max()}")
        
        return self.electricity_data, self.weather_data
    
    def handle_missing_values(self, method: str = 'interpolate') -> None:
        """
        Handle missing values in both datasets.
        
        Args:
            method: Method to handle missing values ('interpolate', 'forward_fill', 'backward_fill', 'drop')
        """
        print("Handling missing values...")
        
        # Check for missing values
        elec_missing = self.electricity_data.isnull().sum()
        weather_missing = self.weather_data.isnull().sum()
        
        print(f"Electricity missing values: {elec_missing.sum()}")
        print(f"Weather missing values: {weather_missing.sum()}")
        
        if method == 'interpolate':
            # Linear interpolation for time series data
            self.electricity_data = self.electricity_data.interpolate(method='linear')
            self.weather_data = self.weather_data.interpolate(method='linear')
            
        elif method == 'forward_fill':
            self.electricity_data = self.electricity_data.fillna(method='ffill')
            self.weather_data = self.weather_data.fillna(method='ffill')
            
        elif method == 'backward_fill':
            self.electricity_data = self.electricity_data.fillna(method='bfill')
            self.weather_data = self.weather_data.fillna(method='bfill')
            
        elif method == 'drop':
            self.electricity_data = self.electricity_data.dropna()
            self.weather_data = self.weather_data.dropna()
        
        # Fill any remaining missing values at the edges
        self.electricity_data = self.electricity_data.fillna(method='ffill').fillna(method='bfill')
        self.weather_data = self.weather_data.fillna(method='ffill').fillna(method='bfill')
        
        print("Missing values handled successfully.")
    
    def resample_data(self, frequency: str = 'H') -> None:
        """
        Resample data to ensure consistent frequency.
        
        Args:
            frequency: Pandas frequency string ('H' for hourly, 'D' for daily)
        """
        print(f"Resampling data to {frequency} frequency...")
        
        if frequency == 'H':
            # Hourly resampling - use mean for aggregation
            self.electricity_data = self.electricity_data.resample('H').mean()
            self.weather_data = self.weather_data.resample('H').mean()
            
        elif frequency == 'D':
            # Daily resampling
            self.electricity_data = self.electricity_data.resample('D').agg({
                'kwh': 'sum'  # Sum electricity consumption for daily totals
            })
            self.weather_data = self.weather_data.resample('D').mean()
            
        print(f"Resampled electricity data shape: {self.electricity_data.shape}")
        print(f"Resampled weather data shape: {self.weather_data.shape}")
    
    def detect_outliers(self, method: str = 'iqr', remove: bool = True) -> Dict:
        """
        Detect and optionally remove outliers from electricity consumption data.
        
        Args:
            method: Method for outlier detection ('iqr', 'zscore', 'isolation_forest')
            remove: Whether to remove detected outliers
            
        Returns:
            Dictionary with outlier statistics
        """
        print(f"Detecting outliers using {method} method...")
        
        outlier_stats = {}
        
        if method == 'iqr':
            # Interquartile Range method
            Q1 = self.electricity_data['kwh'].quantile(0.25)
            Q3 = self.electricity_data['kwh'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (self.electricity_data['kwh'] < lower_bound) | (self.electricity_data['kwh'] > upper_bound)
            
        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs((self.electricity_data['kwh'] - self.electricity_data['kwh'].mean()) / 
                             self.electricity_data['kwh'].std())
            outliers = z_scores > 3
            
        elif method == 'isolation_forest':
            # Isolation Forest method
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(self.electricity_data[['kwh']])
            outliers = outlier_labels == -1
        
        outlier_count = outliers.sum()
        outlier_percentage = (outlier_count / len(self.electricity_data)) * 100
        
        outlier_stats = {
            'method': method,
            'outlier_count': outlier_count,
            'outlier_percentage': outlier_percentage,
            'outlier_indices': self.electricity_data.index[outliers].tolist()
        }
        
        print(f"Detected {outlier_count} outliers ({outlier_percentage:.2f}% of data)")
        
        if remove and outlier_count > 0:
            # Remove outliers from electricity data
            self.electricity_data = self.electricity_data[~outliers]
            print(f"Removed {outlier_count} outliers from electricity data")
            
        return outlier_stats
    
    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge electricity and weather datasets on timestamp.
        
        Returns:
            Merged DataFrame
        """
        print("Merging electricity and weather datasets...")
        
        # Merge on timestamp index
        self.merged_data = pd.merge(
            self.electricity_data, 
            self.weather_data,
            left_index=True, 
            right_index=True, 
            how='inner'
        )
        
        print(f"Merged data shape: {self.merged_data.shape}")
        print(f"Date range: {self.merged_data.index.min()} to {self.merged_data.index.max()}")
        
        return self.merged_data
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic time-based features to the dataset.
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            DataFrame with added time features
        """
        print("Adding time-based features...")
        
        df = df.copy()
        
        # Basic time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['day_of_year'] = df.index.dayofyear
        
        # Cyclical encoding for better ML performance
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Weekend flag
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        print(f"Added {len([col for col in df.columns if col not in ['kwh'] + list(self.weather_data.columns)])} time features")
        
        return df
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics of the processed data.
        
        Returns:
            Dictionary with data summary
        """
        if self.merged_data is None:
            raise ValueError("No merged data available. Run merge_datasets() first.")
            
        summary = {
            'total_records': len(self.merged_data),
            'date_range': {
                'start': self.merged_data.index.min(),
                'end': self.merged_data.index.max(),
                'duration_days': (self.merged_data.index.max() - self.merged_data.index.min()).days
            },
            'electricity_stats': {
                'mean_kwh': self.merged_data['kwh'].mean(),
                'median_kwh': self.merged_data['kwh'].median(),
                'std_kwh': self.merged_data['kwh'].std(),
                'min_kwh': self.merged_data['kwh'].min(),
                'max_kwh': self.merged_data['kwh'].max()
            },
            'weather_stats': {}
        }
        
        # Weather statistics
        weather_cols = ['temperature_c', 'humidity_percent', 'wind_speed_kmh', 
                       'precipitation_mm', 'cloud_cover_percent', 'solar_irradiance_wm2']
        
        for col in weather_cols:
            if col in self.merged_data.columns:
                summary['weather_stats'][col] = {
                    'mean': self.merged_data[col].mean(),
                    'std': self.merged_data[col].std(),
                    'min': self.merged_data[col].min(),
                    'max': self.merged_data[col].max()
                }
        
        return summary
    
    def save_processed_data(self, output_path: str) -> None:
        """
        Save processed data to CSV file.
        
        Args:
            output_path: Path to save the processed data
        """
        if self.merged_data is None:
            raise ValueError("No merged data available. Run merge_datasets() first.")
            
        self.merged_data.to_csv(output_path)
        print(f"Processed data saved to {output_path}")
    
    def preprocess_pipeline(self, 
                           missing_method: str = 'interpolate',
                           frequency: str = 'H',
                           outlier_method: str = 'iqr',
                           remove_outliers: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            missing_method: Method to handle missing values
            frequency: Resampling frequency
            outlier_method: Outlier detection method
            remove_outliers: Whether to remove outliers
            
        Returns:
            Fully preprocessed DataFrame
        """
        print("Starting complete preprocessing pipeline...")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Handle missing values
        self.handle_missing_values(method=missing_method)
        
        # Step 3: Resample to consistent frequency
        self.resample_data(frequency=frequency)
        
        # Step 4: Detect and handle outliers
        outlier_stats = self.detect_outliers(method=outlier_method, remove=remove_outliers)
        
        # Step 5: Merge datasets
        merged_df = self.merge_datasets()
        
        # Step 6: Add time features
        processed_df = self.add_time_features(merged_df)
        
        # Step 7: Get summary
        summary = self.get_data_summary()
        
        print("Preprocessing pipeline completed successfully!")
        print(f"Final dataset shape: {processed_df.shape}")
        
        return processed_df

def main():
    """Example usage of the DataPreprocessor class."""
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        electricity_path='data/raw/electricity_consumption.csv',
        weather_path='data/raw/weather_data.csv'
    )
    
    # Run complete preprocessing pipeline
    processed_data = preprocessor.preprocess_pipeline()
    
    # Save processed data
    preprocessor.save_processed_data('data/processed/processed_data.csv')
    
    # Print summary
    summary = preprocessor.get_data_summary()
    print("\nData Summary:")
    print(f"Total records: {summary['total_records']}")
    print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Duration: {summary['date_range']['duration_days']} days")
    print(f"Average consumption: {summary['electricity_stats']['mean_kwh']:.2f} kWh")
    print(f"Peak consumption: {summary['electricity_stats']['max_kwh']:.2f} kWh")

if __name__ == "__main__":
    main()