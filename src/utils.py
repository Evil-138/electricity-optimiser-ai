"""
Utility functions for Energy Consumption Predictor project.
Contains helper functions for configuration loading, logging, data validation, and common operations.
"""

import os
import yaml
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        return get_default_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration if config file is not available.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'DATA_PATHS': {
            'raw_electricity': 'data/raw/electricity_consumption.csv',
            'raw_weather': 'data/raw/weather_data.csv',
            'processed': 'data/processed/processed_data.csv',
            'engineered_features': 'data/processed/engineered_features.csv'
        },
        'MODEL_PATHS': {
            'models_dir': 'models',
            'best_model_path': 'models/best_model.joblib'
        },
        'PREPROCESSING': {
            'missing_method': 'interpolate',
            'frequency': 'H',
            'outlier_method': 'iqr',
            'remove_outliers': True
        },
        'FEATURE_ENGINEERING': {
            'target_col': 'kwh',
            'lag_hours': [1, 2, 3, 6, 12, 24, 48, 72, 168],
            'rolling_windows': [6, 12, 24, 48, 168],
            'rolling_stats': ['mean', 'std', 'min', 'max'],
            'ema_alphas': [0.1, 0.3, 0.5, 0.7]
        },
        'TRAINING': {
            'test_size': 0.2,
            'validation_splits': 3,
            'random_state': 42
        },
        'API': {
            'host': '0.0.0.0',
            'port': 5000,
            'debug': False,
            'max_forecast_hours': 720,
            'default_forecast_hours': 24
        }
    }

def setup_logging(config: Dict[str, Any] = None, log_dir: str = 'logs') -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        config: Configuration dictionary
        log_dir: Directory to save log files
        
    Returns:
        Configured logger
    """
    if config is None:
        config = load_config()
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Get logging configuration
    log_config = config.get('LOGGING', {})
    log_level = getattr(logging, log_config.get('level', 'INFO').upper())
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Set up logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(f'{log_dir}/energy_predictor.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('EnergyPredictor')
    logger.info("Logging initialized successfully")
    
    return logger

def validate_data_file(file_path: str, required_columns: List[str] = None) -> bool:
    """
    Validate that a data file exists and has required columns.
    
    Args:
        file_path: Path to the data file
        required_columns: List of required column names
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    try:
        # Try to read the file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, nrows=1)  # Only read first row to check columns
        elif file_path.endswith(('.json', '.jsonl')):
            with open(file_path, 'r') as f:
                data = json.load(f)
                df = pd.DataFrame([data] if isinstance(data, dict) else data[:1])
        else:
            print(f"Unsupported file format: {file_path}")
            return False
        
        # Check required columns
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                print(f"Missing required columns in {file_path}: {missing_columns}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Error validating file {file_path}: {e}")
        return False

def create_directory_structure(base_path: str = '.') -> None:
    """
    Create the complete directory structure for the project.
    
    Args:
        base_path: Base path for the project
    """
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'src',
        'api',
        'notebooks',
        'config',
        'tests',
        'visualizations',
        'logs'
    ]
    
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def validate_datetime_column(df: pd.DataFrame, datetime_col: str) -> bool:
    """
    Validate that a DataFrame has a proper datetime column.
    
    Args:
        df: DataFrame to validate
        datetime_col: Name of the datetime column
        
    Returns:
        True if valid, False otherwise
    """
    if datetime_col not in df.columns:
        print(f"Datetime column '{datetime_col}' not found")
        return False
    
    try:
        # Try to convert to datetime
        pd.to_datetime(df[datetime_col])
        return True
    except Exception as e:
        print(f"Invalid datetime format in column '{datetime_col}': {e}")
        return False

def check_data_gaps(df: pd.DataFrame, datetime_col: str = 'timestamp', 
                   expected_freq: str = 'H') -> Dict[str, Any]:
    """
    Check for gaps in time series data.
    
    Args:
        df: DataFrame with datetime column
        datetime_col: Name of the datetime column
        expected_freq: Expected frequency of the data
        
    Returns:
        Dictionary with gap analysis results
    """
    if datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.set_index(datetime_col)
    
    # Generate expected time range
    expected_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=expected_freq
    )
    
    # Find missing timestamps
    missing_timestamps = expected_range.difference(df.index)
    
    # Calculate statistics
    total_expected = len(expected_range)
    total_actual = len(df)
    missing_count = len(missing_timestamps)
    completeness_percentage = (total_actual / total_expected) * 100
    
    return {
        'total_expected': total_expected,
        'total_actual': total_actual,
        'missing_count': missing_count,
        'completeness_percentage': completeness_percentage,
        'missing_timestamps': missing_timestamps.tolist()[:10],  # First 10 missing
        'largest_gap': max([(missing_timestamps[i+1] - missing_timestamps[i]).total_seconds() / 3600 
                           for i in range(len(missing_timestamps)-1)], default=0) if len(missing_timestamps) > 1 else 0
    }

def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging and monitoring.
    
    Returns:
        Dictionary with system information
    """
    import psutil
    import platform
    
    # Memory information
    memory = psutil.virtual_memory()
    
    # Disk information
    disk = psutil.disk_usage('.')
    
    # CPU information
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': cpu_count,
        'cpu_usage_percent': cpu_percent,
        'memory_total': format_bytes(memory.total),
        'memory_available': format_bytes(memory.available),
        'memory_usage_percent': memory.percent,
        'disk_total': format_bytes(disk.total),
        'disk_free': format_bytes(disk.free),
        'disk_usage_percent': (disk.used / disk.total) * 100
    }

def calculate_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive forecast evaluation metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary with metric values
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    
    # Handle NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {metric: np.nan for metric in ['mae', 'rmse', 'mape', 'r2', 'mase', 'smape']}
    
    # Basic metrics
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean) * 100
    
    # R-squared
    ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
    ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Mean Absolute Scaled Error (MASE)
    # Uses naive forecast (previous value) as baseline
    if len(y_true_clean) > 1:
        naive_mae = np.mean(np.abs(y_true_clean[1:] - y_true_clean[:-1]))
        mase = mae / naive_mae if naive_mae != 0 else np.nan
    else:
        mase = np.nan
    
    # Symmetric Mean Absolute Percentage Error (SMAPE)
    smape = np.mean(200 * np.abs(y_true_clean - y_pred_clean) / 
                    (np.abs(y_true_clean) + np.abs(y_pred_clean))) if len(y_true_clean) > 0 else np.nan
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'mase': mase,
        'smape': smape
    }

def generate_time_features(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Generate comprehensive time-based features from timestamps.
    
    Args:
        timestamps: DatetimeIndex
        
    Returns:
        DataFrame with time features
    """
    df = pd.DataFrame(index=timestamps)
    
    # Basic time features
    df['hour'] = timestamps.hour
    df['day_of_week'] = timestamps.dayofweek
    df['day_of_month'] = timestamps.day
    df['month'] = timestamps.month
    df['quarter'] = timestamps.quarter
    df['year'] = timestamps.year
    df['day_of_year'] = timestamps.dayofyear
    df['week_of_year'] = timestamps.isocalendar().week
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Binary features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_business_day'] = (df['day_of_week'] < 5).astype(int)
    df['is_month_start'] = timestamps.is_month_start.astype(int)
    df['is_month_end'] = timestamps.is_month_end.astype(int)
    df['is_quarter_start'] = timestamps.is_quarter_start.astype(int)
    df['is_quarter_end'] = timestamps.is_quarter_end.astype(int)
    df['is_year_start'] = timestamps.is_year_start.astype(int)
    df['is_year_end'] = timestamps.is_year_end.astype(int)
    
    return df

def save_results_summary(results: Dict[str, Any], output_path: str) -> None:
    """
    Save comprehensive results summary to JSON file.
    
    Args:
        results: Results dictionary from training pipeline
        output_path: Path to save the summary
    """
    # Prepare summary data (convert numpy arrays to lists for JSON serialization)
    summary = {}
    
    for key, value in results.items():
        if isinstance(value, dict):
            summary[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray):
                    summary[key][subkey] = subvalue.tolist()
                elif isinstance(subvalue, (np.int32, np.int64, np.float32, np.float64)):
                    summary[key][subkey] = float(subvalue)
                elif isinstance(subvalue, pd.Timestamp):
                    summary[key][subkey] = subvalue.isoformat()
                else:
                    summary[key][subkey] = subvalue
        elif isinstance(value, (list, tuple)):
            summary[key] = list(value)
        elif isinstance(value, np.ndarray):
            summary[key] = value.tolist()
        elif isinstance(value, (np.int32, np.int64, np.float32, np.float64)):
            summary[key] = float(value)
        elif isinstance(value, pd.Timestamp):
            summary[key] = value.isoformat()
        else:
            summary[key] = value
    
    # Add metadata
    summary['summary_metadata'] = {
        'created_at': datetime.now().isoformat(),
        'summary_version': '1.0',
        'total_metrics_count': len(summary.get('metrics', {})),
        'total_models_count': len(summary.get('models', {}))
    }
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Results summary saved to {output_path}")

class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        print(f"Starting {self.description}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = end_time - self.start_time
        print(f"{self.description} completed in {duration.total_seconds():.2f} seconds")

def main():
    """Example usage of utility functions."""
    print("Energy Consumption Predictor - Utilities")
    print("="*50)
    
    # Load configuration
    config = load_config()
    print(f"Configuration loaded: {len(config)} sections")
    
    # Set up logging
    logger = setup_logging(config)
    logger.info("Utilities module loaded successfully")
    
    # Create directory structure
    create_directory_structure()
    
    # Get system information
    sys_info = get_system_info()
    print(f"\nSystem Information:")
    for key, value in sys_info.items():
        print(f"  {key}: {value}")
    
    # Example timing
    with Timer("Example operation"):
        import time
        time.sleep(1)  # Simulate some work

if __name__ == "__main__":
    main()