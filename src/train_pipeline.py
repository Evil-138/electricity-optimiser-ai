"""
Training pipeline for Energy Consumption Predictor.
Handles model training, validation, and persistence using joblib.
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.modeling import EnergyPredictionPipeline, ModelEvaluator

class ModelTrainer:
    """
    Complete training pipeline that orchestrates data preprocessing,
    feature engineering, model training, and model persistence.
    """
    
    def __init__(self, 
                 electricity_path: str,
                 weather_path: str,
                 models_dir: str = 'models',
                 processed_data_dir: str = 'data/processed'):
        """
        Initialize the trainer.
        
        Args:
            electricity_path: Path to electricity consumption data
            weather_path: Path to weather data
            models_dir: Directory to save trained models
            processed_data_dir: Directory to save processed data
        """
        self.electricity_path = electricity_path
        self.weather_path = weather_path
        self.models_dir = models_dir
        self.processed_data_dir = processed_data_dir
        
        # Create directories if they don't exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(processed_data_dir, exist_ok=True)
        
        # Initialize components
        self.preprocessor = None
        self.feature_engineer = None
        self.pipeline = None
        
        # Store results
        self.trained_models = {}
        self.model_metadata = {}
        self.feature_names = []
        
    def preprocess_data(self, 
                       missing_method: str = 'interpolate',
                       frequency: str = 'H',
                       outlier_method: str = 'iqr',
                       remove_outliers: bool = True,
                       save_processed: bool = True) -> pd.DataFrame:
        """
        Run data preprocessing pipeline.
        
        Args:
            missing_method: Method to handle missing values
            frequency: Resampling frequency
            outlier_method: Outlier detection method
            remove_outliers: Whether to remove outliers
            save_processed: Whether to save processed data
            
        Returns:
            Preprocessed DataFrame
        """
        print("Starting data preprocessing...")
        
        self.preprocessor = DataPreprocessor(self.electricity_path, self.weather_path)
        
        processed_data = self.preprocessor.preprocess_pipeline(
            missing_method=missing_method,
            frequency=frequency,
            outlier_method=outlier_method,
            remove_outliers=remove_outliers
        )
        
        if save_processed:
            processed_path = os.path.join(self.processed_data_dir, 'processed_data.csv')
            self.preprocessor.save_processed_data(processed_path)
            
        return processed_data
    
    def engineer_features(self, 
                         df: pd.DataFrame,
                         lag_hours: List[int] = [1, 2, 3, 6, 12, 24, 48, 72, 168],
                         rolling_windows: List[int] = [6, 12, 24, 48, 168],
                         rolling_stats: List[str] = ['mean', 'std', 'min', 'max'],
                         ema_alphas: List[float] = [0.1, 0.3, 0.5, 0.7],
                         save_features: bool = True) -> pd.DataFrame:
        """
        Run feature engineering pipeline.
        
        Args:
            df: Input DataFrame from preprocessing
            lag_hours: List of lag periods in hours
            rolling_windows: List of rolling window sizes
            rolling_stats: List of statistical functions for rolling windows
            ema_alphas: List of smoothing parameters for EMA
            save_features: Whether to save engineered features
            
        Returns:
            DataFrame with engineered features
        """
        print("Starting feature engineering...")
        
        self.feature_engineer = FeatureEngineer(target_col='kwh')
        
        engineered_data = self.feature_engineer.engineer_all_features(
            df,
            lag_hours=lag_hours,
            rolling_windows=rolling_windows,
            rolling_stats=rolling_stats,
            ema_alphas=ema_alphas
        )
        
        self.feature_names = self.feature_engineer.get_feature_names()
        
        if save_features:
            features_path = os.path.join(self.processed_data_dir, 'engineered_features.csv')
            engineered_data.to_csv(features_path)
            print(f"Engineered features saved to {features_path}")
        
        return engineered_data
    
    def train_models(self, 
                    df: pd.DataFrame,
                    test_size: float = 0.2,
                    validation_splits: int = 3,
                    exclude_cols: List[str] = None) -> Dict[str, Any]:
        """
        Train all models with cross-validation.
        
        Args:
            df: DataFrame with engineered features
            test_size: Proportion of data for final testing
            validation_splits: Number of time series cross-validation splits
            exclude_cols: Columns to exclude from features
            
        Returns:
            Dictionary with training results
        """
        print("Starting model training...")
        
        self.pipeline = EnergyPredictionPipeline(target_col='kwh')
        
        # Train models with the pipeline
        results = self.pipeline.train_pipeline(df, test_size=test_size, exclude_cols=exclude_cols)
        
        # Perform cross-validation on training data
        print("Performing cross-validation...")
        X, y = self.pipeline.prepare_features(df, exclude_cols)
        
        # Use only training portion for CV
        split_idx = int(len(X) * (1 - test_size))
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        
        cv_results = {}
        for model_name, model in self.pipeline.models.items():
            if hasattr(model, 'fit') and hasattr(model, 'predict'):
                print(f"Cross-validating {model_name}...")
                try:
                    cv_result = ModelEvaluator.cross_validate_timeseries(
                        model, X_train, y_train, n_splits=validation_splits
                    )
                    cv_results[model_name] = cv_result
                except Exception as e:
                    print(f"Cross-validation failed for {model_name}: {e}")
        
        results['cross_validation'] = cv_results
        
        # Store models for saving
        self.trained_models = self.pipeline.models
        
        return results
    
    def save_models(self, model_version: str = None) -> Dict[str, str]:
        """
        Save trained models and metadata using joblib.
        
        Args:
            model_version: Version string for the models
            
        Returns:
            Dictionary with saved model paths
        """
        print("Saving trained models...")
        
        if model_version is None:
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_paths = {}
        
        for model_name, model in self.trained_models.items():
            # Create model filename
            model_filename = f"{model_name}_{model_version}.joblib"
            model_path = os.path.join(self.models_dir, model_filename)
            
            try:
                # Save model
                joblib.dump(model, model_path)
                saved_paths[model_name] = model_path
                print(f"Saved {model_name} to {model_path}")
                
                # Save model metadata
                metadata = {
                    'model_name': model_name,
                    'version': model_version,
                    'timestamp': datetime.now().isoformat(),
                    'model_path': model_path,
                    'feature_names': self.feature_names,
                    'metrics': self.pipeline.metrics.get(model_name, {}),
                    'model_type': type(model).__name__
                }
                
                metadata_filename = f"{model_name}_{model_version}_metadata.joblib"
                metadata_path = os.path.join(self.models_dir, metadata_filename)
                joblib.dump(metadata, metadata_path)
                
                self.model_metadata[model_name] = metadata
                
            except Exception as e:
                print(f"Error saving model {model_name}: {e}")
        
        # Save feature names separately
        feature_names_path = os.path.join(self.models_dir, f"feature_names_{model_version}.joblib")
        joblib.dump(self.feature_names, feature_names_path)
        
        # Save preprocessing and feature engineering pipelines
        if self.preprocessor:
            preprocessor_path = os.path.join(self.models_dir, f"preprocessor_{model_version}.joblib")
            joblib.dump(self.preprocessor, preprocessor_path)
            
        if self.feature_engineer:
            feature_engineer_path = os.path.join(self.models_dir, f"feature_engineer_{model_version}.joblib")
            joblib.dump(self.feature_engineer, feature_engineer_path)
        
        print(f"Saved {len(saved_paths)} models with version {model_version}")
        
        return saved_paths
    
    def get_best_model(self, metric: str = 'rmse') -> Tuple[str, Any]:
        """
        Get the best performing model based on a metric.
        
        Args:
            metric: Metric to use for comparison ('mae', 'rmse', 'mape', 'r2')
            
        Returns:
            Tuple of (model_name, model_object)
        """
        if not self.pipeline or not self.pipeline.metrics:
            raise ValueError("Models must be trained first")
        
        if metric in ['mae', 'rmse', 'mape']:
            # Lower is better
            best_model_name = min(
                self.pipeline.metrics.items(), 
                key=lambda x: x[1].get(metric, float('inf'))
            )[0]
        else:  # r2
            # Higher is better
            best_model_name = max(
                self.pipeline.metrics.items(), 
                key=lambda x: x[1].get(metric, -float('inf'))
            )[0]
        
        best_model = self.trained_models.get(best_model_name)
        
        return best_model_name, best_model
    
    def run_complete_pipeline(self, 
                             config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline from data to saved models.
        
        Args:
            config: Configuration dictionary with pipeline parameters
            
        Returns:
            Dictionary with complete results
        """
        if config is None:
            config = self.get_default_config()
        
        print("="*60)
        print("STARTING COMPLETE ENERGY PREDICTION TRAINING PIPELINE")
        print("="*60)
        
        results = {}
        
        try:
            # Step 1: Preprocess data
            processed_data = self.preprocess_data(**config.get('preprocessing', {}))
            results['preprocessing'] = {
                'shape': processed_data.shape,
                'date_range': (processed_data.index.min(), processed_data.index.max())
            }
            
            # Step 2: Engineer features
            engineered_data = self.engineer_features(processed_data, **config.get('feature_engineering', {}))
            results['feature_engineering'] = {
                'shape': engineered_data.shape,
                'num_features': len(self.feature_names)
            }
            
            # Step 3: Train models
            training_results = self.train_models(engineered_data, **config.get('training', {}))
            results['training'] = training_results
            
            # Step 4: Save models
            saved_paths = self.save_models(config.get('model_version'))
            results['saved_models'] = saved_paths
            
            # Step 5: Get best model
            best_model_name, best_model = self.get_best_model(config.get('best_model_metric', 'rmse'))
            results['best_model'] = {
                'name': best_model_name,
                'metrics': self.pipeline.metrics.get(best_model_name, {})
            }
            
            print("\n" + "="*60)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Best model: {best_model_name}")
            print(f"Models saved to: {self.models_dir}")
            print(f"Processed data saved to: {self.processed_data_dir}")
            
            return results
            
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            raise e
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration for the pipeline."""
        return {
            'preprocessing': {
                'missing_method': 'interpolate',
                'frequency': 'H',
                'outlier_method': 'iqr',
                'remove_outliers': True,
                'save_processed': True
            },
            'feature_engineering': {
                'lag_hours': [1, 2, 3, 6, 12, 24, 48, 72, 168],
                'rolling_windows': [6, 12, 24, 48, 168],
                'rolling_stats': ['mean', 'std', 'min', 'max'],
                'ema_alphas': [0.1, 0.3, 0.5, 0.7],
                'save_features': True
            },
            'training': {
                'test_size': 0.2,
                'validation_splits': 3,
                'exclude_cols': None
            },
            'best_model_metric': 'rmse',
            'model_version': None
        }

class ModelLoader:
    """Load trained models and make predictions."""
    
    def __init__(self, models_dir: str = 'models'):
        """Initialize the loader."""
        self.models_dir = models_dir
        self.loaded_models = {}
        self.metadata = {}
        
    def list_available_models(self) -> List[str]:
        """List all available model files."""
        model_files = []
        
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.joblib') and not filename.endswith('_metadata.joblib'):
                model_files.append(filename)
                
        return model_files
    
    def load_model(self, model_path: str) -> Tuple[Any, Dict]:
        """
        Load a specific model and its metadata.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Tuple of (model, metadata)
        """
        # Load model
        model = joblib.load(model_path)
        
        # Try to load metadata
        metadata_path = model_path.replace('.joblib', '_metadata.joblib')
        metadata = {}
        
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
        
        return model, metadata
    
    def load_best_model(self, version: str = None, metric: str = 'rmse') -> Tuple[str, Any, Dict]:
        """
        Load the best model from a specific version.
        
        Args:
            version: Model version to load from
            metric: Metric to determine best model
            
        Returns:
            Tuple of (model_name, model, metadata)
        """
        available_models = self.list_available_models()
        
        if version:
            # Filter by version
            version_models = [m for m in available_models if version in m]
        else:
            version_models = available_models
        
        if not version_models:
            raise ValueError(f"No models found for version {version}")
        
        best_model_path = None
        best_score = float('inf') if metric in ['mae', 'rmse', 'mape'] else -float('inf')
        
        for model_file in version_models:
            model_path = os.path.join(self.models_dir, model_file)
            _, metadata = self.load_model(model_path)
            
            if 'metrics' in metadata and metric in metadata['metrics']:
                score = metadata['metrics'][metric]
                
                if metric in ['mae', 'rmse', 'mape'] and score < best_score:
                    best_score = score
                    best_model_path = model_path
                elif metric == 'r2' and score > best_score:
                    best_score = score
                    best_model_path = model_path
        
        if best_model_path is None:
            raise ValueError(f"No valid models found with metric {metric}")
        
        model, metadata = self.load_model(best_model_path)
        model_name = metadata.get('model_name', 'unknown')
        
        return model_name, model, metadata

def main():
    """Example usage of the training pipeline."""
    
    # Initialize trainer
    trainer = ModelTrainer(
        electricity_path='data/raw/electricity_consumption.csv',
        weather_path='data/raw/weather_data.csv',
        models_dir='models',
        processed_data_dir='data/processed'
    )
    
    # Get default configuration
    config = trainer.get_default_config()
    
    # Optionally modify configuration
    config['training']['test_size'] = 0.3  # Use 30% for testing
    config['preprocessing']['remove_outliers'] = False  # Keep outliers
    
    # Run complete pipeline
    results = trainer.run_complete_pipeline(config)
    
    # Print summary
    print("\nPipeline Results Summary:")
    print(f"Processed data shape: {results['preprocessing']['shape']}")
    print(f"Engineered features: {results['feature_engineering']['num_features']}")
    print(f"Models trained: {len(results['saved_models'])}")
    print(f"Best model: {results['best_model']['name']}")
    
    # Example of loading a model
    print("\nTesting model loading...")
    loader = ModelLoader('models')
    available_models = loader.list_available_models()
    print(f"Available models: {len(available_models)}")
    
    if available_models:
        try:
            model_name, model, metadata = loader.load_best_model()
            print(f"Loaded best model: {model_name}")
            print(f"Model metrics: {metadata.get('metrics', {})}")
        except Exception as e:
            print(f"Error loading model: {e}")

if __name__ == "__main__":
    main()