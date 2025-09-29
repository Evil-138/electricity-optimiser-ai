"""
Modeling pipeline for Energy Consumption Predictor.
Includes baseline models, XGBoost/LightGBM, and optional SARIMAX/Prophet models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Try to import optional packages
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: Statsmodels not available. Install with: pip install statsmodels")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available. Install with: pip install prophet")

class BaselineModels:
    """Collection of baseline models for energy consumption prediction."""
    
    @staticmethod
    def naive_persistence(y_true: pd.Series, lag_hours: int = 24) -> np.ndarray:
        """
        Naive persistence baseline - use value from lag_hours ago.
        
        Args:
            y_true: Target series
            lag_hours: Number of hours to lag
            
        Returns:
            Predictions array
        """
        predictions = y_true.shift(lag_hours).fillna(y_true.mean())
        return predictions.values
    
    @staticmethod
    def seasonal_naive(y_true: pd.Series, season_length: int = 168) -> np.ndarray:
        """
        Seasonal naive baseline - use same hour from previous week.
        
        Args:
            y_true: Target series
            season_length: Length of season in hours (168 = 1 week)
            
        Returns:
            Predictions array
        """
        predictions = y_true.shift(season_length).fillna(y_true.mean())
        return predictions.values
    
    @staticmethod
    def moving_average(y_true: pd.Series, window: int = 24) -> np.ndarray:
        """
        Moving average baseline.
        
        Args:
            y_true: Target series
            window: Moving average window size
            
        Returns:
            Predictions array
        """
        predictions = y_true.rolling(window=window).mean().fillna(y_true.mean())
        return predictions.values

class CustomRegressor(BaseEstimator, RegressorMixin):
    """
    Wrapper class for various regression models to provide consistent interface.
    """
    
    def __init__(self, model_type: str = 'xgboost', **kwargs):
        """
        Initialize the regressor.
        
        Args:
            model_type: Type of model ('xgboost', 'lightgbm', 'random_forest')
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the model."""
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1
            }
            params = {**default_params, **self.kwargs}
            self.model = xgb.XGBRegressor(**params)
            
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            params = {**default_params, **self.kwargs}
            self.model = lgb.LGBMRegressor(**params)
            
        elif self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
            params = {**default_params, **self.kwargs}
            self.model = RandomForestRegressor(**params)
            
        else:
            raise ValueError(f"Model type '{self.model_type}' not supported or required package not installed")
        
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None

class SARIMAXModel:
    """SARIMAX model wrapper."""
    
    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,24), **kwargs):
        """Initialize SARIMAX model."""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Statsmodels not available. Install with: pip install statsmodels")
            
        self.order = order
        self.seasonal_order = seasonal_order
        self.kwargs = kwargs
        self.model = None
        self.fitted_model = None
    
    def fit(self, y: pd.Series):
        """Fit SARIMAX model."""
        try:
            self.model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order, **self.kwargs)
            self.fitted_model = self.model.fit(disp=False)
        except Exception as e:
            print(f"SARIMAX fitting failed: {e}")
            # Fallback to simpler ARIMA
            try:
                self.model = SARIMAX(y, order=(1,1,1), seasonal_order=(0,0,0,0))
                self.fitted_model = self.model.fit(disp=False)
            except Exception as e2:
                print(f"Fallback ARIMA also failed: {e2}")
                raise e2
        return self
    
    def predict(self, steps: int = 24) -> np.ndarray:
        """Make predictions."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast.values

class ProphetModel:
    """Prophet model wrapper."""
    
    def __init__(self, **kwargs):
        """Initialize Prophet model."""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available. Install with: pip install prophet")
            
        self.kwargs = kwargs
        self.model = None
    
    def fit(self, df: pd.DataFrame, target_col: str = 'kwh'):
        """Fit Prophet model."""
        # Prepare data for Prophet
        prophet_df = pd.DataFrame({
            'ds': df.index,
            'y': df[target_col]
        })
        
        default_params = {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': True,
            'seasonality_mode': 'multiplicative'
        }
        params = {**default_params, **self.kwargs}
        
        self.model = Prophet(**params)
        self.model.fit(prophet_df)
        return self
    
    def predict(self, periods: int = 24, freq: str = 'H') -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
            
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        
        # Return only the forecasted values (last 'periods' values)
        return forecast['yhat'].iloc[-periods:].values

class ModelEvaluator:
    """Evaluate model performance with various metrics."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metric names and values
        """
        # Handle NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'r2': np.nan}
        
        metrics = {
            'mae': mean_absolute_error(y_true_clean, y_pred_clean),
            'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
            'mape': mean_absolute_percentage_error(y_true_clean, y_pred_clean) * 100,
        }
        
        # R-squared
        ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
        ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return metrics
    
    @staticmethod
    def cross_validate_timeseries(model, X: pd.DataFrame, y: pd.Series, 
                                 n_splits: int = 5) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Target
            n_splits: Number of cross-validation splits
            
        Returns:
            Dictionary with metric lists for each fold
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = {'mae': [], 'rmse': [], 'mape': [], 'r2': []}
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = ModelEvaluator.calculate_metrics(y_test.values, y_pred)
            
            for metric, value in metrics.items():
                cv_results[metric].append(value)
        
        return cv_results

class EnergyPredictionPipeline:
    """Complete pipeline for energy consumption prediction."""
    
    def __init__(self, target_col: str = 'kwh'):
        """Initialize the pipeline."""
        self.target_col = target_col
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.feature_names = None
        
    def prepare_features(self, df: pd.DataFrame, 
                        exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling.
        
        Args:
            df: Input DataFrame with features and target
            exclude_cols: Columns to exclude from features
            
        Returns:
            Tuple of (features, target)
        """
        if exclude_cols is None:
            exclude_cols = [self.target_col]
        else:
            exclude_cols = list(exclude_cols) + [self.target_col]
        
        # Remove any columns that shouldn't be features
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[self.target_col]
        
        # Remove rows with NaN in target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Remove rows with too many NaN features (more than 50%)
        nan_threshold = len(X.columns) * 0.5
        mask = X.isna().sum(axis=1) < nan_threshold
        X = X[mask]
        y = y[mask]
        
        # Fill remaining NaN values
        X = X.fillna(X.median())
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train_baseline_models(self, y: pd.Series) -> Dict[str, np.ndarray]:
        """Train baseline models."""
        print("Training baseline models...")
        
        baselines = {}
        
        # Naive persistence (24 hours)
        baselines['naive_24h'] = BaselineModels.naive_persistence(y, lag_hours=24)
        
        # Seasonal naive (1 week)
        baselines['seasonal_naive'] = BaselineModels.seasonal_naive(y, season_length=168)
        
        # Moving average
        baselines['moving_avg_24h'] = BaselineModels.moving_average(y, window=24)
        
        return baselines
    
    def train_ml_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train machine learning models."""
        print("Training machine learning models...")
        
        models = {}
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            print("Training XGBoost model...")
            xgb_params = {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            models['xgboost'] = CustomRegressor('xgboost', **xgb_params)
            models['xgboost'].fit(X, y)
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            print("Training LightGBM model...")
            lgb_params = {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            models['lightgbm'] = CustomRegressor('lightgbm', **lgb_params)
            models['lightgbm'].fit(X, y)
        
        # Random Forest (always available)
        print("Training Random Forest model...")
        rf_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        models['random_forest'] = CustomRegressor('random_forest', **rf_params)
        models['random_forest'].fit(X, y)
        
        return models
    
    def train_time_series_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train time series models."""
        print("Training time series models...")
        
        models = {}
        
        # SARIMAX
        if STATSMODELS_AVAILABLE:
            print("Training SARIMAX model...")
            try:
                models['sarimax'] = SARIMAXModel(order=(1,1,1), seasonal_order=(1,1,1,24))
                models['sarimax'].fit(df[self.target_col])
            except Exception as e:
                print(f"SARIMAX training failed: {e}")
        
        # Prophet
        if PROPHET_AVAILABLE:
            print("Training Prophet model...")
            try:
                models['prophet'] = ProphetModel()
                models['prophet'].fit(df, self.target_col)
            except Exception as e:
                print(f"Prophet training failed: {e}")
        
        return models
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series,
                       baseline_predictions: Dict[str, np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models."""
        print("Evaluating models...")
        
        results = {}
        
        # Evaluate baselines
        if baseline_predictions:
            for name, pred in baseline_predictions.items():
                # Align predictions with test set
                if len(pred) == len(y_test):
                    results[name] = ModelEvaluator.calculate_metrics(y_test.values, pred)
                else:
                    print(f"Skipping baseline {name} - prediction length mismatch")
        
        # Evaluate ML models
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                try:
                    pred = model.predict(X_test)
                    results[name] = ModelEvaluator.calculate_metrics(y_test.values, pred)
                except Exception as e:
                    print(f"Error evaluating model {name}: {e}")
                    results[name] = {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'r2': np.nan}
        
        return results
    
    def get_feature_importance(self, model_name: str = None) -> Optional[Dict[str, float]]:
        """Get feature importance from a model."""
        if model_name and model_name in self.models:
            model = self.models[model_name]
            importance = model.get_feature_importance()
            if importance is not None and self.feature_names:
                return dict(zip(self.feature_names, importance))
        return None
    
    def train_pipeline(self, df: pd.DataFrame, 
                      test_size: float = 0.2,
                      exclude_cols: List[str] = None) -> Dict[str, Any]:
        """
        Run complete training pipeline.
        
        Args:
            df: Input DataFrame with features and target
            test_size: Proportion of data for testing
            exclude_cols: Columns to exclude from features
            
        Returns:
            Dictionary with results
        """
        print("Starting complete training pipeline...")
        
        # Prepare features
        X, y = self.prepare_features(df, exclude_cols)
        
        # Time-based split (no shuffle for time series)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Train models
        baseline_predictions = self.train_baseline_models(y)
        self.models.update(self.train_ml_models(X_train, y_train))
        self.models.update(self.train_time_series_models(df.iloc[:split_idx]))
        
        # Evaluate models
        # Get baseline predictions for test set
        test_baselines = {}
        for name, full_pred in baseline_predictions.items():
            test_baselines[name] = full_pred[split_idx:]
        
        self.metrics = self.evaluate_models(X_test, y_test, test_baselines)
        
        # Store predictions for analysis
        self.predictions = {'y_test': y_test}
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                try:
                    pred = model.predict(X_test)
                    self.predictions[name] = pred
                except:
                    pass
        
        results = {
            'models': self.models,
            'metrics': self.metrics,
            'predictions': self.predictions,
            'feature_names': self.feature_names,
            'test_indices': y_test.index
        }
        
        print("Training pipeline completed!")
        self.print_results_summary()
        
        return results
    
    def print_results_summary(self):
        """Print summary of model performance."""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        
        if not self.metrics:
            print("No metrics available.")
            return
        
        # Sort models by RMSE
        sorted_models = sorted(self.metrics.items(), key=lambda x: x[1].get('rmse', float('inf')))
        
        print(f"{'Model':<15} {'MAE':<8} {'RMSE':<8} {'MAPE (%)':<10} {'R²':<8}")
        print("-" * 60)
        
        for name, metrics in sorted_models:
            mae = metrics.get('mae', np.nan)
            rmse = metrics.get('rmse', np.nan)
            mape = metrics.get('mape', np.nan)
            r2 = metrics.get('r2', np.nan)
            
            print(f"{name:<15} {mae:<8.3f} {rmse:<8.3f} {mape:<10.2f} {r2:<8.3f}")
        
        print("\nBest model (by RMSE): " + sorted_models[0][0])

def main():
    """Example usage of the modeling pipeline."""
    try:
        # Load engineered features
        df = pd.read_csv('data/processed/engineered_features.csv', 
                        index_col='timestamp', parse_dates=['timestamp'])
        print(f"Loaded data with shape: {df.shape}")
        
        # Initialize pipeline
        pipeline = EnergyPredictionPipeline(target_col='kwh')
        
        # Train all models
        results = pipeline.train_pipeline(df, test_size=0.2)
        
        # Get feature importance for best model
        best_model = min(pipeline.metrics.items(), key=lambda x: x[1].get('rmse', float('inf')))[0]
        importance = pipeline.get_feature_importance(best_model)
        
        if importance:
            print(f"\nTop 10 features for {best_model}:")
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, imp in sorted_features:
                print(f"{feature}: {imp:.4f}")
        
    except FileNotFoundError:
        print("Engineered features not found. Please run feature engineering first.")

if __name__ == "__main__":
    main()