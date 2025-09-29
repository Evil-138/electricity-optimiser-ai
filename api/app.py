"""
Flask API for Energy Consumption Predictor.
Provides REST endpoints for energy consumption forecasting.
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import the custom modules
try:
    from src.train_pipeline import ModelLoader
    from src.feature_engineering import FeatureEngineer
    from src.data_preprocessing import DataPreprocessor
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    print("Warning: Custom modules not found. Some features may be limited.")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnergyPredictionAPI:
    """Main API class for energy consumption prediction."""
    
    def __init__(self, models_dir: str = 'models'):
        """Initialize the API."""
        self.models_dir = models_dir
        self.model_loader = None
        self.loaded_models = {}
        self.feature_engineer = None
        self.preprocessor = None
        
        if MODULES_AVAILABLE:
            self.model_loader = ModelLoader(models_dir)
        
        self.load_models()
    
    def load_models(self):
        """Load available models."""
        if not self.model_loader:
            logger.warning("Model loader not available")
            return
            
        try:
            # List available models
            available_models = self.model_loader.list_available_models()
            logger.info(f"Found {len(available_models)} available models")
            
            # Try to load the best model
            if available_models:
                model_name, model, metadata = self.model_loader.load_best_model()
                self.loaded_models['best'] = {
                    'name': model_name,
                    'model': model,
                    'metadata': metadata
                }
                logger.info(f"Loaded best model: {model_name}")
                
                # Load feature names
                feature_names = metadata.get('feature_names', [])
                if feature_names:
                    self.feature_names = feature_names
                    logger.info(f"Loaded {len(feature_names)} feature names")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def prepare_features_for_prediction(self, input_data: Dict) -> pd.DataFrame:
        """
        Prepare features from input data for prediction.
        
        Args:
            input_data: Dictionary with recent history and weather forecast
            
        Returns:
            DataFrame with prepared features
        """
        # Extract historical electricity consumption
        if 'recent_consumption' not in input_data:
            raise ValueError("recent_consumption is required")
        
        recent_consumption = input_data['recent_consumption']
        
        # Extract weather forecast
        weather_forecast = input_data.get('weather_forecast', {})
        
        # Create DataFrame from recent consumption
        if isinstance(recent_consumption, list):
            # Convert to time series (assuming hourly data)
            timestamps = pd.date_range(
                end=datetime.now(), 
                periods=len(recent_consumption), 
                freq='H'
            )
            consumption_df = pd.DataFrame({
                'timestamp': timestamps,
                'kwh': recent_consumption
            })
        else:
            # Assume dictionary with timestamps
            consumption_df = pd.DataFrame(recent_consumption)
            consumption_df['timestamp'] = pd.to_datetime(consumption_df['timestamp'])
        
        consumption_df.set_index('timestamp', inplace=True)
        
        # Add weather data if provided
        if weather_forecast:
            weather_df = pd.DataFrame(weather_forecast)
            if 'timestamp' in weather_df.columns:
                weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
                weather_df.set_index('timestamp', inplace=True)
                
                # Merge consumption and weather data
                merged_df = pd.merge(consumption_df, weather_df, 
                                   left_index=True, right_index=True, how='left')
            else:
                # Use latest weather for all timestamps
                for col, value in weather_forecast.items():
                    consumption_df[col] = value
                merged_df = consumption_df
        else:
            # Create dummy weather features with reasonable defaults
            merged_df = consumption_df.copy()
            merged_df['temperature_c'] = 20.0  # Default temperature
            merged_df['humidity_percent'] = 60.0
            merged_df['wind_speed_kmh'] = 10.0
            merged_df['precipitation_mm'] = 0.0
            merged_df['cloud_cover_percent'] = 50.0
            merged_df['solar_irradiance_wm2'] = 300.0
        
        # Create basic time features (simplified version)
        merged_df['hour'] = merged_df.index.hour
        merged_df['day_of_week'] = merged_df.index.dayofweek
        merged_df['month'] = merged_df.index.month
        merged_df['is_weekend'] = (merged_df.index.dayofweek >= 5).astype(int)
        
        # Create basic lag features (if we have enough data)
        for lag in [1, 2, 24]:
            if len(merged_df) > lag:
                merged_df[f'kwh_lag_{lag}h'] = merged_df['kwh'].shift(lag)
        
        # Create basic rolling features
        for window in [6, 24]:
            if len(merged_df) > window:
                merged_df[f'kwh_rolling_{window}h_mean'] = merged_df['kwh'].rolling(window).mean()
        
        # Fill NaN values
        merged_df = merged_df.fillna(merged_df.median())
        
        return merged_df
    
    def predict(self, input_data: Dict, forecast_hours: int = 24) -> Dict:
        """
        Make energy consumption predictions.
        
        Args:
            input_data: Dictionary with recent history and weather forecast
            forecast_hours: Number of hours to forecast
            
        Returns:
            Dictionary with predictions and metadata
        """
        if not self.loaded_models:
            raise ValueError("No models loaded")
        
        try:
            # Prepare features
            features_df = self.prepare_features_for_prediction(input_data)
            
            # Get the best model
            best_model_info = self.loaded_models['best']
            model = best_model_info['model']
            model_name = best_model_info['name']
            
            # Use the last row of features for prediction
            if hasattr(model, 'predict'):
                # For ML models, predict using features
                latest_features = features_df.iloc[-1:].copy()
                
                # Ensure we have all required features
                if hasattr(self, 'feature_names'):
                    missing_features = set(self.feature_names) - set(latest_features.columns)
                    for feature in missing_features:
                        latest_features[feature] = 0.0  # Default value
                    
                    # Reorder columns to match training
                    try:
                        latest_features = latest_features[self.feature_names]
                    except KeyError:
                        # Use available features
                        available_features = [f for f in self.feature_names if f in latest_features.columns]
                        latest_features = latest_features[available_features]
                
                # Make predictions (repeat for multiple hours if needed)
                predictions = []
                for i in range(forecast_hours):
                    pred = model.predict(latest_features)[0]
                    predictions.append(max(0, pred))  # Ensure non-negative
                
            else:
                # For time series models, generate forecast
                if hasattr(model, 'predict'):
                    predictions = model.predict(steps=forecast_hours)
                    predictions = [max(0, p) for p in predictions]  # Ensure non-negative
                else:
                    # Fallback to simple persistence
                    last_value = features_df['kwh'].iloc[-1]
                    predictions = [last_value] * forecast_hours
            
            # Generate forecast timestamps
            start_time = features_df.index[-1] + timedelta(hours=1)
            forecast_timestamps = [
                (start_time + timedelta(hours=i)).isoformat() 
                for i in range(forecast_hours)
            ]
            
            # Prepare response
            response = {
                'success': True,
                'model_used': model_name,
                'forecast_hours': forecast_hours,
                'predictions': [
                    {
                        'timestamp': ts,
                        'predicted_kwh': round(pred, 3)
                    }
                    for ts, pred in zip(forecast_timestamps, predictions)
                ],
                'summary': {
                    'total_predicted_kwh': round(sum(predictions), 3),
                    'average_hourly_kwh': round(np.mean(predictions), 3),
                    'peak_hour_kwh': round(max(predictions), 3),
                    'min_hour_kwh': round(min(predictions), 3)
                },
                'metadata': {
                    'model_metrics': best_model_info['metadata'].get('metrics', {}),
                    'prediction_timestamp': datetime.now().isoformat(),
                    'input_data_points': len(features_df)
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }

# Initialize the API
prediction_api = EnergyPredictionAPI()

@app.route('/')
def home():
    """Home page with API documentation."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Energy Consumption Predictor API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
            .endpoint { background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { background-color: #27ae60; color: white; padding: 5px 10px; border-radius: 3px; display: inline-block; }
            .example { background-color: #34495e; color: white; padding: 10px; border-radius: 3px; margin: 10px 0; }
            pre { background-color: #2c3e50; color: white; padding: 10px; border-radius: 3px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Energy Consumption Predictor API</h1>
            <p>RESTful API for predicting energy consumption using machine learning models</p>
        </div>
        
        <h2>Available Endpoints</h2>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> /health</h3>
            <p>Check API health status and available models</p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /predict</h3>
            <p>Predict energy consumption for the next 7-30 days</p>
            <h4>Request Body:</h4>
            <pre>{
  "recent_consumption": [2.3, 1.8, 1.5, ...],  // Recent hourly kWh values
  "weather_forecast": {
    "temperature_c": 22.5,
    "humidity_percent": 65.0,
    "wind_speed_kmh": 12.0,
    "precipitation_mm": 0.0,
    "cloud_cover_percent": 40.0,
    "solar_irradiance_wm2": 450.0
  },
  "forecast_hours": 168  // Number of hours to forecast (default: 24)
}</pre>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> /models</h3>
            <p>List available models and their performance metrics</p>
        </div>
        
        <h2>Example Usage</h2>
        <div class="example">
            <h4>Curl Example:</h4>
            <pre>curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "recent_consumption": [2.3, 1.8, 1.5, 1.2, 1.1, 1.3, 2.1, 3.2],
    "weather_forecast": {
      "temperature_c": 22.5,
      "humidity_percent": 65.0
    },
    "forecast_hours": 24
  }'</pre>
        </div>
        
        <h2>Response Format</h2>
        <pre>{
  "success": true,
  "model_used": "xgboost",
  "forecast_hours": 24,
  "predictions": [
    {
      "timestamp": "2024-01-01T00:00:00",
      "predicted_kwh": 2.456
    }
  ],
  "summary": {
    "total_predicted_kwh": 89.234,
    "average_hourly_kwh": 3.718,
    "peak_hour_kwh": 7.890,
    "min_hour_kwh": 1.234
  },
  "metadata": {
    "model_metrics": {"rmse": 0.123, "mae": 0.089},
    "prediction_timestamp": "2024-01-01T12:00:00"
  }
}</pre>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        available_models = len(prediction_api.loaded_models)
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'models_loaded': available_models,
            'api_version': '1.0.0',
            'modules_available': MODULES_AVAILABLE
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint."""
    try:
        # Get JSON data from request
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({
                'success': False,
                'error': 'No input data provided'
            }), 400
        
        # Get forecast hours (default 24)
        forecast_hours = input_data.get('forecast_hours', 24)
        
        # Validate forecast hours
        if forecast_hours < 1 or forecast_hours > 720:  # Max 30 days
            return jsonify({
                'success': False,
                'error': 'forecast_hours must be between 1 and 720'
            }), 400
        
        # Make prediction
        result = prediction_api.predict(input_data, forecast_hours)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List available models and their metrics."""
    try:
        models_info = []
        
        for model_key, model_info in prediction_api.loaded_models.items():
            metadata = model_info.get('metadata', {})
            models_info.append({
                'name': model_info.get('name'),
                'type': metadata.get('model_type'),
                'version': metadata.get('version'),
                'metrics': metadata.get('metrics', {}),
                'timestamp': metadata.get('timestamp')
            })
        
        return jsonify({
            'success': True,
            'models': models_info,
            'total_models': len(models_info)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint for multiple scenarios."""
    try:
        input_data = request.get_json()
        
        if not input_data or 'scenarios' not in input_data:
            return jsonify({
                'success': False,
                'error': 'scenarios array is required'
            }), 400
        
        scenarios = input_data['scenarios']
        results = []
        
        for i, scenario in enumerate(scenarios):
            try:
                forecast_hours = scenario.get('forecast_hours', 24)
                result = prediction_api.predict(scenario, forecast_hours)
                result['scenario_id'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'scenario_id': i,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'batch_results': results,
            'total_scenarios': len(scenarios),
            'successful_predictions': sum(1 for r in results if r.get('success', False))
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': ['/health', '/predict', '/models', '/predict/batch']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Configuration
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Start the Flask app
    print("="*60)
    print("ENERGY CONSUMPTION PREDICTOR API")
    print("="*60)
    print(f"Starting API server on {HOST}:{PORT}")
    print(f"Debug mode: {DEBUG}")
    print(f"Available models: {len(prediction_api.loaded_models)}")
    print("API Documentation: http://localhost:5000/")
    print("="*60)
    
    app.run(host=HOST, port=PORT, debug=DEBUG)