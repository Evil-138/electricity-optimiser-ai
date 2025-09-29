"""
Test script for the Energy Consumption Predictor API.
"""

import requests
import json
import time
from datetime import datetime, timedelta

# API Configuration
API_BASE_URL = "http://localhost:5000"

def test_health_endpoint():
    """Test the health check endpoint."""
    print("Testing health endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_prediction_endpoint():
    """Test the main prediction endpoint."""
    print("\nTesting prediction endpoint...")
    
    # Sample data
    test_data = {
        "recent_consumption": [
            2.3, 1.8, 1.5, 1.2, 1.1, 1.3, 2.1, 3.2,
            4.1, 3.8, 3.5, 3.7, 4.2, 4.5, 4.8, 5.1,
            5.3, 6.2, 7.8, 8.1, 7.5, 6.8, 5.2, 3.8
        ],
        "weather_forecast": {
            "temperature_c": 22.5,
            "humidity_percent": 65.0,
            "wind_speed_kmh": 12.0,
            "precipitation_mm": 0.0,
            "cloud_cover_percent": 40.0,
            "solar_irradiance_wm2": 450.0
        },
        "forecast_hours": 24
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            headers={"Content-Type": "application/json"},
            json=test_data
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        if result.get('success'):
            print("✓ Prediction successful!")
            print(f"Model used: {result.get('model_used')}")
            print(f"Forecast hours: {result.get('forecast_hours')}")
            print(f"Total predicted kWh: {result.get('summary', {}).get('total_predicted_kwh')}")
            print(f"Average hourly kWh: {result.get('summary', {}).get('average_hourly_kwh')}")
            print(f"First 5 predictions:")
            
            for i, pred in enumerate(result.get('predictions', [])[:5]):
                print(f"  {pred['timestamp']}: {pred['predicted_kwh']} kWh")
                
        else:
            print("✗ Prediction failed!")
            print(f"Error: {result.get('error')}")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_models_endpoint():
    """Test the models listing endpoint."""
    print("\nTesting models endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        if result.get('success'):
            print("✓ Models endpoint working!")
            print(f"Total models: {result.get('total_models')}")
            
            for model in result.get('models', []):
                print(f"  - {model.get('name')} ({model.get('type')})")
                metrics = model.get('metrics', {})
                if metrics:
                    print(f"    RMSE: {metrics.get('rmse', 'N/A')}")
                    print(f"    MAE: {metrics.get('mae', 'N/A')}")
        else:
            print("✗ Models endpoint failed!")
            print(f"Error: {result.get('error')}")
            
        return result.get('success', False)
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("\nTesting batch prediction endpoint...")
    
    batch_data = {
        "scenarios": [
            {
                "recent_consumption": [2.0, 1.5, 1.2, 1.0, 1.1, 2.0, 3.5, 4.2],
                "weather_forecast": {"temperature_c": 20.0, "humidity_percent": 60.0},
                "forecast_hours": 12
            },
            {
                "recent_consumption": [3.0, 2.5, 2.2, 2.0, 2.1, 3.0, 4.5, 5.2],
                "weather_forecast": {"temperature_c": 25.0, "humidity_percent": 70.0},
                "forecast_hours": 12
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            headers={"Content-Type": "application/json"},
            json=batch_data
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        if result.get('success'):
            print("✓ Batch prediction successful!")
            print(f"Total scenarios: {result.get('total_scenarios')}")
            print(f"Successful predictions: {result.get('successful_predictions')}")
        else:
            print("✗ Batch prediction failed!")
            print(f"Error: {result.get('error')}")
            
        return result.get('success', False)
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid data."""
    print("\nTesting error handling...")
    
    # Test with invalid data
    invalid_data = {
        "forecast_hours": 1000  # Too many hours
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            headers={"Content-Type": "application/json"},
            json=invalid_data
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        if response.status_code == 400 and not result.get('success'):
            print("✓ Error handling working correctly!")
            print(f"Error message: {result.get('error')}")
            return True
        else:
            print("✗ Error handling not working as expected")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def run_all_tests():
    """Run all API tests."""
    print("="*60)
    print("ENERGY CONSUMPTION PREDICTOR API TESTS")
    print("="*60)
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Prediction", test_prediction_endpoint),
        ("Models Listing", test_models_endpoint),
        ("Batch Prediction", test_batch_prediction),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running {test_name} test...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results.append((test_name, False))
        
        time.sleep(0.5)  # Small delay between tests
    
    # Print summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:<20} {status}")
        if success:
            passed += 1
    
    print(f"\nTests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Check the API server logs.")

if __name__ == "__main__":
    run_all_tests()