#!/usr/bin/env python3
"""
Simple test script to verify the web API is working correctly.
"""

import requests
import json

def test_health():
    """Test the health endpoint."""
    try:
        response = requests.get('http://localhost:5000/health')
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
    except Exception as e:
        print(f"Health check failed: {e}")
    return False

def test_prediction():
    """Test the prediction endpoint."""
    try:
        data = {
            "recent_consumption": [2.1, 1.8, 1.9, 2.2, 2.5, 3.1, 3.8, 4.2],
            "weather_forecast": {
                "temperature_c": 22.5,
                "humidity_percent": 65,
                "wind_speed_kmh": 12
            }
        }
        
        response = requests.post(
            'http://localhost:5000/predict',
            headers={'Content-Type': 'application/json'},
            json=data
        )
        
        print(f"Prediction test: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction: {result.get('prediction', 'N/A')} kWh")
            print(f"Model: {result.get('model_used', 'N/A')}")
            return True
        else:
            print(f"Error response: {response.text}")
    except Exception as e:
        print(f"Prediction test failed: {e}")
    return False

def test_batch_prediction():
    """Test the batch prediction endpoint."""
    try:
        data = {
            "recent_consumption": [2.1, 1.8, 1.9, 2.2, 2.5, 3.1, 3.8, 4.2],
            "forecast_hours": 6,
            "model_name": "xgboost"
        }
        
        response = requests.post(
            'http://localhost:5000/predict/batch',
            headers={'Content-Type': 'application/json'},
            json=data
        )
        
        print(f"Batch prediction test: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            predictions = result.get('predictions', [])
            print(f"Generated {len(predictions)} predictions")
            if predictions:
                print(f"First 3 predictions: {[round(p, 3) for p in predictions[:3]]}")
            return True
        else:
            print(f"Error response: {response.text}")
    except Exception as e:
        print(f"Batch prediction test failed: {e}")
    return False

if __name__ == '__main__':
    print("🧪 Testing Energy Consumption Predictor API")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Single Prediction", test_prediction),
        ("Batch Prediction", test_batch_prediction)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        if test_func():
            print("✅ PASSED")
            passed += 1
        else:
            print("❌ FAILED")
    
    print(f"\n🏆 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All API tests passed! The web app is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the web app logs.")