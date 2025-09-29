#!/usr/bin/env python3
"""
Quick test script to verify the Energy Consumption Predictor installation.
"""

import sys
import os
sys.path.append('src')

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
        
        import numpy as np
        print("✅ numpy imported successfully")
        
        import sklearn
        print("✅ scikit-learn imported successfully")
        
        import xgboost
        print("✅ xgboost imported successfully")
        
        import lightgbm
        print("✅ lightgbm imported successfully")
        
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported successfully")
        
        import plotly
        print("✅ plotly imported successfully")
        
        import flask
        print("✅ flask imported successfully")
        
        from data_preprocessing import DataPreprocessor
        print("✅ DataPreprocessor imported successfully")
        
        from feature_engineering import FeatureEngineer
        print("✅ FeatureEngineer imported successfully")
        
        from modeling import EnergyPredictionPipeline
        print("✅ EnergyPredictionPipeline imported successfully")
        
        from utils import load_config
        print("✅ Utils imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_files():
    """Test that data files exist and can be loaded."""
    print("\n📊 Testing data files...")
    
    try:
        electricity_path = 'data/raw/electricity_consumption.csv'
        weather_path = 'data/raw/weather_data.csv'
        
        if os.path.exists(electricity_path) and os.path.exists(weather_path):
            import pandas as pd
            elec_df = pd.read_csv(electricity_path)
            weather_df = pd.read_csv(weather_path)
            
            print(f"✅ Electricity data: {elec_df.shape[0]} records")
            print(f"✅ Weather data: {weather_df.shape[0]} records")
            return True
        else:
            print("❌ Sample data files not found")
            return False
            
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_models():
    """Test that trained models exist."""
    print("\n🤖 Testing trained models...")
    
    try:
        models_dir = 'models'
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
            if model_files:
                print(f"✅ Found {len(model_files)} trained models:")
                for model_file in model_files:
                    print(f"   - {model_file}")
                return True
            else:
                print("⚠️ No trained model files found (.joblib)")
                return False
        else:
            print("⚠️ Models directory not found")
            return False
            
    except Exception as e:
        print(f"❌ Model check error: {e}")
        return False

def test_config():
    """Test that configuration loads correctly."""
    print("\n⚙️ Testing configuration...")
    
    try:
        from utils import load_config
        config = load_config()
        
        if config and 'DATA_PATHS' in config:
            print("✅ Configuration loaded successfully")
            print(f"   - Found {len(config)} configuration sections")
            return True
        else:
            print("❌ Configuration is empty or missing key sections")
            return False
            
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("🧪 ENERGY CONSUMPTION PREDICTOR - INSTALLATION TEST")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Data Files", test_data_files),
        ("Trained Models", test_models),
        ("Configuration", test_config)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "="*60)
    print("📋 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready for use.")
        print("\n📚 Next steps:")
        print("   1. Run training: python main.py train")
        print("   2. Start API: python main.py serve")
        print("   3. Open notebook: notebooks/Energy_Consumption_Predictor_Workflow.ipynb")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)