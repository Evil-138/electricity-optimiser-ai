"""
Main entry point for Energy Consumption Predictor project.
This script orchestrates the complete pipeline from data processing to model deployment.
"""

import os
import sys
import argparse
from datetime import datetime

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import load_config, setup_logging, Timer, save_results_summary
from src.train_pipeline import ModelTrainer

def run_training_pipeline(config_path: str = 'config/config.yaml'):
    """
    Run the complete training pipeline.
    
    Args:
        config_path: Path to configuration file
    """
    print("="*80)
    print("ENERGY CONSUMPTION PREDICTOR - COMPLETE TRAINING PIPELINE")
    print("="*80)
    
    # Load configuration
    config = load_config(config_path)
    logger = setup_logging(config)
    
    # Get paths from config
    data_paths = config.get('DATA_PATHS', {})
    model_paths = config.get('MODEL_PATHS', {})
    
    # Initialize trainer
    with Timer("Training Pipeline"):
        trainer = ModelTrainer(
            electricity_path=data_paths.get('raw_electricity', 'data/raw/electricity_consumption.csv'),
            weather_path=data_paths.get('raw_weather', 'data/raw/weather_data.csv'),
            models_dir=model_paths.get('models_dir', 'models'),
            processed_data_dir='data/processed'
        )
        
        # Create training configuration from loaded config
        training_config = {
            'preprocessing': config.get('PREPROCESSING', {}),
            'feature_engineering': config.get('FEATURE_ENGINEERING', {}),
            'training': config.get('TRAINING', {}),
            'model_version': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'best_model_metric': 'rmse'
        }
        
        # Run complete pipeline
        results = trainer.run_complete_pipeline(training_config)
        
        # Save results summary
        summary_path = f"results_summary_{training_config['model_version']}.json"
        save_results_summary(results, summary_path)
        
        logger.info("Training pipeline completed successfully!")
        
        return results

def run_api_server(config_path: str = 'config/config.yaml'):
    """
    Start the Flask API server.
    
    Args:
        config_path: Path to configuration file
    """
    print("="*80)
    print("ENERGY CONSUMPTION PREDICTOR - API SERVER")
    print("="*80)
    
    # Load configuration
    config = load_config(config_path)
    api_config = config.get('API', {})
    
    # Set environment variables for Flask app
    os.environ['HOST'] = api_config.get('host', '0.0.0.0')
    os.environ['PORT'] = str(api_config.get('port', 5000))
    os.environ['DEBUG'] = str(api_config.get('debug', False))
    
    # Import and run the Flask app
    from api.app import app
    
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 5000)
    debug = api_config.get('debug', False)
    
    print(f"Starting API server on {host}:{port}")
    print(f"Debug mode: {debug}")
    print("API documentation available at: http://localhost:5000/")
    print("="*80)
    
    app.run(host=host, port=port, debug=debug)

def run_data_generation():
    """Generate sample data for the project."""
    print("="*80)
    print("GENERATING SAMPLE DATA")
    print("="*80)
    
    try:
        # Run the data generation script
        import subprocess
        result = subprocess.run([sys.executable, 'generate_sample_data.py'], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Sample data generated successfully!")
            print(result.stdout)
        else:
            print("Error generating sample data:")
            print(result.stderr)
    except Exception as e:
        print(f"Error running data generation: {e}")

def run_tests():
    """Run API tests."""
    print("="*80)
    print("RUNNING API TESTS")
    print("="*80)
    
    try:
        # Run the API tests
        import subprocess
        result = subprocess.run([sys.executable, 'api/test_api.py'], 
                               capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Test warnings/errors:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("All tests completed successfully!")
        else:
            print("Some tests may have failed. Check the output above.")
            
    except Exception as e:
        print(f"Error running tests: {e}")

def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description="Energy Consumption Predictor - Complete ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                    # Run complete training pipeline
  python main.py serve                    # Start API server
  python main.py generate-data           # Generate sample data
  python main.py test                    # Run API tests
  python main.py --config custom.yaml train  # Use custom config file
        """
    )
    
    parser.add_argument(
        'command',
        choices=['train', 'serve', 'generate-data', 'test'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Set up basic logging for main script
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    try:
        if args.command == 'train':
            results = run_training_pipeline(args.config)
            print(f"\nTraining completed! Check the results summary and model files.")
            
        elif args.command == 'serve':
            run_api_server(args.config)
            
        elif args.command == 'generate-data':
            run_data_generation()
            
        elif args.command == 'test':
            print("Note: Make sure the API server is running before running tests.")
            print("You can start it with: python main.py serve")
            run_tests()
            
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()