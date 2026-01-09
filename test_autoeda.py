#!/usr/bin/env python3
"""
Test script for AutoEDA system

This script tests the main components of the AutoEDA system to ensure
they are working correctly before running the full application.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
# Import project modules for tests (make available in module scope)
from models.lstm_autoencoder import LSTMAutoencoder
from models.attention_mechanism import AttentionMechanism
from utils.data_processor import DataProcessor
from utils.anomaly_detector import AnomalyDetector
from utils.synthetic_generator import SyntheticDataGenerator
from utils.visualization import VisualizationGenerator
from config.config import Config, DevelopmentConfig
def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from models.lstm_autoencoder import LSTMAutoencoder
        from models.attention_mechanism import AttentionMechanism
        from utils.data_processor import DataProcessor
        from utils.anomaly_detector import AnomalyDetector
        from utils.synthetic_generator import SyntheticDataGenerator
        from utils.visualization import VisualizationGenerator
        from config.config import Config
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_processor():
    """Test the DataProcessor class"""
    print("\nTesting DataProcessor...")
    
    try:
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=100, freq='H')
        data = {
            'timestamp': dates,
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(0, 1, 100),
            'feature_3': np.random.normal(0, 1, 100)
        }
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # Test DataProcessor
        processor = DataProcessor()
        processed_data = processor.preprocess_data(df, {})
        # processed_data is a dict with key 'data' containing sequences
        seq_shape = processed_data.get('data').shape if isinstance(processed_data, dict) else None
        print(f"✓ DataProcessor test passed - Input shape: {df.shape}, Sequences shape: {seq_shape}")
        return True
    except Exception as e:
        print(f"✗ DataProcessor test failed: {e}")
        return False

def test_lstm_autoencoder():
    """Test the LSTM Autoencoder class"""
    print("\nTesting LSTM Autoencoder...")
    
    try:
        # Create sample data
        X = np.random.random((100, 10, 3))  # 100 samples, 10 timesteps, 3 features
        
        # Test LSTM Autoencoder (builds model in constructor)
        autoencoder = LSTMAutoencoder(input_shape=(10, 3), config={'encoder_units':[32,16], 'decoder_units':[16,32]})
        # Show summary (prints to stdout)
        autoencoder.summary()
        print("✓ LSTM Autoencoder model built successfully")
        
        return True
    except Exception as e:
        print(f"✗ LSTM Autoencoder test failed: {e}")
        return False

def test_attention_mechanism():
    """Test the Attention Mechanism class"""
    print("\nTesting Attention Mechanism...")
    
    try:
        # Use the internal correlation-based weights computation without constructing full attention model
        X = np.random.random((5, 10, 3))
        attention = object.__new__(AttentionMechanism)
        weights = AttentionMechanism._calculate_correlation_weights(attention, X)
        print(f"✓ Attention Mechanism test built weights with shape: {weights.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Attention Mechanism test failed: {e}")
        return False

def test_anomaly_detector():
    """Test the Anomaly Detector class"""
    print("\nTesting Anomaly Detector...")
    
    try:
        # Create sample data
        X = np.random.random((100, 10, 3))
        
        # Test Anomaly Detector - needs a model with get_reconstruction_error
        detector = AnomalyDetector()
        autoencoder = LSTMAutoencoder(input_shape=(10,3))
        # mark as trained so get_reconstruction_error can run (uses model.predict)
        autoencoder.is_trained = True
        results = detector.detect_anomalies(autoencoder, X, method='reconstruction')
        
        print(f"✓ Anomaly Detector test passed")
        print(f"  Anomalies detected: {results.get('n_anomalies', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"✗ Anomaly Detector test failed: {e}")
        return False

def test_synthetic_generator():
    """Test the Synthetic Data Generator class"""
    print("\nTesting Synthetic Data Generator...")
    
    try:
        # Create sample data
        X = np.random.random((100, 10, 3))
        
        # Test Synthetic Data Generator (requires a model with encode/decode)
        generator = SyntheticDataGenerator()
        autoencoder = LSTMAutoencoder(input_shape=(10,3))
        autoencoder.is_trained = True
        synthetic_data = generator.generate_data(autoencoder, X, n_samples=50)
        
        print(f"✓ Synthetic Data Generator test passed")
        print(f"  Generated {len(synthetic_data)} synthetic samples")
        
        return True
    except Exception as e:
        print(f"✗ Synthetic Data Generator test failed: {e}")
        return False

def test_visualization():
    """Test the Visualization Generator class"""
    print("\nTesting Visualization Generator...")
    
    try:
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=100, freq='H')
        data = {
            'timestamp': dates,
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(0, 1, 100)
        }
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # Test Visualization Generator
        viz_gen = VisualizationGenerator()
        
        # Test visualization generation
        filepath = viz_gen.generate_data_overview(df)
        
        print(f"✓ Visualization Generator test passed")
        print(f"  Generated visualization: {filepath}")
        
        return True
    except Exception as e:
        print(f"✗ Visualization Generator test failed: {e}")
        return False

def test_config():
    """Test the configuration system"""
    print("\nTesting Configuration...")
    
    try:
        from config.config import Config, DevelopmentConfig
        
        # Test default config
        config = Config()
        print(f"✓ Default config loaded - Debug mode: {config.DEBUG}")
        
        # Test development config
        dev_config = DevelopmentConfig()
        print(f"✓ Development config loaded - Debug mode: {dev_config.DEBUG}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("AutoEDA System Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config,
        test_data_processor,
        test_lstm_autoencoder,
        test_attention_mechanism,
        test_anomaly_detector,
        test_synthetic_generator,
        test_visualization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AutoEDA system is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

def main():
    """Main function"""
    if run_all_tests():
        print("\nYou can now run the AutoEDA application with:")
        print("python app.py")
    else:
        print("\nPlease fix the failing tests before running the application.")

if __name__ == "__main__":
    main()
