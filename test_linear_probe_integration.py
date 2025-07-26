#!/usr/bin/env python3
"""
Test script to verify linear probing integration works correctly.
"""

import sys
import pathlib
import torch

# Add the dreamerv3-torch directory to the path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

def test_import():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from eval_ego_linear_probe import run_ego_probe, EGO_FACTORS
        print("✓ eval_ego_linear_probe imported successfully")
        print(f"  Ego factors: {EGO_FACTORS}")
    except ImportError as e:
        print(f"✗ Failed to import eval_ego_linear_probe: {e}")
        return False
    
    try:
        from enhanced_vizdoom import EnhancedViZDoom, extract_ground_truth_factors
        print("✓ enhanced_vizdoom imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import enhanced_vizdoom: {e}")
        return False
    
    try:
        from evaluate_agent import AgentEvaluator
        print("✓ evaluate_agent imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import evaluate_agent: {e}")
        return False
    
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score
        print("✓ sklearn components imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import sklearn: {e}")
        print("  Please install scikit-learn: pip install scikit-learn")
        return False
    
    return True


def test_config_parameters():
    """Test that configuration parameters are properly set."""
    print("\nTesting configuration parameters...")
    
    try:
        import ruamel.yaml as yaml
        
        # Load config
        config_path = pathlib.Path(__file__).parent / 'configs.yaml'
        with open(config_path, 'r') as f:
            configs = yaml.safe_load(f)
        
        # Check base config has linear probing parameters
        base_config = configs.get('defaults', {})
        
        required_params = [
            'linear_probe_every',
            'linear_probe_episodes_train', 
            'linear_probe_episodes_test',
            'linear_probe_max_steps',
            'linear_probe_log_per_factor'
        ]
        
        missing_params = []
        for param in required_params:
            if param not in base_config:
                missing_params.append(param)
            else:
                print(f"✓ {param}: {base_config[param]}")
        
        if missing_params:
            print(f"✗ Missing config parameters: {missing_params}")
            return False
        
        print("✓ All linear probing config parameters found")
        return True
        
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False


def test_dreamer_integration():
    """Test that Dreamer class has linear probing integration."""
    print("\nTesting Dreamer integration...")
    
    try:
        import dreamer
        
        # Check that LINEAR_PROBE_AVAILABLE is defined
        if hasattr(dreamer, 'LINEAR_PROBE_AVAILABLE'):
            print(f"✓ LINEAR_PROBE_AVAILABLE: {dreamer.LINEAR_PROBE_AVAILABLE}")
        else:
            print("✗ LINEAR_PROBE_AVAILABLE not found in dreamer module")
            return False
        
        # Check that run_ego_probe is imported if available
        if dreamer.LINEAR_PROBE_AVAILABLE:
            if hasattr(dreamer, 'run_ego_probe'):
                print("✓ run_ego_probe imported in dreamer module")
            else:
                print("✗ run_ego_probe not found in dreamer module")
                return False
        
        print("✓ Dreamer integration looks good")
        return True
        
    except Exception as e:
        print(f"✗ Failed to test Dreamer integration: {e}")
        return False


def test_ego_factors():
    """Test ego factor extraction and conversion."""
    print("\nTesting ego factor extraction...")
    
    try:
        from enhanced_vizdoom import extract_ground_truth_factors
        import numpy as np
        
        # Create mock observation
        mock_obs = {
            'game_variables': [100.0, 200.0, 0.0, 45.0, 1.0, 0.5, 0.0, 100.0],  # x, y, z, angle, vx, vy, vz, health
            'labels': np.zeros((64, 64), dtype=np.uint8)
        }
        
        # Extract factors
        factors = extract_ground_truth_factors(mock_obs)
        
        # Check that ego factors are present
        ego_factor_keys = [
            'ego_pos_x', 'ego_pos_y', 'ego_pos_z',
            'ego_angle_sin', 'ego_angle_cos',
            'ego_vel_x', 'ego_vel_y', 'ego_vel_z',
            'ego_delta_x', 'ego_delta_y', 'ego_delta_z', 'ego_delta_angle'
        ]
        
        missing_factors = []
        for key in ego_factor_keys:
            if key not in factors:
                missing_factors.append(key)
            else:
                print(f"✓ {key}: {factors[key]}")
        
        if missing_factors:
            print(f"✗ Missing ego factors: {missing_factors}")
            return False
        
        # Test angle conversion
        angle = factors.get('ego_delta_angle', 0.0)
        sin_val = np.sin(angle)
        cos_val = np.cos(angle)
        
        print(f"✓ Angle conversion test: angle={angle}, sin={sin_val:.3f}, cos={cos_val:.3f}")
        print("✓ Ego factor extraction working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Failed to test ego factor extraction: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Linear Probing Integration Test")
    print("=" * 60)
    
    tests = [
        test_import,
        test_config_parameters,
        test_dreamer_integration,
        test_ego_factors
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Linear probing integration is ready.")
        print("\nTo use linear probing during training:")
        print("1. Make sure sklearn is installed: pip install scikit-learn")
        print("2. Run training with VizDoom tasks (e.g., vizdoom_basic)")
        print("3. Linear probing will run every 20,000 steps by default")
        print("4. Check tensorboard for 'linear_probe/ego_r2_total_*' metrics")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
