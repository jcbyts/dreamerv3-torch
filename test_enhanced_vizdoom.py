#!/usr/bin/env python3
"""
Test script to verify enhanced VizDoom environment and see what ground truth factors we can extract
"""

import numpy as np
from enhanced_vizdoom import EnhancedViZDoom, extract_ground_truth_factors


def test_enhanced_vizdoom():
    """Test the enhanced VizDoom environment."""
    print("Testing Enhanced VizDoom Environment")
    print("=" * 50)
    
    # Test different scenarios
    scenarios = ['basic', 'deadly_corridor', 'defend_the_center']
    
    for scenario in scenarios:
        print(f"\nTesting scenario: {scenario}")
        print("-" * 30)
        
        try:
            # Create environment
            env = EnhancedViZDoom(scenario, enable_labels=True, enable_depth=True)
            
            # Reset and take a few steps
            obs = env.reset()
            print(f"Observation keys: {list(obs.keys())}")
            print(f"Image shape: {obs['image'].shape}")
            
            if 'labels' in obs:
                print(f"Labels shape: {obs['labels'].shape}")
                print(f"Unique labels: {np.unique(obs['labels'])}")
            
            if 'depth' in obs:
                print(f"Depth shape: {obs['depth'].shape}")
                print(f"Depth range: [{np.min(obs['depth']):.2f}, {np.max(obs['depth']):.2f}]")
            
            # Extract ground truth factors
            factors = extract_ground_truth_factors(obs)
            print(f"Ground truth factors: {list(factors.keys())}")
            
            for factor_name, value in factors.items():
                if isinstance(value, (int, float)):
                    print(f"  {factor_name}: {value:.4f}")
                else:
                    print(f"  {factor_name}: {value}")
            
            # Take a few random actions to see variation
            print("\nTaking 5 random actions to see factor variation:")
            for step in range(5):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                factors = extract_ground_truth_factors(obs)
                
                print(f"  Step {step+1}:")
                for factor_name, value in factors.items():
                    if isinstance(value, (int, float)):
                        print(f"    {factor_name}: {value:.4f}")
            
            env.close()
            print(f"✓ {scenario} test completed successfully")
            
        except Exception as e:
            print(f"✗ {scenario} test failed: {e}")
            continue
    
    print("\n" + "=" * 50)
    print("Enhanced VizDoom test completed!")


if __name__ == "__main__":
    test_enhanced_vizdoom()
