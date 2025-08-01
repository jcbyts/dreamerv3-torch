#!/usr/bin/env python3
"""
Test video logging and linear probing on currently running models
"""

import argparse
import pathlib
import sys
import torch
import numpy as np

# Add current directory to path
sys.path.append(str(pathlib.Path(__file__).parent))

def test_video_logging(model_dir, config_name, device='cuda:0'):
    """Test if video logging works with current model."""
    print(f"\n🎬 Testing video logging on {model_dir}")
    
    try:
        from evaluate_agent import AgentEvaluator
        
        # Create evaluator
        evaluator = AgentEvaluator(model_dir, config_name, device)
        
        # Test video prediction
        print("  Testing video prediction...")
        video_data = evaluator.generate_video_predictions(episodes=1, max_steps=50)
        
        if video_data is not None:
            print(f"  ✅ Video prediction successful: {video_data.shape}")
            return True
        else:
            print("  ❌ Video prediction failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Video logging test failed: {e}")
        return False

def test_linear_probing(model_dir, config_name, device='cuda:0'):
    """Test if linear probing works with current model."""
    print(f"\n🧠 Testing linear probing on {model_dir}")
    
    try:
        from eval_ego_linear_probe import run_ego_probe
        from torch.utils.tensorboard import SummaryWriter
        
        # Create a temporary writer for testing
        test_logdir = pathlib.Path(model_dir) / "test_linear_probe"
        test_logdir.mkdir(exist_ok=True)
        writer = SummaryWriter(str(test_logdir))
        
        print("  Running ego motion linear probing...")
        results = run_ego_probe(
            model_path=model_dir,
            config_name=config_name,
            step=999999,  # Test step
            writer=writer,
            device=device,
            episodes_train=3,  # Small number for quick test
            episodes_test=2,
            max_steps=100
        )
        
        writer.close()
        
        if results and 'r2_total' in results:
            print(f"  ✅ Linear probing successful: R² = {results['r2_total']:.3f}")
            return True
        else:
            print("  ❌ Linear probing failed - no results")
            return False
            
    except Exception as e:
        print(f"  ❌ Linear probing test failed: {e}")
        return False

def find_current_models():
    """Find currently running model directories."""
    logdir = pathlib.Path("./logdir")
    
    # Look for recent comparison directories
    comparison_dirs = []
    for d in logdir.glob("comparison_*"):
        if d.is_dir():
            comparison_dirs.append(d)
    
    # Sort by modification time, get most recent
    if comparison_dirs:
        comparison_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        recent_dir = comparison_dirs[0]
        
        print(f"📁 Found recent comparison: {recent_dir.name}")
        
        # Find model subdirectories
        models = []
        for model_dir in recent_dir.glob("*_seed_*"):
            if model_dir.is_dir() and (model_dir / "latest.pt").exists():
                models.append(model_dir)
        
        return models
    
    return []

def main():
    parser = argparse.ArgumentParser(description='Test video logging and linear probing on current models')
    parser.add_argument('--model-dir', type=str, help='Specific model directory to test')
    parser.add_argument('--config', type=str, help='Config name (e.g., vizdoom_health_gathering)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--test-video', action='store_true', help='Test video logging only')
    parser.add_argument('--test-linear', action='store_true', help='Test linear probing only')
    
    args = parser.parse_args()
    
    print("🧪 Testing Current Models")
    print("=" * 50)
    
    # Find models to test
    if args.model_dir:
        models = [pathlib.Path(args.model_dir)]
        config_name = args.config or "vizdoom_health_gathering"
    else:
        models = find_current_models()
        config_name = "vizdoom_health_gathering"  # Default for current runs
    
    if not models:
        print("❌ No models found to test")
        return
    
    print(f"Found {len(models)} models to test")
    
    # Test each model
    results = {}
    for model_dir in models[:2]:  # Test first 2 models to save time
        model_name = model_dir.name
        print(f"\n🔍 Testing model: {model_name}")
        
        # Determine config from model name
        if 'health' in str(model_dir):
            config = 'vizdoom_health_gathering'
        elif 'deadly' in str(model_dir):
            config = 'vizdoom_deadly_corridor'
        else:
            config = config_name
        
        results[model_name] = {}
        
        # Test video logging
        if not args.test_linear:
            video_ok = test_video_logging(str(model_dir), config, args.device)
            results[model_name]['video'] = video_ok
        
        # Test linear probing  
        if not args.test_video:
            linear_ok = test_linear_probing(str(model_dir), config, args.device)
            results[model_name]['linear'] = linear_ok
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 50)
    for model_name, tests in results.items():
        print(f"{model_name}:")
        if 'video' in tests:
            status = "✅" if tests['video'] else "❌"
            print(f"  Video logging: {status}")
        if 'linear' in tests:
            status = "✅" if tests['linear'] else "❌"
            print(f"  Linear probing: {status}")

if __name__ == "__main__":
    main()
