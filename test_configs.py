import yaml
import gym
import numpy as np
import argparse
from dreamer import Dreamer
import tools

def load_config(config_file):
    """Load configuration from YAML file using the same approach as main dreamer.py"""
    with open(config_file, 'r') as f:
        configs = yaml.safe_load(f)

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    # Get defaults
    defaults = configs['defaults'].copy()

    # Convert to argparse namespace (same as main dreamer.py)
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        # Fix: Apply the type conversion to the value to ensure proper types
        typed_value = arg_type(value) if not isinstance(value, (dict, list)) else value
        parser.add_argument(f"--{key}", type=arg_type, default=typed_value)

    # Parse empty args to get namespace with defaults
    config = parser.parse_args([])
    return config

def test_config(config_name, config_file):
    """Test a specific configuration."""
    print(f"\n=== Testing {config_name} ===")
    
    try:
        # Load config
        config = load_config(config_file)
        print(f"✓ Config loaded successfully")
        
        # Create mock observation and action spaces
        obs_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        })
        act_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # Set num_actions from action space (same as main dreamer.py)
        config.num_actions = act_space.n if hasattr(act_space, "n") else act_space.shape[0]

        # Create mock logger and dataset
        class MockLogger:
            def __init__(self):
                self.step = 0

        class MockDataset:
            def __init__(self):
                pass

        logger = MockLogger()
        dataset = MockDataset()

        # Create Dreamer instance
        dreamer = Dreamer(obs_space, act_space, config, logger, dataset)
        print(f"✓ Dreamer instance created successfully")
        
        # Check model type
        if hasattr(config, 'use_cnvae') and config.use_cnvae:
            print(f"✓ Using CNVAE encoder/decoder")
        else:
            print(f"✓ Using standard encoder/decoder")
            
        if hasattr(config, 'use_poisson') and config.use_poisson:
            print(f"✓ Using Poisson latents in RSSM")
        else:
            print(f"✓ Using categorical latents in RSSM")
            
        # Check KL annealing setup
        if hasattr(dreamer, '_betas') and dreamer._betas is not None:
            print(f"✓ KL annealing configured with {len(dreamer._betas)} steps")
            if hasattr(config, 'kl_anneal_complete_at_prefill') and config.kl_anneal_complete_at_prefill:
                print(f"✓ KL annealing will complete during prefill phase")
        
        print(f"✓ {config_name} configuration test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ {config_name} configuration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test all 4 configurations."""
    configs = [
        ("Vanilla Dreamer", "configs_dreamer.yaml"),
        ("pDreamer (Poisson RSSM)", "configs_pdreamer.yaml"), 
        ("hiDreamer (CNVAE + Categorical)", "configs_hidreamer.yaml"),
        ("phiDreamer (CNVAE + Poisson)", "configs_phidreamer.yaml")
    ]
    
    results = []
    for name, config_file in configs:
        success = test_config(name, config_file)
        results.append((name, success))
    
    print(f"\n=== SUMMARY ===")
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print(f"\n🎉 All configurations work correctly!")
    else:
        print(f"\n❌ Some configurations failed")
    
    return all_passed

if __name__ == "__main__":
    main()
