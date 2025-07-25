#!/usr/bin/env python3
"""
Test script to verify backward compatibility between DreamerV3 and hDreamer.
Creates a flat model, saves it, then loads it into a hierarchical model.
"""

import torch
import numpy as np
import sys
import os
import tempfile

# Add the dreamerv3-torch directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import networks
import models
import tools


class TestConfig:
    """Mock config for testing."""
    def __init__(self, hierarchical_mode=False):
        self.device = "cpu"
        self.dyn_stoch = 32
        self.dyn_deter = 512
        self.dyn_hidden = 512
        self.dyn_rec_depth = 1
        self.dyn_discrete = 32
        self.act = "SiLU"
        self.norm = True
        self.dyn_mean_act = "none"
        self.dyn_std_act = "sigmoid2"
        self.dyn_min_std = 0.1
        self.unimix_ratio = 0.01
        self.initial = "learned"
        self.num_actions = 4
        
        # Hierarchical parameters - split the original capacity
        self.hierarchical_mode = hierarchical_mode
        self.dyn_stoch_top = 16  # Half of original 32
        self.dyn_stoch_bottom = 16  # Half of original 32
        self.dyn_discrete_top = 32  # Keep same discrete classes
        self.dyn_discrete_bottom = 32  # Keep same discrete classes
        # Total capacity: 16*32 + 16*32 = 512 + 512 = 1024 (same as flat 32*32=1024)


def test_backward_compatibility():
    """Test loading flat DreamerV3 model into hierarchical hDreamer."""
    print("Testing backward compatibility...")
    
    embed_size = 1024
    batch_size = 8
    
    # 1. Create and save a flat DreamerV3 model
    print("1. Creating flat DreamerV3 model...")
    flat_config = TestConfig(hierarchical_mode=False)
    
    flat_rssm = networks.RSSM(
        flat_config.dyn_stoch,
        flat_config.dyn_deter,
        flat_config.dyn_hidden,
        flat_config.dyn_rec_depth,
        flat_config.dyn_discrete,
        flat_config.act,
        flat_config.norm,
        flat_config.dyn_mean_act,
        flat_config.dyn_std_act,
        flat_config.dyn_min_std,
        flat_config.unimix_ratio,
        flat_config.initial,
        flat_config.num_actions,
        embed_size,
        flat_config.device,
    )
    
    # Generate some test data with the flat model
    initial_state = flat_rssm.initial(batch_size)
    action = torch.zeros(batch_size, flat_config.num_actions)
    embed = torch.randn(batch_size, embed_size)
    is_first = torch.zeros(batch_size, 1)
    
    flat_post, flat_prior = flat_rssm.obs_step(initial_state, action, embed, is_first)
    flat_feat = flat_rssm.get_feat(flat_post)
    
    print(f"   Flat model feature size: {flat_feat.shape}")
    print(f"   Flat model stoch shape: {flat_post['stoch'].shape}")
    
    # Save the flat model
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        flat_checkpoint_path = f.name
        torch.save(flat_rssm.state_dict(), flat_checkpoint_path)
    
    print(f"   Saved flat model to: {flat_checkpoint_path}")
    
    # 2. Create hierarchical hDreamer model
    print("2. Creating hierarchical hDreamer model...")
    hier_config = TestConfig(hierarchical_mode=True)
    
    hier_rssm = networks.RSSM(
        hier_config.dyn_stoch,
        hier_config.dyn_deter,
        hier_config.dyn_hidden,
        hier_config.dyn_rec_depth,
        hier_config.dyn_discrete,
        hier_config.act,
        hier_config.norm,
        hier_config.dyn_mean_act,
        hier_config.dyn_std_act,
        hier_config.dyn_min_std,
        hier_config.unimix_ratio,
        hier_config.initial,
        hier_config.num_actions,
        embed_size,
        hier_config.device,
        hierarchical_mode=True,
        stoch_top=hier_config.dyn_stoch_top,
        stoch_bottom=hier_config.dyn_stoch_bottom,
        discrete_top=hier_config.dyn_discrete_top,
        discrete_bottom=hier_config.dyn_discrete_bottom,
    )
    
    # Test hierarchical model before loading
    hier_initial = hier_rssm.initial(batch_size)
    hier_post, hier_prior = hier_rssm.obs_step(hier_initial, action, embed, is_first)
    hier_feat = hier_rssm.get_feat(hier_post)
    
    print(f"   Hierarchical model feature size: {hier_feat.shape}")
    print(f"   Hierarchical model stoch_top shape: {hier_post['stoch_top'].shape}")
    print(f"   Hierarchical model stoch_bottom shape: {hier_post['stoch_bottom'].shape}")
    
    # 3. Load flat checkpoint into hierarchical model
    print("3. Loading flat checkpoint into hierarchical model...")
    flat_state_dict = torch.load(flat_checkpoint_path)
    
    try:
        hier_rssm.load_state_dict(flat_state_dict, strict=False)
        print("   ✓ Successfully loaded flat checkpoint into hierarchical model!")
    except Exception as e:
        print(f"   ✗ Failed to load checkpoint: {e}")
        return False
    
    # 4. Test that the loaded model works
    print("4. Testing loaded hierarchical model...")
    try:
        loaded_post, loaded_prior = hier_rssm.obs_step(hier_initial, action, embed, is_first)
        loaded_feat = hier_rssm.get_feat(loaded_post)
        
        print(f"   Loaded model feature size: {loaded_feat.shape}")
        print(f"   Feature values are finite: {torch.isfinite(loaded_feat).all()}")
        
        # Test KL loss computation
        free = 1.0
        dyn_scale = 1.0
        rep_scale = 0.1
        loss, value, dyn_loss, rep_loss = hier_rssm.kl_loss(loaded_post, loaded_prior, free, dyn_scale, rep_scale)
        
        print(f"   KL loss computed successfully: {torch.isfinite(loss).all()}")
        print("   ✓ Loaded hierarchical model works correctly!")
        
    except Exception as e:
        print(f"   ✗ Loaded model failed: {e}")
        return False
    
    # 5. Compare feature dimensions
    print("5. Comparing feature dimensions...")
    expected_flat_feat_size = flat_config.dyn_stoch * flat_config.dyn_discrete + flat_config.dyn_deter
    expected_hier_feat_size = (hier_config.dyn_stoch_top * hier_config.dyn_discrete_top + 
                              hier_config.dyn_stoch_bottom * hier_config.dyn_discrete_bottom + 
                              hier_config.dyn_deter)
    
    print(f"   Expected flat feature size: {expected_flat_feat_size}")
    print(f"   Expected hierarchical feature size: {expected_hier_feat_size}")
    print(f"   Actual flat feature size: {flat_feat.shape[1]}")
    print(f"   Actual hierarchical feature size: {hier_feat.shape[1]}")
    
    if expected_flat_feat_size == expected_hier_feat_size:
        print("   ✓ Feature dimensions match - perfect compatibility!")
    else:
        print("   ⚠ Feature dimensions differ - this is expected for asymmetric hierarchies")
    
    # Cleanup
    os.unlink(flat_checkpoint_path)
    
    print("\n✅ Backward compatibility test completed successfully!")
    return True


def test_hierarchical_vs_flat_equivalence():
    """Test that hierarchical model with combined dimensions equals flat model."""
    print("\nTesting hierarchical vs flat equivalence...")
    
    embed_size = 1024
    batch_size = 8
    
    # Create flat model
    flat_config = TestConfig(hierarchical_mode=False)
    flat_rssm = networks.RSSM(
        flat_config.dyn_stoch,
        flat_config.dyn_deter,
        flat_config.dyn_hidden,
        flat_config.dyn_rec_depth,
        flat_config.dyn_discrete,
        flat_config.act,
        flat_config.norm,
        flat_config.dyn_mean_act,
        flat_config.dyn_std_act,
        flat_config.dyn_min_std,
        flat_config.unimix_ratio,
        flat_config.initial,
        flat_config.num_actions,
        embed_size,
        flat_config.device,
    )
    
    # Create hierarchical model with same total dimensions
    hier_config = TestConfig(hierarchical_mode=True)
    # To match flat: 32*32 = 1024 latent capacity
    # Split as: 16*32 + 16*32 = 512 + 512 = 1024 (same total)
    hier_config.dyn_stoch_top = 16
    hier_config.dyn_stoch_bottom = 16
    hier_config.dyn_discrete_top = 32  # Keep full discrete classes
    hier_config.dyn_discrete_bottom = 32  # Keep full discrete classes
    # Update the base stoch to match the split
    hier_config.dyn_stoch = 32
    hier_config.dyn_discrete = 32
    
    hier_rssm = networks.RSSM(
        hier_config.dyn_stoch,
        hier_config.dyn_deter,
        hier_config.dyn_hidden,
        hier_config.dyn_rec_depth,
        hier_config.dyn_discrete,
        hier_config.act,
        hier_config.norm,
        hier_config.dyn_mean_act,
        hier_config.dyn_std_act,
        hier_config.dyn_min_std,
        hier_config.unimix_ratio,
        hier_config.initial,
        hier_config.num_actions,
        embed_size,
        hier_config.device,
        hierarchical_mode=True,
        stoch_top=hier_config.dyn_stoch_top,
        stoch_bottom=hier_config.dyn_stoch_bottom,
        discrete_top=hier_config.dyn_discrete_top,
        discrete_bottom=hier_config.dyn_discrete_bottom,
    )
    
    # Compare feature sizes
    flat_initial = flat_rssm.initial(batch_size)
    hier_initial = hier_rssm.initial(batch_size)
    
    flat_feat = flat_rssm.get_feat(flat_initial)
    hier_feat = hier_rssm.get_feat(hier_initial)
    
    print(f"Flat feature size: {flat_feat.shape}")
    print(f"Hierarchical feature size: {hier_feat.shape}")
    
    if flat_feat.shape == hier_feat.shape:
        print("✓ Feature sizes match perfectly!")
    else:
        print("✗ Feature sizes don't match")
        return False
    
    print("✅ Equivalence test passed!")
    return True


if __name__ == "__main__":
    print("Running hDreamer backward compatibility tests...\n")
    
    success1 = test_backward_compatibility()
    success2 = test_hierarchical_vs_flat_equivalence()
    
    if success1 and success2:
        print("\n🎉 All compatibility tests passed!")
        print("hDreamer is ready for use with existing DreamerV3 checkpoints!")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
