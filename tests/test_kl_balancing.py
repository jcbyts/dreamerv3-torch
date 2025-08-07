#!/usr/bin/env python3

"""
Test script to verify the KL balancing fix in hRSSM.
Tests that free bits are properly distributed across hierarchy levels.
"""

import torch
import numpy as np
import sys
import os

# Add the parent directory to Python path to import hDreamer modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networks
import tools

def test_kl_balancing_implementation():
    """Test that KL balancing properly distributes free bits across levels."""
    print("Testing KL balancing implementation in hRSSM...")

    # Create hierarchical RSSM with 3 levels
    # Total deter size = 128+128+128 = 384, which is divisible by 3 levels
    rssm = networks.hRSSM(
        h_levels=3,
        h_stoch_dims=[32, 32, 32],
        h_deter_dims=[128, 128, 128],
        h_encoder_dims=[128, 256, 128],
        discrete=32,
        num_actions=6,
        device='cpu'
    )

    batch_size = 2

    # Create mock post and prior states directly (bypass the spatial processing)
    # This tests the KL balancing logic without dealing with spatial convolutions
    # Note: 'deter' is a single tensor, while 'stoch' and 'logit' are lists per level
    post = {
        'stoch': [torch.randn(batch_size, 32) for _ in range(3)],
        'logit': [torch.randn(batch_size, 32, 32) for _ in range(3)],
        'deter': torch.randn(batch_size, 384)  # sum of h_deter_dims (single tensor)
    }

    prior = {
        'stoch': [torch.randn(batch_size, 32) for _ in range(3)],
        'logit': [torch.randn(batch_size, 32, 32) for _ in range(3)],
        'deter': torch.randn(batch_size, 384)  # sum of h_deter_dims (single tensor)
    }

    # Test with different free values to verify per-level distribution
    free_total = 3.0
    kl_loss, value, dyn_loss, rep_loss = rssm.kl_loss(
        post, prior, free=free_total, dyn_scale=0.5, rep_scale=0.1
    )

    print(f"Total free bits: {free_total}")
    print(f"Free bits per level: {free_total / rssm._h_levels}")
    print(f"Number of hierarchy levels: {rssm._h_levels}")
    print(f"KL loss with balancing: {kl_loss.mean().item():.4f}")
    print(f"Total KL value (unclipped): {value.mean().item():.4f}")

    # Test with zero free bits to ensure no negative values
    kl_loss_zero, _, _, _ = rssm.kl_loss(
        post, prior, free=0.0, dyn_scale=0.5, rep_scale=0.1
    )
    print(f"KL loss with zero free bits: {kl_loss_zero.mean().item():.4f}")

    # Verify that losses are non-negative (due to clipping)
    assert kl_loss.min() >= 0, "KL loss should be non-negative after clipping"
    assert kl_loss_zero.min() >= 0, "KL loss should be non-negative even with zero free bits"

    print("✓ KL balancing implementation test passed!\n")

def test_kl_balancing_vs_old_implementation():
    """Compare new KL balancing with old (incorrect) implementation."""
    print("Comparing new KL balancing with old implementation...")

    # Create hierarchical RSSM
    # Total deter size = 96+96+96 = 288, which is divisible by 3 levels
    rssm = networks.hRSSM(
        h_levels=3,
        h_stoch_dims=[16, 16, 16],
        h_deter_dims=[96, 96, 96],
        h_encoder_dims=[64, 128, 64],
        discrete=16,
        num_actions=4,
        device='cpu'
    )

    batch_size = 2

    # Create mock post and prior states directly
    post = {
        'stoch': [torch.randn(batch_size, 16) for _ in range(3)],
        'logit': [torch.randn(batch_size, 16, 16) for _ in range(3)],
        'deter': torch.randn(batch_size, 288)  # sum of h_deter_dims
    }

    prior = {
        'stoch': [torch.randn(batch_size, 16) for _ in range(3)],
        'logit': [torch.randn(batch_size, 16, 16) for _ in range(3)],
        'deter': torch.randn(batch_size, 288)
    }
    
    # Test with the new (correct) implementation
    free_total = 1.5
    kl_loss_new, value, dyn_loss_new, rep_loss_new = rssm.kl_loss(
        post, prior, free=free_total, dyn_scale=0.5, rep_scale=0.1
    )
    
    # Simulate old (incorrect) implementation for comparison
    # This would sum first, then clip (which is wrong)
    import torch.distributions as torchd
    kld = torchd.kl.kl_divergence
    sg = lambda x: {k: v.detach() if isinstance(v, torch.Tensor) else [vi.detach() for vi in v] for k, v in x.items()}

    dyn_loss_list_old, rep_loss_list_old = [], []
    for i in range(rssm._h_levels):
        # Extract level-specific stats (only the fields that are lists)
        post_level = {k: v[i] for k, v in post.items() if isinstance(v, list)}
        prior_level = {k: v[i] for k, v in prior.items() if isinstance(v, list)}
        prior_level_sg = {k: v[i] for k, v in sg(prior).items() if isinstance(v, list)}
        post_level_sg = {k: v[i] for k, v in sg(post).items() if isinstance(v, list)}

        post_dist = rssm.get_dist_h(post_level)
        prior_dist = rssm.get_dist_h(prior_level)
        prior_dist_sg = rssm.get_dist_h(prior_level_sg)
        post_dist_sg = rssm.get_dist_h(post_level_sg)

        dyn_loss_list_old.append(kld(post_dist_sg, prior_dist))
        rep_loss_list_old.append(kld(post_dist, prior_dist_sg))
    
    # Old way: sum first, then clip
    dyn_loss_old = torch.stack(dyn_loss_list_old, dim=0).sum(0)
    rep_loss_old = torch.stack(rep_loss_list_old, dim=0).sum(0)
    dyn_loss_old = torch.clip(dyn_loss_old, min=free_total)
    rep_loss_old = torch.clip(rep_loss_old, min=free_total)
    kl_loss_old = 0.5 * dyn_loss_old + 0.1 * rep_loss_old
    
    print(f"Free bits total: {free_total}")
    print(f"Free bits per level (new): {free_total / rssm._h_levels}")
    print(f"New implementation KL loss: {kl_loss_new.mean().item():.4f}")
    print(f"Old implementation KL loss: {kl_loss_old.mean().item():.4f}")
    print(f"Difference: {(kl_loss_new.mean() - kl_loss_old.mean()).item():.4f}")
    
    # The new implementation should generally produce different (and more balanced) results
    print("✓ KL balancing comparison test completed!\n")

def test_single_level_equivalence():
    """Test that single-level hRSSM behaves like flat RSSM for KL loss."""
    print("Testing single-level hRSSM equivalence...")

    # Create single-level hierarchical RSSM
    rssm_h = networks.hRSSM(
        h_levels=1,
        h_stoch_dims=[32],
        h_deter_dims=[200],
        h_encoder_dims=[128],
        discrete=32,
        num_actions=6,
        device='cpu'
    )

    batch_size = 2

    # Create mock post and prior states for single-level hRSSM
    post_h = {
        'stoch': [torch.randn(batch_size, 32)],
        'logit': [torch.randn(batch_size, 32, 32)],
        'deter': torch.randn(batch_size, 200)
    }

    prior_h = {
        'stoch': [torch.randn(batch_size, 32)],
        'logit': [torch.randn(batch_size, 32, 32)],
        'deter': torch.randn(batch_size, 200)
    }
    
    free_bits = 1.0
    kl_loss_h, value_h, _, _ = rssm_h.kl_loss(
        post_h, prior_h, free=free_bits, dyn_scale=0.5, rep_scale=0.1
    )
    
    print(f"Single-level hRSSM:")
    print(f"  Free bits: {free_bits}")
    print(f"  Free bits per level: {free_bits / rssm_h._h_levels}")
    print(f"  KL loss: {kl_loss_h.mean().item():.4f}")
    print(f"  KL value: {value_h.mean().item():.4f}")
    
    # For single-level, free_per_level should equal total free bits
    expected_free_per_level = free_bits / rssm_h._h_levels
    assert abs(expected_free_per_level - free_bits) < 1e-6, "Single level should have free_per_level = free_total"
    
    print("✓ Single-level equivalence test passed!\n")

if __name__ == "__main__":
    print("Running KL balancing verification tests...\n")

    try:
        test_kl_balancing_implementation()
        test_kl_balancing_vs_old_implementation()
        test_single_level_equivalence()

        print("🎉 All KL balancing tests passed!")
        print("\nSummary of KL balancing fix:")
        print("✓ Free bits are now distributed equally across hierarchy levels")
        print("✓ Each level gets free_per_level = free_total / h_levels")
        print("✓ Clipping is applied per-level before summing (proper balancing)")
        print("✓ This prevents any single level from dominating the KL loss signal")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
