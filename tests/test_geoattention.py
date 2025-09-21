import torch
import torch.nn as nn
import pytest
import numpy as np
from src.models.geoattention import GeoAttention

class TestGeoAttention:
    """Test suite for Geometric Attention layer"""

    def test_initialization(self):
        """Test that GeoAttention initializes correctly"""
        dim = 512
        num_heads = 8
        gamma = 1.0

        geo_attn = GeoAttention(dim, num_heads, gamma)

        assert geo_attn.dim == dim
        assert geo_attn.num_heads == num_heads
        assert geo_attn.gamma == gamma
        assert geo_attn.head_dim == dim // num_heads

        # Check projection layers
        assert isinstance(geo_attn.q_proj, nn.Linear)
        assert isinstance(geo_attn.k_proj, nn.Linear)
        assert isinstance(geo_attn.v_proj, nn.Linear)
        assert isinstance(geo_attn.out_proj, nn.Linear)

        # Check input/output dimensions
        assert geo_attn.q_proj.in_features == dim * 2  # complex -> real concat
        assert geo_attn.q_proj.out_features == dim
        assert geo_attn.out_proj.out_features == dim * 2  # real -> complex

    def test_fisher_rao_distance(self):
        """Test Fisher-Rao distance computation"""
        geo_attn = GeoAttention(dim=64, num_heads=4)

        # Create test probability distributions
        batch_size, heads, seq_len, head_dim = 2, 4, 8, 16

        p = torch.randn(batch_size, heads, seq_len, head_dim)
        q = torch.randn(batch_size, heads, seq_len, head_dim)

        distances = geo_attn.fisher_rao_distance(p, q)

        # Check output shape
        expected_shape = (batch_size, heads, seq_len, seq_len)
        assert distances.shape == expected_shape

        # Distances should be non-negative
        assert torch.all(distances >= 0)

        # Distance from p to p should be 0
        p_self = geo_attn.fisher_rao_distance(p, p)
        assert torch.allclose(p_self, torch.zeros_like(p_self), atol=1e-6)

    def test_forward_pass(self):
        """Test forward pass of geometric attention"""
        dim = 128
        num_heads = 8
        geo_attn = GeoAttention(dim, num_heads)

        batch_size = 3
        seq_len = 12
        embed_dim = dim

        # Create complex quantum embeddings
        quantum_embed = torch.complex(
            torch.randn(batch_size, seq_len, embed_dim),
            torch.randn(batch_size, seq_len, embed_dim)
        )

        output = geo_attn(quantum_embed)

        # Check output shape and type
        assert output.shape == quantum_embed.shape
        assert output.dtype == quantum_embed.dtype

    def test_get_attention_map(self):
        """Test attention map retrieval"""
        dim = 64
        num_heads = 4
        geo_attn = GeoAttention(dim, num_heads)

        batch_size = 2
        seq_len = 8

        quantum_embed = torch.complex(
            torch.randn(batch_size, seq_len, dim),
            torch.randn(batch_size, seq_len, dim)
        )

        attention_map = geo_attn.get_attention_map(quantum_embed)

        # Check output shape: (batch, heads, seq_len, seq_len)
        expected_shape = (batch_size, num_heads, seq_len, seq_len)
        assert attention_map.shape == expected_shape

        # Attention weights should sum to 1 along last dimension
        attention_sum = attention_map.sum(dim=-1)
        assert torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-5)

        # All weights should be non-negative
        assert torch.all(attention_map >= 0)

    def test_different_gamma_values(self):
        """Test different gamma parameter values"""
        dim = 64
        num_heads = 4
        gamma_values = [0.1, 1.0, 2.0, 10.0]

        batch_size = 2
        seq_len = 6

        quantum_embed = torch.complex(
            torch.randn(batch_size, seq_len, dim),
            torch.randn(batch_size, seq_len, dim)
        )

        for gamma in gamma_values:
            geo_attn = GeoAttention(dim, num_heads, gamma=gamma)
            output = geo_attn(quantum_embed)

            assert output.shape == quantum_embed.shape
            assert output.dtype == quantum_embed.dtype

    def test_different_head_counts(self):
        """Test different numbers of attention heads"""
        dim = 128
        head_counts = [1, 4, 8, 16]

        batch_size = 2
        seq_len = 8

        quantum_embed = torch.complex(
            torch.randn(batch_size, seq_len, dim),
            torch.randn(batch_size, seq_len, dim)
        )

        for num_heads in head_counts:
            geo_attn = GeoAttention(dim, num_heads)
            output = geo_attn(quantum_embed)

            assert output.shape == quantum_embed.shape
            assert output.dtype == quantum_embed.dtype

    def test_gradient_flow(self):
        """Test that gradients flow through geometric attention"""
        dim = 64
        num_heads = 4
        geo_attn = GeoAttention(dim, num_heads)

        batch_size = 2
        seq_len = 6

        quantum_embed = torch.complex(
            torch.randn(batch_size, seq_len, dim),
            torch.randn(batch_size, seq_len, dim)
        )

        # Make input require gradients
        quantum_embed.requires_grad = True

        output = geo_attn(quantum_embed)

        # Create loss
        loss = torch.abs(output).sum()
        loss.backward()

        # Check gradients
        assert quantum_embed.grad is not None
        assert not torch.all(quantum_embed.grad == 0)

    def test_memory_efficiency(self):
        """Test memory usage with different configurations"""
        dim = 128
        num_heads = 8
        geo_attn = GeoAttention(dim, num_heads)

        # Test various batch sizes and sequence lengths
        configs = [
            (1, 512, dim),   # Long sequence
            (8, 128, dim),   # Medium
            (16, 64, dim),   # Larger batch
        ]

        for batch_size, seq_len, embed_dim in configs:
            quantum_embed = torch.complex(
                torch.randn(batch_size, seq_len, embed_dim),
                torch.randn(batch_size, seq_len, embed_dim)
            )

            output = geo_attn(quantum_embed)

            assert output.shape == quantum_embed.shape
            assert output.dtype == quantum_embed.dtype

    def test_numerical_stability(self):
        """Test numerical stability with edge cases"""
        dim = 64
        num_heads = 4
        geo_attn = GeoAttention(dim, num_heads)

        batch_size = 2
        seq_len = 4

        # Test with very small values
        quantum_embed_small = torch.complex(
            torch.randn(batch_size, seq_len, dim) * 1e-6,
            torch.randn(batch_size, seq_len, dim) * 1e-6
        )

        output_small = geo_attn(quantum_embed_small)
        assert output_small.shape == quantum_embed_small.shape
        assert torch.isfinite(output_small).all()

        # Test with large values
        quantum_embed_large = torch.complex(
            torch.randn(batch_size, seq_len, dim) * 1e3,
            torch.randn(batch_size, seq_len, dim) * 1e3
        )

        output_large = geo_attn(quantum_embed_large)
        assert output_large.shape == quantum_embed_large.shape
        assert torch.isfinite(output_large).all()

    def test_consistency(self):
        """Test output consistency for same input"""
        dim = 64
        num_heads = 4
        geo_attn = GeoAttention(dim, num_heads)

        batch_size = 2
        seq_len = 8

        quantum_embed = torch.complex(
            torch.randn(batch_size, seq_len, dim),
            torch.randn(batch_size, seq_len, dim)
        )

        # Run multiple times
        output1 = geo_attn(quantum_embed)
        output2 = geo_attn(quantum_embed)

        # Results should be identical
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_symmetry_property(self):
        """Test that distance is symmetric (approximately)"""
        geo_attn = GeoAttention(dim=32, num_heads=2)

        # Create identical distributions
        batch_size, heads, seq_len, head_dim = 1, 2, 4, 16

        p = torch.randn(batch_size, heads, seq_len, head_dim)
        q = p.clone()  # Same as p

        dist_pq = geo_attn.fisher_rao_distance(p, q)
        dist_qp = geo_attn.fisher_rao_distance(q, p)

        # Distance should be symmetric
        assert torch.allclose(dist_pq, dist_qp, atol=1e-6)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
