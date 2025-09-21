import torch
import torch.nn as nn
import pytest
import numpy as np
from src.models.topofilter import TopoFilter

class TestTopoFilter:
    """Test suite for Topological Filtering layer"""

    def test_initialization(self):
        """Test that TopoFilter initializes correctly"""
        persistence_threshold = 0.1
        tau = 1.0

        topofilter = TopoFilter(persistence_threshold, tau)

        assert topofilter.persistence_threshold == persistence_threshold
        assert topofilter.tau == tau

    def test_approximate_persistence(self):
        """Test approximate persistence computation"""
        topofilter = TopoFilter()

        # Create test distance matrix
        batch_size = 8
        distances = torch.rand(batch_size, batch_size)

        persistence_scores = topofilter.approximate_persistence(distances)

        # Check output shape
        assert persistence_scores.shape == (batch_size,)
        assert persistence_scores.dtype == distances.dtype

        # All scores should be non-negative
        assert torch.all(persistence_scores >= 0)

    def test_compute_persistence_weights(self):
        """Test persistence weight computation"""
        topofilter = TopoFilter(persistence_threshold=0.1, tau=1.0)

        # Create test embeddings
        batch_size = 6
        embed_dim = 64
        embeddings = torch.complex(
            torch.randn(batch_size, embed_dim),
            torch.randn(batch_size, embed_dim)
        )

        weights = topofilter.compute_persistence_weights(embeddings)

        # Check output shape
        assert weights.shape == (batch_size,)
        assert weights.dtype == embeddings.dtype

        # Weights should be in [0, 1] range due to exp(-x) and thresholding
        assert torch.all(weights >= 0)
        assert torch.all(weights <= 1)

    def test_forward_pass(self):
        """Test forward pass of topological filtering"""
        topofilter = TopoFilter()

        # Create test quantum embeddings
        batch_size = 4
        seq_len = 10
        embed_dim = 64

        quantum_embed = torch.complex(
            torch.randn(batch_size, seq_len, embed_dim),
            torch.randn(batch_size, seq_len, embed_dim)
        )

        output = topofilter(quantum_embed)

        # Check output shape and type
        assert output.shape == quantum_embed.shape
        assert output.dtype == quantum_embed.dtype

        # Output should be different from input (filtered)
        assert not torch.allclose(output, quantum_embed, atol=1e-6)

    def test_get_persistence_scores(self):
        """Test persistence scores retrieval"""
        topofilter = TopoFilter()

        batch_size = 2
        seq_len = 8
        embed_dim = 32

        quantum_embed = torch.complex(
            torch.randn(batch_size, seq_len, embed_dim),
            torch.randn(batch_size, seq_len, embed_dim)
        )

        scores = topofilter.get_persistence_scores(quantum_embed)

        # Check output shape
        assert scores.shape == (batch_size, seq_len)
        assert scores.dtype == torch.float32  # Scores are real-valued

        # Scores should be non-negative
        assert torch.all(scores >= 0)

    def test_get_filter_mask(self):
        """Test filter mask generation"""
        persistence_threshold = 0.2
        topofilter = TopoFilter(persistence_threshold=persistence_threshold)

        batch_size = 3
        seq_len = 6
        embed_dim = 32

        quantum_embed = torch.complex(
            torch.randn(batch_size, seq_len, embed_dim),
            torch.randn(batch_size, seq_len, embed_dim)
        )

        mask = topofilter.get_filter_mask(quantum_embed)

        # Check output shape and type
        assert mask.shape == (batch_size, seq_len)
        assert mask.dtype == torch.float32

        # Mask values should be 0 or 1
        assert torch.all((mask == 0) | (mask == 1))

    def test_different_thresholds(self):
        """Test different persistence thresholds"""
        thresholds = [0.0, 0.1, 0.5, 1.0]

        batch_size = 5
        seq_len = 4
        embed_dim = 32

        quantum_embed = torch.complex(
            torch.randn(batch_size, seq_len, embed_dim),
            torch.randn(batch_size, seq_len, embed_dim)
        )

        for threshold in thresholds:
            topofilter = TopoFilter(persistence_threshold=threshold)
            output = topofilter(quantum_embed)
            mask = topofilter.get_filter_mask(quantum_embed)

            # Check that output shape is preserved
            assert output.shape == quantum_embed.shape
            assert mask.shape == (batch_size, seq_len)

    def test_gradient_flow(self):
        """Test that gradients flow through topological filtering"""
        topofilter = TopoFilter()

        batch_size = 2
        seq_len = 6
        embed_dim = 32

        quantum_embed = torch.complex(
            torch.randn(batch_size, seq_len, embed_dim),
            torch.randn(batch_size, seq_len, embed_dim)
        )

        # Make input require gradients
        quantum_embed.requires_grad = True

        output = topofilter(quantum_embed)

        # Create a loss function
        loss = torch.abs(output).sum()
        loss.backward()

        # Check that gradients are computed
        assert quantum_embed.grad is not None
        assert not torch.all(quantum_embed.grad == 0)

    def test_memory_efficiency(self):
        """Test memory efficiency with different batch sizes"""
        topofilter = TopoFilter()

        embed_dim = 64

        # Test various configurations
        configs = [
            (1, 512, embed_dim),   # Small batch, long sequence
            (8, 128, embed_dim),   # Medium batch and sequence
            (32, 32, embed_dim),   # Large batch, short sequence
        ]

        for batch_size, seq_len, embed_dim in configs:
            quantum_embed = torch.complex(
                torch.randn(batch_size, seq_len, embed_dim),
                torch.randn(batch_size, seq_len, embed_dim)
            )

            output = topofilter(quantum_embed)

            assert output.shape == quantum_embed.shape
            assert output.dtype == quantum_embed.dtype

    def test_consistency(self):
        """Test that results are consistent for same input"""
        topofilter = TopoFilter()

        batch_size = 3
        seq_len = 8
        embed_dim = 32

        quantum_embed = torch.complex(
            torch.randn(batch_size, seq_len, embed_dim),
            torch.randn(batch_size, seq_len, embed_dim)
        )

        # Run multiple times
        output1 = topofilter(quantum_embed)
        output2 = topofilter(quantum_embed)

        # Results should be identical for same input
        assert torch.allclose(output1, output2, atol=1e-6)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
