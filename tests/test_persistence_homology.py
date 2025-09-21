#!/usr/bin/env python3
"""
Comprehensive tests for persistence homology implementation
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.topofilter import TopoFilter


class TestPersistenceHomology:
    """Test suite for persistence homology implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.topofilter = TopoFilter(
            persistence_threshold=0.1,
            tau=1.0,
            max_dimension=1,
            use_gudhi=True
        )
        
        # Create test embeddings
        self.test_embeddings = torch.complex(
            torch.randn(2, 10, 32),
            torch.randn(2, 10, 32)
        )
    
    def test_gudhi_availability(self):
        """Test GUDHI availability detection"""
        # Test with GUDHI available
        with patch('models.topofilter.GUDHI_AVAILABLE', True):
            topofilter = TopoFilter(use_gudhi=True)
            assert topofilter.use_gudhi == True
        
        # Test with GUDHI not available
        with patch('models.topofilter.GUDHI_AVAILABLE', False):
            topofilter = TopoFilter(use_gudhi=True)
            assert topofilter.use_gudhi == False
    
    def test_fallback_persistence_computation(self):
        """Test fallback persistence computation"""
        embeddings = np.random.randn(10, 32)
        scores = self.topofilter._approximate_persistence_fallback(embeddings)
        
        assert len(scores) == 10
        assert np.all(scores >= 0)
        assert not np.any(np.isnan(scores))
    
    @patch('models.topofilter.gudhi')
    def test_real_persistence_homology_success(self, mock_gudhi):
        """Test successful real persistence homology computation"""
        # Mock GUDHI components
        mock_rips = MagicMock()
        mock_simplex_tree = MagicMock()
        mock_persistence = [
            (0, (0.0, 1.0)),  # Connected component
            (0, (0.5, 2.0)),  # Another component
            (1, (1.0, 3.0))   # 1-dimensional feature
        ]
        
        mock_simplex_tree.persistence.return_value = mock_persistence
        mock_rips.create_simplex_tree.return_value = mock_simplex_tree
        mock_gudhi.RipsComplex.return_value = mock_rips
        
        embeddings = np.random.randn(5, 32)
        scores = self.topofilter.compute_real_persistence_homology(embeddings)
        
        assert len(scores) == 5
        assert np.all(scores >= 0)
        assert not np.any(np.isnan(scores))
    
    @patch('models.topofilter.gudhi')
    def test_real_persistence_homology_failure(self, mock_gudhi):
        """Test fallback when GUDHI computation fails"""
        # Mock GUDHI to raise exception
        mock_gudhi.RipsComplex.side_effect = Exception("GUDHI error")
        
        embeddings = np.random.randn(5, 32)
        scores = self.topofilter.compute_real_persistence_homology(embeddings)
        
        # Should fall back to approximate method
        assert len(scores) == 5
        assert np.all(scores >= 0)
    
    def test_compute_persistence_weights(self):
        """Test persistence weights computation"""
        weights = self.topofilter.compute_persistence_weights(self.test_embeddings)
        
        assert weights.shape == (2, 10)  # batch_size, seq_len
        assert torch.all(weights >= 0)
        assert torch.all(weights <= 1)  # exp(-x) is always <= 1
    
    def test_forward_pass(self):
        """Test forward pass through TopoFilter"""
        filtered = self.topofilter.forward(self.test_embeddings)
        
        assert filtered.shape == self.test_embeddings.shape
        assert filtered.dtype == torch.complex64
    
    def test_get_persistence_scores(self):
        """Test getting persistence scores"""
        scores = self.topofilter.get_persistence_scores(self.test_embeddings)
        
        assert scores.shape == (2, 10)  # batch_size, seq_len
        assert torch.all(scores >= 0)
    
    def test_get_filter_mask(self):
        """Test getting filter mask"""
        mask = self.topofilter.get_filter_mask(self.test_embeddings)
        
        assert mask.shape == (2, 10)  # batch_size, seq_len
        assert torch.all((mask == 0) | (mask == 1))  # binary mask
    
    def test_persistence_threshold_filtering(self):
        """Test that persistence threshold filtering works"""
        # Create embeddings with known persistence scores
        embeddings = torch.complex(
            torch.randn(1, 5, 32),
            torch.randn(1, 5, 32)
        )
        
        # Mock persistence scores: some above threshold, some below
        mock_scores = torch.tensor([[0.05, 0.15, 0.08, 0.20, 0.03]])
        
        with patch.object(self.topofilter, 'get_persistence_scores', return_value=mock_scores):
            mask = self.topofilter.get_filter_mask(embeddings)
            
            # Points with persistence > 0.1 should be kept (mask = 1)
            # Points with persistence <= 0.1 should be filtered (mask = 0)
            expected_mask = torch.tensor([[0, 1, 0, 1, 0]])
            assert torch.allclose(mask, expected_mask)
    
    def test_tau_parameter_effect(self):
        """Test that tau parameter affects weight computation"""
        embeddings = torch.complex(
            torch.randn(1, 3, 32),
            torch.randn(1, 3, 32)
        )
        
        # Test with different tau values
        topofilter_low_tau = TopoFilter(tau=0.1)
        topofilter_high_tau = TopoFilter(tau=10.0)
        
        weights_low = topofilter_low_tau.compute_persistence_weights(embeddings)
        weights_high = topofilter_high_tau.compute_persistence_weights(embeddings)
        
        # Low tau should create more extreme weights
        # High tau should create more uniform weights
        assert torch.std(weights_low) > torch.std(weights_high)
    
    def test_numerical_stability(self):
        """Test numerical stability with edge cases"""
        # Test with very small embeddings
        small_embeddings = torch.complex(
            torch.randn(1, 2, 32) * 1e-6,
            torch.randn(1, 2, 32) * 1e-6
        )
        
        filtered = self.topofilter.forward(small_embeddings)
        assert not torch.any(torch.isnan(filtered))
        assert not torch.any(torch.isinf(filtered))
        
        # Test with very large embeddings
        large_embeddings = torch.complex(
            torch.randn(1, 2, 32) * 1e6,
            torch.randn(1, 2, 32) * 1e6
        )
        
        filtered = self.topofilter.forward(large_embeddings)
        assert not torch.any(torch.isnan(filtered))
        assert not torch.any(torch.isinf(filtered))
    
    def test_batch_processing(self):
        """Test processing multiple batches"""
        batch_sizes = [1, 2, 4, 8]
        seq_lens = [5, 10, 20]
        
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                embeddings = torch.complex(
                    torch.randn(batch_size, seq_len, 32),
                    torch.randn(batch_size, seq_len, 32)
                )
                
                filtered = self.topofilter.forward(embeddings)
                assert filtered.shape == embeddings.shape
                
                scores = self.topofilter.get_persistence_scores(embeddings)
                assert scores.shape == (batch_size, seq_len)
                
                mask = self.topofilter.get_filter_mask(embeddings)
                assert mask.shape == (batch_size, seq_len)


class TestTopoFilterIntegration:
    """Integration tests for TopoFilter with other components"""
    
    def test_with_qembed(self):
        """Test TopoFilter integration with QEmbed"""
        from models.qembed import QEmbed
        
        qembed = QEmbed(vocab_size=100, embed_dim=32)
        topofilter = TopoFilter()
        
        # Create one-hot input
        input_tokens = torch.eye(100).unsqueeze(0)  # (1, 100, 100)
        
        # Get quantum embeddings
        quantum_embeddings = qembed(input_tokens)
        
        # Apply topological filtering
        filtered_embeddings = topofilter(quantum_embeddings)
        
        assert filtered_embeddings.shape == quantum_embeddings.shape
        assert filtered_embeddings.dtype == torch.complex64
    
    def test_with_geoattention(self):
        """Test TopoFilter integration with GeoAttention"""
        from models.geoattention import GeoAttention
        
        topofilter = TopoFilter()
        geo_attention = GeoAttention(dim=32, num_heads=4)
        
        # Create test embeddings
        embeddings = torch.complex(
            torch.randn(1, 10, 32),
            torch.randn(1, 10, 32)
        )
        
        # Apply topological filtering
        filtered = topofilter(embeddings)
        
        # Apply geometric attention
        attended = geo_attention(filtered)
        
        assert attended.shape == embeddings.shape
        assert attended.dtype == torch.complex64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
