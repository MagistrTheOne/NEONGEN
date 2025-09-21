import torch
import torch.nn as nn
import pytest
import numpy as np
from src.models.qembed import QEmbed
from torch.nn import functional as F

class TestQEmbed:
    """Test suite for Quantum Embedding layer"""
    
    def test_initialization(self):
        """Test that QEmbed initializes correctly"""
        vocab_size = 1000
        embed_dim = 512
        
        qembed = QEmbed(vocab_size, embed_dim)
        
        assert qembed.vocab_size == vocab_size
        assert qembed.embed_dim == embed_dim
        assert isinstance(qembed.real_proj, nn.Linear)
        assert isinstance(qembed.imag_proj, nn.Linear)
        assert qembed.real_proj.in_features == vocab_size
        assert qembed.real_proj.out_features == embed_dim
        assert qembed.imag_proj.in_features == vocab_size
        assert qembed.imag_proj.out_features == embed_dim
    
    def test_forward_pass(self):
        """Test forward pass produces complex embeddings"""
        vocab_size = 100
        embed_dim = 64
        batch_size = 4
        seq_len = 10
        
        qembed = QEmbed(vocab_size, embed_dim)
        
        # Create one-hot encoded input
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        x_onehot = F.one_hot(x, num_classes=vocab_size).float()
        
        # Forward pass
        output = qembed(x_onehot)
        
        # Check output shape and type
        assert output.shape == (batch_size, seq_len, embed_dim)
        assert output.dtype == torch.complex64
        
        # Check that real and imaginary parts exist
        assert torch.any(output.real != 0)
        assert torch.any(output.imag != 0)
    
    def test_normalization(self):
        """Test that embeddings are properly normalized"""
        vocab_size = 50
        embed_dim = 32
        
        qembed = QEmbed(vocab_size, embed_dim)
        
        # Test input
        x = torch.eye(vocab_size).unsqueeze(0)  # batch of one-hot vectors
        
        output = qembed(x)
        
        # Check L2 normalization
        norms = torch.norm(output, p=2, dim=-1)
        
        # Should be approximately 1 (allowing for numerical precision)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    def test_amplitudes(self):
        """Test probability amplitudes calculation"""
        vocab_size = 100
        embed_dim = 64
        
        qembed = QEmbed(vocab_size, embed_dim)
        
        x = F.one_hot(torch.randint(0, vocab_size, (5,)), num_classes=vocab_size).float().unsqueeze(0)
        
        amplitudes = qembed.get_amplitudes(x)
        
        # Amplitudes should be probabilities (sum to 1)
        prob_sum = amplitudes.sum(dim=-1)
        assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-6)
        
        # Amplitudes should be non-negative
        assert torch.all(amplitudes >= 0)
    
    def test_phase_calculation(self):
        """Test phase angle calculation"""
        vocab_size = 50
        embed_dim = 32
        
        qembed = QEmbed(vocab_size, embed_dim)
        
        x = torch.eye(vocab_size).unsqueeze(0)
        
        phases = qembed.get_phase(x)
        
        # Phases should be in range [-pi, pi]
        assert torch.all(phases >= -torch.pi)
        assert torch.all(phases <= torch.pi)
        
        # Phases should be real numbers
        assert phases.dtype == torch.float32
    
    def test_gradient_flow(self):
        """Test that gradients flow through the layer"""
        vocab_size = 100
        embed_dim = 64
        
        qembed = QEmbed(vocab_size, embed_dim)
        
        x = F.one_hot(torch.randint(0, vocab_size, (3,)), num_classes=vocab_size).float().unsqueeze(0)
        x.requires_grad = True
        
        output = qembed(x)
        
        # Create a simple loss
        loss = torch.abs(output).sum()
        loss.backward()
        
        # Check that gradients are computed
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
    
    def test_different_batch_sizes(self):
        """Test with different batch sizes"""
        vocab_size = 100
        embed_dim = 64
        
        qembed = QEmbed(vocab_size, embed_dim)
        
        # Test various batch sizes
        for batch_size in [1, 4, 16]:
            x = torch.randint(0, vocab_size, (batch_size, 8))
            x_onehot = F.one_hot(x, num_classes=vocab_size).float()
            
            output = qembed(x_onehot)
            
            assert output.shape == (batch_size, 8, embed_dim)
            assert output.dtype == torch.complex64

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
