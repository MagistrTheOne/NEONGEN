import torch
import torch.nn as nn
import torch.nn.functional as F

class QEmbed(nn.Module):
    """
    Quantum-inspired Embedding Layer with complex number support
    Implements amplitude encoding for token embeddings
    
    Args:
        vocab_size (int): Size of vocabulary
        embed_dim (int): Dimension of complex embedding space
    """
    
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Separate projections for real and imaginary parts
        self.real_proj = nn.Linear(vocab_size, embed_dim)
        self.imag_proj = nn.Linear(vocab_size, embed_dim)
        
        # Initialize with small complex values
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with small complex values"""
        # Xavier initialization for both real and imaginary parts
        nn.init.xavier_uniform_(self.real_proj.weight)
        nn.init.xavier_uniform_(self.imag_proj.weight)
        
        # Initialize biases to small values
        nn.init.constant_(self.real_proj.bias, 0.0)
        nn.init.constant_(self.imag_proj.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for quantum embedding
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, vocab_size)
            or (batch_size, vocab_size) for single tokens
            
        Returns:
            torch.Tensor: Complex embeddings of shape (batch_size, seq_len, embed_dim)
        """
        # Ensure input is in the right format
        if x.dim() == 2:
            # Single token case - add sequence dimension
            x = x.unsqueeze(1)
        
        # Project to real and imaginary components
        real_part = self.real_proj(x)
        imag_part = self.imag_proj(x)
        
        # Create complex tensor
        complex_embed = torch.complex(real_part, imag_part)
        
        # Normalize to maintain probabilistic interpretation
        # L2 normalization along embedding dimension
        norm = torch.norm(complex_embed, p=2, dim=-1, keepdim=True)
        # Avoid division by zero
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        
        normalized_embed = complex_embed / norm
        
        return normalized_embed
    
    def get_amplitudes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability amplitudes from embeddings
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Probability amplitudes |ψ|²
        """
        complex_embed = self.forward(x)
        # |ψ|² = real² + imag²
        probabilities = torch.abs(complex_embed) ** 2
        return probabilities
    
    def get_phase(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get phase information from embeddings
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Phase angles in radians
        """
        complex_embed = self.forward(x)
        # Phase = arctan2(imag, real)
        phase = torch.angle(complex_embed)
        return phase
