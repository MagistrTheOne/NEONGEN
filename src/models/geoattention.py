import torch
import torch.nn as nn
import torch.nn.functional as F

class GeoAttention(nn.Module):
    """
    Geometric Attention Layer using Fisher-Rao distance
    Implements attention mechanism based on statistical distances
    
    Args:
        dim (int): Dimension of input embeddings
        num_heads (int): Number of attention heads
        gamma (float): Scaling parameter for distance (default: 1.0)
    """
    
    def __init__(self, dim: int, num_heads: int, gamma: float = 1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.gamma = gamma
        self.head_dim = dim // num_heads
        
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        # Projection layers for converting complex to real representations
        self.q_proj = nn.Linear(dim * 2, dim)  # complex (real+imag) -> real
        self.k_proj = nn.Linear(dim * 2, dim)
        self.v_proj = nn.Linear(dim * 2, dim)
        self.out_proj = nn.Linear(dim, dim * 2)  # real -> complex (real+imag)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize projection weights"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        nn.init.constant_(self.q_proj.bias, 0.0)
        nn.init.constant_(self.k_proj.bias, 0.0)
        nn.init.constant_(self.v_proj.bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)
    
    def fisher_rao_distance(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Compute Fisher-Rao distance using Bhattacharyya approximation
        
        Args:
            p (torch.Tensor): Probability distribution 1
            q (torch.Tensor): Probability distribution 2
            
        Returns:
            torch.Tensor: Fisher-Rao distance between distributions
        """
        # Ensure inputs are probability distributions
        p_norm = F.softmax(p, dim=-1)
        q_norm = F.softmax(q, dim=-1)
        
        # Bhattacharyya coefficient: Σ √(p_i * q_i)
        bc_coeff = torch.sum(torch.sqrt(p_norm * q_norm), dim=-1)
        
        # Clamp to avoid numerical issues with acos
        bc_coeff = torch.clamp(bc_coeff, -1.0, 1.0)
        
        # Fisher-Rao distance: arccos(Bhattacharyya coefficient)
        distance = torch.acos(bc_coeff)
        
        return distance
    
    def forward(self, quantum_embed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for geometric attention
        
        Args:
            quantum_embed (torch.Tensor): Complex embeddings from previous layers
            
        Returns:
            torch.Tensor: Attended complex embeddings
        """
        batch_size, seq_len, embed_dim = quantum_embed.shape
        
        # Convert complex to real representation (concat real and imag parts)
        real_embed = torch.cat([quantum_embed.real, quantum_embed.imag], dim=-1)
        
        # Project to query, key, value spaces
        Q = self.q_proj(real_embed)  # (batch_size, seq_len, dim)
        K = self.k_proj(real_embed)  # (batch_size, seq_len, dim)
        V = self.v_proj(real_embed)  # (batch_size, seq_len, dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute Fisher-Rao distances between all query-key pairs
        # Q shape: (batch_size, num_heads, seq_len, head_dim)
        # K shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Expand dimensions for pairwise computation
        Q_expanded = Q.unsqueeze(3)  # (batch, heads, q_len, 1, head_dim)
        K_expanded = K.unsqueeze(2)  # (batch, heads, 1, k_len, head_dim)
        
        # Compute distances for all pairs
        distances = self.fisher_rao_distance(Q_expanded, K_expanded)
        
        # Compute attention weights: softmax(-gamma * distance^2)
        attention_scores = -self.gamma * distances.pow(2)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Reshape back to original dimensions
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.dim
        )
        
        # Project back to complex representation
        output = self.out_proj(attended)
        
        # Split into real and imaginary parts
        real_out, imag_out = torch.chunk(output, 2, dim=-1)
        
        # Convert back to complex tensor
        complex_output = torch.complex(real_out, imag_out)
        
        return complex_output
    
    def get_attention_map(self, quantum_embed: torch.Tensor) -> torch.Tensor:
        """
        Get raw attention map for analysis
        
        Args:
            quantum_embed (torch.Tensor): Complex embeddings
            
        Returns:
            torch.Tensor: Attention weights matrix
        """
        batch_size, seq_len, embed_dim = quantum_embed.shape
        real_embed = torch.cat([quantum_embed.real, quantum_embed.imag], dim=-1)
        
        Q = self.q_proj(real_embed)
        K = self.k_proj(real_embed)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute distances
        Q_expanded = Q.unsqueeze(3)
        K_expanded = K.unsqueeze(2)
        distances = self.fisher_rao_distance(Q_expanded, K_expanded)
        
        # Compute attention weights
        attention_scores = -self.gamma * distances.pow(2)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        return attention_weights
