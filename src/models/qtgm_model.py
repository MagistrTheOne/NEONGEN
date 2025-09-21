import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from .qembed import QEmbed
from .topofilter import TopoFilter
from .geoattention import GeoAttention

class QTGTransformerBlock(nn.Module):
    """
    Single QTG Transformer block with quantum embeddings,
    topological filtering, and geometric attention
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        persistence_threshold: float = 0.1,
        tau: float = 1.0,
        gamma: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # QTG Components - use simple linear transformation for internal processing
        self.qembed_proj = nn.Linear(embed_dim * 2, embed_dim * 2)  # Real to complex projection
        self.topofilter = TopoFilter(persistence_threshold, tau)
        self.geo_attention = GeoAttention(embed_dim, num_heads, gamma)

        # Feed-forward network (works with real representation)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 4),  # complex -> real expansion
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim * 2)   # back to complex
        )

        # Custom normalization for complex numbers
        self.norm1 = self._create_complex_norm(embed_dim)
        self.norm2 = self._create_complex_norm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def _create_complex_norm(self, embed_dim: int) -> nn.Module:
        """Create normalization layer that works with complex numbers"""
        return nn.LayerNorm(embed_dim * 2)  # Normalize concatenated real/imag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through QTG transformer block

        Args:
            x (torch.Tensor): Input complex embeddings (batch, seq, embed_dim)

        Returns:
            torch.Tensor: Processed complex embeddings
        """
        batch_size, seq_len, embed_dim = x.shape

        # Convert complex to real representation for processing
        x_real = torch.cat([x.real, x.imag], dim=-1)

        # Apply dropout to real representation (dropout doesn't work with complex)
        x_real = self.dropout(x_real)

        # Multi-head QTG attention with residual connection
        # 1. Quantum embedding projection (real -> complex representation)
        q_embed_real = self.qembed_proj(x_real)  # (batch, seq, embed_dim*2) real
        q_embed = torch.complex(
            q_embed_real[:, :, :embed_dim],
            q_embed_real[:, :, embed_dim:]
        )  # Convert to complex

        # 2. Topological filtering
        filtered = self.topofilter(q_embed)  # (batch, seq, embed_dim) complex

        # 3. Geometric attention
        attended = self.geo_attention(filtered)  # (batch, seq, embed_dim) complex

        # Residual connection
        attended_real = torch.cat([attended.real, attended.imag], dim=-1)
        x_norm = self.norm1(x_real)
        attended_norm = attended_real + x_norm

        # Feed-forward with residual connection
        ff_input = attended_norm.view(batch_size * seq_len, -1)
        ff_output = self.feed_forward(ff_input)
        ff_output = ff_output.view(batch_size, seq_len, -1)

        # Residual connection
        ff_norm = self.norm2(attended_norm)
        output = ff_output + ff_norm

        # Convert back to complex
        real_out, imag_out = torch.chunk(output, 2, dim=-1)
        complex_output = torch.complex(real_out, imag_out)

        return complex_output


class QTGModel(nn.Module):
    """
    Full Quantum-Topological-Geometric Fusion Model
    Integrates QEmbed, TopoFilter, and GeoAttention in transformer architecture
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        max_seq_len: int = 512,
        persistence_threshold: float = 0.1,
        tau: float = 1.0,
        gamma: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # QTG Transformer blocks
        self.blocks = nn.ModuleList([
            QTGTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                persistence_threshold=persistence_threshold,
                tau=tau,
                gamma=gamma,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Language model head
        self.lm_head = nn.Linear(embed_dim * 2, vocab_size)  # complex -> vocab

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through QTG model

        Args:
            input_ids (torch.Tensor): Token indices (batch_size, seq_len)
            attention_mask (torch.Tensor, optional): Attention mask

        Returns:
            Dict[str, torch.Tensor]: Model outputs including logits and attentions
        """
        batch_size, seq_len = input_ids.shape

        # Get token and position embeddings
        token_embed = self.token_embedding(input_ids)  # (batch, seq, embed_dim)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_embed = self.position_embedding(positions)  # (batch, seq, embed_dim)

        # Combine embeddings
        x = token_embed + pos_embed  # (batch, seq, embed_dim) real

        # Convert to complex representation
        # Initialize with zero imaginary part
        x_complex = torch.complex(x, torch.zeros_like(x))

        # Apply dropout to real representation (dropout doesn't work with complex)
        x_real = torch.cat([x_complex.real, x_complex.imag], dim=-1)
        x_real = self.dropout(x_real)

        # Convert back to complex
        x_complex = torch.complex(x_real[:, :, :x_real.size(-1)//2],
                                x_real[:, :, x_real.size(-1)//2:])

        # Store attention maps for analysis
        attention_maps = []

        # Pass through QTG transformer blocks
        for block in self.blocks:
            x_complex = block(x_complex)

            # Store attention map from last layer
            if hasattr(block.geo_attention, 'get_attention_map'):
                attn_map = block.geo_attention.get_attention_map(x_complex)
                attention_maps.append(attn_map)

        # Convert complex output to real for language modeling
        x_real = torch.cat([x_complex.real, x_complex.imag], dim=-1)

        # Language model head
        logits = self.lm_head(x_real)  # (batch, seq, vocab_size)

        # Apply attention mask if provided
        if attention_mask is not None:
            logits = logits.masked_fill(attention_mask.unsqueeze(-1) == 0, float('-inf'))

        return {
            'logits': logits,
            'last_hidden_state': x_complex,
            'attention_maps': attention_maps
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Generate text using QTG model

        Args:
            input_ids (torch.Tensor): Input token sequence
            max_new_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_k (int, optional): Top-k sampling
            top_p (float, optional): Top-p sampling
            do_sample (bool): Whether to sample or greedy decode

        Returns:
            torch.Tensor: Generated token sequence
        """
        self.eval()

        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Get logits for next token
            with torch.no_grad():
                outputs = self(generated)
                next_token_logits = outputs['logits'][:, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply top-k sampling
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(-1, top_k_indices, top_k_logits)

            # Apply top-p sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens with cumulative probability above top_p
                sorted_logits[cumulative_probs > top_p] = float('-inf')
                next_token_logits.scatter_(-1, sorted_indices, sorted_logits)

            # Sample next token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def get_topology_info(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get topological information for analysis

        Args:
            input_ids (torch.Tensor): Input tokens

        Returns:
            Dict[str, torch.Tensor]: Topological analysis results
        """
        self.eval()

        with torch.no_grad():
            # Get embeddings
            token_embed = self.token_embedding(input_ids)
            positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
            pos_embed = self.position_embedding(positions)
            x = token_embed + pos_embed
            x_complex = torch.complex(x, torch.zeros_like(x))

            topology_info = {}

            # Get persistence scores from each layer
            for i, block in enumerate(self.blocks):
                persistence_scores = block.topofilter.get_persistence_scores(x_complex)
                filter_mask = block.topofilter.get_filter_mask(x_complex)

                topology_info[f'layer_{i}_persistence'] = persistence_scores
                topology_info[f'layer_{i}_filter_mask'] = filter_mask

                # Update x_complex for next layer
                x_complex = block(x_complex)

        return topology_info

    def get_quantum_info(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get quantum information (amplitudes and phases) for analysis

        Args:
            input_ids (torch.Tensor): Input tokens

        Returns:
            Dict[str, torch.Tensor]: Quantum analysis results
        """
        self.eval()

        with torch.no_grad():
            # Get embeddings
            token_embed = self.token_embedding(input_ids)
            positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
            pos_embed = self.position_embedding(positions)
            x = token_embed + pos_embed
            x_complex = torch.complex(x, torch.zeros_like(x))

            quantum_info = {}

            # Get quantum properties from each layer
            for i, block in enumerate(self.blocks):
                amplitudes = block.qembed.get_amplitudes(x_complex)
                phases = block.qembed.get_phase(x_complex)

                quantum_info[f'layer_{i}_amplitudes'] = amplitudes
                quantum_info[f'layer_{i}_phases'] = phases

                # Update x_complex for next layer
                x_complex = block(x_complex)

        return quantum_info
