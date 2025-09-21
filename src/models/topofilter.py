import torch
import torch.nn as nn
import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist, squareform

class TopoFilter(nn.Module):
    """
    Topological Filtering Layer using approximate persistence homology
    Filters noise from quantum embeddings using topological features
    
    Args:
        persistence_threshold (float): Threshold for filtering noise (default: 0.1)
        tau (float): Temperature parameter for weighting (default: 1.0)
    """
    
    def __init__(self, persistence_threshold: float = 0.1, tau: float = 1.0):
        super().__init__()
        self.persistence_threshold = persistence_threshold
        self.tau = tau
    
    def approximate_persistence(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Approximate persistence computation using wavelet transform
        
        Args:
            distances (torch.Tensor): Pairwise distance matrix
            
        Returns:
            torch.Tensor: Persistence scores for each embedding
        """
        # Convert to numpy for scipy operations (detach from graph)
        distances_np = distances.detach().cpu().numpy()
        
        # Simple approximation: use distance to k-nearest neighbors
        # This approximates the persistence of topological features
        k = min(5, distances_np.shape[0] - 1)  # Adaptive k based on batch size
        
        # Get k-nearest neighbors distances
        sorted_distances = np.sort(distances_np, axis=1)
        knn_distances = sorted_distances[:, 1:k+1]  # Exclude self-distance
        
        # Persistence approximation: average distance to k-NN
        persistence_scores = np.mean(knn_distances, axis=1)
        
        # Convert back to torch tensor
        persistence_tensor = torch.from_numpy(persistence_scores).to(distances.device)
        
        return persistence_tensor
    
    def compute_persistence_weights(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute persistence-based weights for filtering
        
        Args:
            embeddings (torch.Tensor): Complex quantum embeddings
            
        Returns:
            torch.Tensor: Weights based on persistence scores
        """
        # Use real part for distance computation (simpler and effective)
        real_embeddings = embeddings.real
        
        # Compute pairwise distances
        distances = torch.cdist(real_embeddings, real_embeddings)
        
        # Compute approximate persistence scores
        persistence_scores = self.approximate_persistence(distances)
        
        # Compute weights: exp(-persistence/tau)
        # Higher persistence -> lower weight (more noise)
        weights = torch.exp(-persistence_scores / self.tau)
        
        # Apply threshold: zero out weights below threshold
        weights = torch.where(
            persistence_scores > self.persistence_threshold,
            weights,
            torch.zeros_like(weights)
        )
        
        return weights
    
    def forward(self, quantum_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for topological filtering
        
        Args:
            quantum_embeddings (torch.Tensor): Complex embeddings from QEmbed layer
            
        Returns:
            torch.Tensor: Filtered complex embeddings
        """
        batch_size, seq_len, embed_dim = quantum_embeddings.shape
        
        # Reshape for per-token processing
        embeddings_flat = quantum_embeddings.view(-1, embed_dim)
        
        # Compute persistence weights
        persistence_weights = self.compute_persistence_weights(embeddings_flat)
        
        # Apply weights to embeddings
        filtered_flat = embeddings_flat * persistence_weights.unsqueeze(-1)
        
        # Reshape back to original dimensions
        filtered_embeddings = filtered_flat.view(batch_size, seq_len, embed_dim)
        
        return filtered_embeddings
    
    def get_persistence_scores(self, quantum_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get raw persistence scores for analysis
        
        Args:
            quantum_embeddings (torch.Tensor): Complex embeddings
            
        Returns:
            torch.Tensor: Persistence scores for each token
        """
        batch_size, seq_len, embed_dim = quantum_embeddings.shape
        embeddings_flat = quantum_embeddings.view(-1, embed_dim)
        real_embeddings = embeddings_flat.real
        
        distances = torch.cdist(real_embeddings, real_embeddings)
        persistence_scores = self.approximate_persistence(distances)
        
        return persistence_scores.view(batch_size, seq_len)
    
    def get_filter_mask(self, quantum_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get binary mask indicating which tokens were filtered out
        
        Args:
            quantum_embeddings (torch.Tensor): Complex embeddings
            
        Returns:
            torch.Tensor: Binary mask (1 = kept, 0 = filtered)
        """
        persistence_scores = self.get_persistence_scores(quantum_embeddings)
        
        # Create mask: 1 if persistence <= threshold, 0 otherwise
        mask = (persistence_scores <= self.persistence_threshold).float()
        
        return mask
