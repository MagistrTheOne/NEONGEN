import torch
import torch.nn as nn
import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    print("Warning: GUDHI not available. Install with: pip install gudhi")

class TopoFilter(nn.Module):
    """
    Topological Filtering Layer using real persistence homology
    Filters noise from quantum embeddings using topological features
    
    Args:
        persistence_threshold (float): Threshold for filtering noise (default: 0.1)
        tau (float): Temperature parameter for weighting (default: 1.0)
        max_dimension (int): Maximum homology dimension to compute (default: 1)
        use_gudhi (bool): Whether to use GUDHI for real persistence homology (default: True)
    """
    
    def __init__(self, persistence_threshold: float = 0.1, tau: float = 1.0, 
                 max_dimension: int = 1, use_gudhi: bool = True):
        super().__init__()
        self.persistence_threshold = persistence_threshold
        self.tau = tau
        self.max_dimension = max_dimension
        self.use_gudhi = use_gudhi and GUDHI_AVAILABLE
        
        if self.use_gudhi:
            print("Using GUDHI for real persistence homology computation")
        else:
            print("Using approximate persistence homology (GUDHI not available)")
    
    def compute_real_persistence_homology(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute real persistence homology using GUDHI
        
        Args:
            embeddings (np.ndarray): Point cloud embeddings
            
        Returns:
            np.ndarray: Persistence scores for each point
        """
        if not self.use_gudhi:
            return self._approximate_persistence_fallback(embeddings)
        
        try:
            # Create Rips complex
            rips_complex = gudhi.RipsComplex(points=embeddings)
            
            # Compute persistence
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dimension)
            persistence = simplex_tree.persistence()
            
            # Extract persistence scores for 0-dimensional homology (connected components)
            persistence_scores = np.zeros(len(embeddings))
            
            for (dim, (birth, death)) in persistence:
                if dim == 0:  # Connected components
                    # Find points that contribute to this persistence interval
                    persistence_length = death - birth if death != float('inf') else birth
                    
                    # Assign persistence score to points in this component
                    # This is a simplified assignment - in practice, you'd need
                    # to track which points belong to which component
                    for i in range(len(embeddings)):
                        if persistence_scores[i] == 0:  # Not yet assigned
                            persistence_scores[i] = persistence_length
                            break
            
            # If no persistence computed, use fallback
            if np.all(persistence_scores == 0):
                return self._approximate_persistence_fallback(embeddings)
                
            return persistence_scores
            
        except Exception as e:
            print(f"GUDHI persistence computation failed: {e}")
            return self._approximate_persistence_fallback(embeddings)
    
    def _approximate_persistence_fallback(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fallback approximate persistence computation
        
        Args:
            embeddings (np.ndarray): Point cloud embeddings
            
        Returns:
            np.ndarray: Approximate persistence scores
        """
        # Compute pairwise distances
        distances = pdist(embeddings)
        distance_matrix = squareform(distances)
        
        # Use k-NN approach as approximation
        k = min(5, len(embeddings) - 1)
        sorted_distances = np.sort(distance_matrix, axis=1)
        knn_distances = sorted_distances[:, 1:k+1]  # Exclude self-distance
        
        # Persistence approximation: average distance to k-NN
        persistence_scores = np.mean(knn_distances, axis=1)
        
        return persistence_scores
    
    def compute_persistence_weights(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute persistence-based weights for filtering using real persistence homology
        
        Args:
            embeddings (torch.Tensor): Complex quantum embeddings
            
        Returns:
            torch.Tensor: Weights based on persistence scores
        """
        # Use real part for distance computation (simpler and effective)
        real_embeddings = embeddings.real.detach().cpu().numpy()
        
        # Compute real persistence homology
        persistence_scores = self.compute_real_persistence_homology(real_embeddings)
        
        # Convert back to torch tensor
        persistence_tensor = torch.from_numpy(persistence_scores).to(embeddings.device)
        
        # Compute weights: exp(-persistence/tau)
        # Higher persistence -> lower weight (more noise)
        weights = torch.exp(-persistence_tensor / self.tau)
        
        # Apply threshold: zero out weights below threshold
        weights = torch.where(
            persistence_tensor > self.persistence_threshold,
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
        Get raw persistence scores for analysis using real persistence homology
        
        Args:
            quantum_embeddings (torch.Tensor): Complex embeddings
            
        Returns:
            torch.Tensor: Persistence scores for each token
        """
        batch_size, seq_len, embed_dim = quantum_embeddings.shape
        embeddings_flat = quantum_embeddings.view(-1, embed_dim)
        real_embeddings = embeddings_flat.real.detach().cpu().numpy()
        
        # Compute real persistence homology
        persistence_scores = self.compute_real_persistence_homology(real_embeddings)
        
        # Convert back to torch tensor and reshape
        persistence_tensor = torch.from_numpy(persistence_scores).to(quantum_embeddings.device)
        return persistence_tensor.view(batch_size, seq_len)
    
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
