import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
from models.qembed import QEmbed
from models.topofilter import TopoFilter
from models.qtgm_model import QTGModel

class QTGLoss(nn.Module):
    """
    Quantum-Topological-Geometric Loss function
    Combines standard language modeling loss with QTG regularization terms
    """

    def __init__(
        self,
        vocab_size: int,
        lambda_topo: float = 0.1,
        lambda_geo: float = 0.05,
        lambda_quantum: float = 0.01,
        ignore_index: int = -100
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.lambda_topo = lambda_topo
        self.lambda_geo = lambda_geo
        self.lambda_quantum = lambda_quantum
        self.ignore_index = ignore_index

        # Cross-entropy loss for language modeling
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        model: QTGModel,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        outputs: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute QTG loss

        Args:
            model (QTGModel): The QTG model
            input_ids (torch.Tensor): Input token ids
            labels (torch.Tensor): Target token ids
            outputs (Dict): Model outputs

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Total loss and loss components
        """
        logits = outputs['logits']
        last_hidden_state = outputs['last_hidden_state']

        # 1. Standard language modeling loss
        l_ce = self.ce_loss(
            logits.view(-1, self.vocab_size),
            labels.view(-1)
        )

        # 2. Topological regularization
        l_topo = self._topological_regularization(model, input_ids, last_hidden_state)

        # 3. Geometric regularization
        l_geo = self._geometric_regularization(model, outputs)

        # 4. Quantum coherence regularization
        l_quantum = self._quantum_regularization(last_hidden_state)

        # Total loss
        total_loss = l_ce + self.lambda_topo * l_topo + self.lambda_geo * l_geo + self.lambda_quantum * l_quantum

        loss_components = {
            'ce_loss': l_ce,
            'topo_loss': l_topo,
            'geo_loss': l_geo,
            'quantum_loss': l_quantum,
            'total_loss': total_loss
        }

        return total_loss, loss_components

    def _topological_regularization(
        self,
        model: QTGModel,
        input_ids: torch.Tensor,
        last_hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Topological regularization encourages topological diversity
        Penalizes low persistence scores (noise) and rewards high persistence

        Args:
            model (QTGModel): The model
            input_ids (torch.Tensor): Input tokens
            last_hidden_state (torch.Tensor): Final hidden states

        Returns:
            torch.Tensor: Topological regularization loss
        """
        topo_loss = 0.0
        batch_size, seq_len = input_ids.shape

        # Get topological information from all layers
        topo_info = model.get_topology_info(input_ids)

        # Compute regularization for each layer
        for layer_name, persistence_scores in topo_info.items():
            if 'persistence' in layer_name:
                # Penalize low persistence (noise) and encourage diversity
                # Loss = mean((1 - persistence_score)^2)
                layer_loss = torch.mean((1.0 - persistence_scores) ** 2)
                topo_loss += layer_loss

        # Average across layers
        topo_loss = topo_loss / len(model.blocks)

        return topo_loss

    def _geometric_regularization(
        self,
        model: QTGModel,
        outputs: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Geometric regularization encourages Fisher-Rao manifold consistency
        Penalizes large distances between consecutive layers

        Args:
            model (QTGModel): The model
            outputs (Dict): Model outputs

        Returns:
            torch.Tensor: Geometric regularization loss
        """
        geo_loss = 0.0

        # Get attention maps from all layers
        attention_maps = outputs.get('attention_maps', [])

        if len(attention_maps) > 1:
            # Compute consistency between consecutive attention maps
            for i in range(len(attention_maps) - 1):
                attn_current = attention_maps[i]
                attn_next = attention_maps[i + 1]

                # Compute KL divergence between attention distributions
                # This encourages geometric consistency
                attn_current_flat = attn_current.view(-1, attn_current.size(-1))
                attn_next_flat = attn_next.view(-1, attn_next.size(-1))

                # KL divergence regularization
                kl_loss = F.kl_div(
                    F.log_softmax(attn_current_flat, dim=-1),
                    F.softmax(attn_next_flat, dim=-1),
                    reduction='batchmean'
                )

                geo_loss += kl_loss

            # Average across layer pairs
            geo_loss = geo_loss / (len(attention_maps) - 1)

        return geo_loss

    def _quantum_regularization(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Quantum coherence regularization maintains quantum state validity
        Penalizes states that violate quantum mechanical constraints

        Args:
            last_hidden_state (torch.Tensor): Final complex hidden states

        Returns:
            torch.Tensor: Quantum regularization loss
        """
        # 1. Normalization check: states should be approximately normalized
        norms = torch.norm(last_hidden_state, p=2, dim=-1)
        norm_penalty = torch.mean((norms - 1.0) ** 2)

        # 2. Coherence penalty: encourage phase relationships
        # Penalize random phase distributions
        phases = torch.angle(last_hidden_state)

        # Compute phase entropy (lower entropy = more coherent)
        phase_probs = F.softmax(phases.view(-1, phases.size(-1)), dim=-1)
        phase_entropy = -torch.sum(phase_probs * torch.log(phase_probs + 1e-8), dim=-1)
        coherence_loss = torch.mean(phase_entropy)

        # 3. Complex conjugate symmetry encouragement
        # For real-valued signals, imaginary part should be small
        imag_penalty = torch.mean(last_hidden_state.imag ** 2)

        quantum_loss = norm_penalty + coherence_loss + imag_penalty

        return quantum_loss


class QTGTrainer:
    """
    Specialized trainer for QTG models with advanced loss tracking
    """

    def __init__(
        self,
        model: QTGModel,
        optimizer: torch.optim.Optimizer,
        loss_fn: QTGLoss,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

        self.model.to(device)

    def training_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Single training step

        Args:
            input_ids (torch.Tensor): Input tokens
            labels (torch.Tensor): Target tokens

        Returns:
            Dict[str, float]: Loss components
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(input_ids)

        # Compute loss
        loss, loss_components = self.loss_fn(self.model, input_ids, labels, outputs)

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step
        self.optimizer.step()

        return {k: v.item() for k, v in loss_components.items()}

    def validation_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Single validation step

        Args:
            input_ids (torch.Tensor): Input tokens
            labels (torch.Tensor): Target tokens

        Returns:
            Dict[str, float]: Loss components
        """
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(input_ids)
            loss, loss_components = self.loss_fn(self.model, input_ids, labels, outputs)

        return {k: v.item() for k, v in loss_components.items()}

    def generate_sample(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 0.8
    ) -> torch.Tensor:
        """
        Generate sample text for monitoring

        Args:
            input_ids (torch.Tensor): Input prompt
            max_length (int): Maximum generation length
            temperature (float): Sampling temperature

        Returns:
            torch.Tensor: Generated sequence
        """
        self.model.eval()

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9
            )

        return generated


def create_qtg_loss(
    vocab_size: int,
    lambda_topo: float = 0.1,
    lambda_geo: float = 0.05,
    lambda_quantum: float = 0.01
) -> QTGLoss:
    """
    Factory function for creating QTG loss

    Args:
        vocab_size (int): Vocabulary size
        lambda_topo (float): Topological regularization weight
        lambda_geo (float): Geometric regularization weight
        lambda_quantum (float): Quantum regularization weight

    Returns:
        QTGLoss: Configured loss function
    """
    return QTGLoss(
        vocab_size=vocab_size,
        lambda_topo=lambda_topo,
        lambda_geo=lambda_geo,
        lambda_quantum=lambda_quantum
    )
