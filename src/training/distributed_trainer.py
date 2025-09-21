"""
Distributed training implementation for QTG models
Supports multi-GPU and multi-node training with advanced optimizations
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import time
import logging
from typing import Dict, Any, Optional, Tuple
from contextlib import contextmanager

from .trainer import QTGTrainer
from .loss_functions import QTGLoss
from ..models.qtgm_model import QTGModel
from ..utils.memory_utils import MemoryOptimizer, MixedPrecisionTrainer


class DistributedQTGTrainer:
    """
    Distributed trainer for QTG models with advanced optimizations
    """
    
    def __init__(
        self,
        model: QTGModel,
        optimizer: torch.optim.Optimizer,
        loss_fn: QTGLoss,
        device: str = 'cuda',
        world_size: int = 1,
        rank: int = 0,
        use_mixed_precision: bool = True,
        use_gradient_checkpointing: bool = True,
        use_zero_optimization: bool = False
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.world_size = world_size
        self.rank = rank
        self.use_mixed_precision = use_mixed_precision
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_zero_optimization = use_zero_optimization
        
        # Initialize components
        self.memory_optimizer = MemoryOptimizer(device=device)
        self.mp_trainer = MixedPrecisionTrainer(use_fp16=use_mixed_precision)
        
        # Setup model for distributed training
        self._setup_distributed_model()
        
        # Setup logging
        self._setup_logging()
        
        # Training state
        self.step_count = 0
        self.epoch_count = 0
        self.best_loss = float('inf')
        
    def _setup_distributed_model(self):
        """Setup model for distributed training"""
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Enable gradient checkpointing if requested
        if self.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Wrap model with DDP
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.rank] if self.device == 'cuda' else None,
                output_device=self.rank if self.device == 'cuda' else None,
                find_unused_parameters=False,  # Optimize for QTG model
                broadcast_buffers=False  # Reduce communication overhead
            )
        
        # Apply memory optimizations
        self.model = self.memory_optimizer.optimize_model_for_memory(self.model)
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        else:
            # Manual gradient checkpointing for QTG blocks
            for module in self.model.modules():
                if hasattr(module, 'gradient_checkpointing_enable'):
                    module.gradient_checkpointing_enable()
    
    def _setup_logging(self):
        """Setup logging for distributed training"""
        if self.rank == 0:  # Only log from rank 0
            logging.basicConfig(
                level=logging.INFO,
                format=f'[Rank {self.rank}] %(asctime)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
    
    def log(self, message: str, level: int = logging.INFO):
        """Log message only from rank 0"""
        if self.logger:
            self.logger.log(level, message)
    
    def training_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        gradient_accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """
        Single distributed training step
        
        Args:
            input_ids: Input token ids
            labels: Target token ids
            gradient_accumulation_steps: Number of steps to accumulate gradients
            
        Returns:
            Dict with loss components
        """
        self.model.train()
        
        # Scale loss by accumulation steps
        loss_scale = 1.0 / gradient_accumulation_steps
        
        total_loss = 0.0
        loss_components = {}
        
        for step in range(gradient_accumulation_steps):
            # Get batch slice
            batch_start = step * (input_ids.size(0) // gradient_accumulation_steps)
            batch_end = (step + 1) * (input_ids.size(0) // gradient_accumulation_steps)
            
            batch_input_ids = input_ids[batch_start:batch_end]
            batch_labels = labels[batch_start:batch_end]
            
            # Forward pass with mixed precision
            with self.mp_trainer.autocast():
                outputs = self.model(batch_input_ids)
                loss, components = self.loss_fn(
                    self.model.module if hasattr(self.model, 'module') else self.model,
                    batch_input_ids,
                    batch_labels,
                    outputs
                )
            
            # Scale loss
            loss = loss * loss_scale
            
            # Backward pass
            scaled_loss = self.mp_trainer.scale_loss(loss)
            scaled_loss.backward()
            
            # Accumulate loss components
            total_loss += loss.item()
            for key, value in components.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value.item() * loss_scale
        
        # Gradient clipping
        if self.world_size > 1:
            # Unscale gradients for clipping
            self.mp_trainer.unscale_gradients(self.optimizer)
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.mp_trainer.step(self.optimizer)
        self.optimizer.zero_grad()
        
        # Update step count
        self.step_count += 1
        
        # Synchronize across ranks
        if self.world_size > 1:
            dist.barrier()
        
        # Average loss components across ranks
        if self.world_size > 1:
            loss_components = self._average_loss_components(loss_components)
        
        return loss_components
    
    def _average_loss_components(self, loss_components: Dict[str, float]) -> Dict[str, float]:
        """Average loss components across all ranks"""
        averaged_components = {}
        
        for key, value in loss_components.items():
            # Convert to tensor for distributed averaging
            value_tensor = torch.tensor(value, device=self.device)
            
            # All-reduce across ranks
            dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
            averaged_value = value_tensor.item() / self.world_size
            
            averaged_components[key] = averaged_value
        
        return averaged_components
    
    def validation_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Single distributed validation step
        
        Args:
            input_ids: Input token ids
            labels: Target token ids
            
        Returns:
            Dict with loss components
        """
        self.model.eval()
        
        with torch.no_grad():
            with self.mp_trainer.autocast():
                outputs = self.model(input_ids)
                loss, components = self.loss_fn(
                    self.model.module if hasattr(self.model, 'module') else self.model,
                    input_ids,
                    labels,
                    outputs
                )
        
        # Convert to float values
        loss_components = {k: v.item() for k, v in components.items()}
        
        # Synchronize across ranks
        if self.world_size > 1:
            dist.barrier()
            loss_components = self._average_loss_components(loss_components)
        
        return loss_components
    
    def save_checkpoint(
        self,
        checkpoint_path: str,
        epoch: int,
        step: int,
        is_best: bool = False
    ):
        """
        Save checkpoint (only from rank 0)
        
        Args:
            checkpoint_path: Path to save checkpoint
            epoch: Current epoch
            step: Current step
            is_best: Whether this is the best checkpoint
        """
        if self.rank == 0:
            # Get model state dict
            model_state_dict = (
                self.model.module.state_dict() 
                if hasattr(self.model, 'module') 
                else self.model.state_dict()
            )
            
            checkpoint = {
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                'step': step,
                'best_loss': self.best_loss,
                'world_size': self.world_size,
                'rank': self.rank
            }
            
            # Save checkpoint
            torch.save(checkpoint, checkpoint_path)
            
            # Save best checkpoint separately
            if is_best:
                best_path = checkpoint_path.replace('.pt', '_best.pt')
                torch.save(checkpoint, best_path)
                self.log(f"Saved best checkpoint to {best_path}")
            
            self.log(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Checkpoint data
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        model_state_dict = checkpoint['model_state_dict']
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.epoch_count = checkpoint.get('epoch', 0)
        self.step_count = checkpoint.get('step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        self.log(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint
    
    def generate_sample(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 0.8
    ) -> torch.Tensor:
        """
        Generate sample text (only from rank 0)
        
        Args:
            input_ids: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated sequence
        """
        if self.rank == 0:
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
        else:
            return input_ids  # Return input for non-rank-0 processes
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics"""
        stats = {
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'best_loss': self.best_loss,
            'world_size': self.world_size,
            'rank': self.rank
        }
        
        # Add memory stats if available
        if torch.cuda.is_available():
            stats['gpu_memory_used'] = torch.cuda.memory_allocated() / (1024**3)  # GB
            stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / (1024**3)  # GB
        
        return stats


def setup_distributed_training(
    rank: int,
    world_size: int,
    backend: str = 'nccl'
):
    """
    Setup distributed training environment
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Distributed backend
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    # Set device
    torch.cuda.set_device(rank)


def cleanup_distributed_training():
    """Cleanup distributed training environment"""
    if dist.is_initialized():
        dist.destroy_process_group()


@contextmanager
def distributed_training_context(rank: int, world_size: int):
    """
    Context manager for distributed training setup/cleanup
    
    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    try:
        setup_distributed_training(rank, world_size)
        yield
    finally:
        cleanup_distributed_training()


def create_distributed_sampler(dataset, world_size: int, rank: int, shuffle: bool = True):
    """
    Create distributed sampler for dataset
    
    Args:
        dataset: PyTorch dataset
        world_size: Total number of processes
        rank: Process rank
        shuffle: Whether to shuffle data
        
    Returns:
        DistributedSampler
    """
    return DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle
    )


def estimate_distributed_training_time(
    model_size: int,
    world_size: int,
    batch_size: int,
    sequence_length: int,
    steps: int
) -> float:
    """
    Estimate distributed training time
    
    Args:
        model_size: Model size in parameters
        world_size: Number of GPUs
        batch_size: Batch size per GPU
        sequence_length: Sequence length
        steps: Number of training steps
        
    Returns:
        Estimated time in hours
    """
    # Rough estimation based on model size and hardware
    # This is a simplified model - actual times will vary
    
    # Base time per step (seconds) - rough approximation
    base_time_per_step = (model_size / 1e9) * 0.1  # 0.1s per billion parameters
    
    # Scale by sequence length
    sequence_scale = (sequence_length / 512) ** 1.5
    
    # Scale by batch size
    batch_scale = (batch_size / 8) ** 0.5
    
    # Distributed efficiency (diminishing returns)
    distributed_efficiency = min(0.9, 0.5 + 0.1 * world_size)
    
    # Calculate time per step
    time_per_step = base_time_per_step * sequence_scale * batch_scale / distributed_efficiency
    
    # Total time
    total_time_seconds = time_per_step * steps
    total_time_hours = total_time_seconds / 3600
    
    return total_time_hours


if __name__ == "__main__":
    # Example usage
    print("Distributed QTG Training Module")
    print("Use this module with torch.distributed.launch or torchrun")
    print("Example: torchrun --nproc_per_node=4 train_distributed.py")
