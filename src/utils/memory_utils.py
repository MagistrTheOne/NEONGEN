import torch
import gc
from typing import Dict, Any, Optional
from contextlib import contextmanager
import psutil
import GPUtil

class MemoryOptimizer:
    """
    Memory optimization utilities for RTX 2080 Super (8GB VRAM)
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.initial_memory = self.get_gpu_memory()

    def get_gpu_memory(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return {'used': 0.0, 'total': 0.0, 'free': 0.0}

        gpu = GPUtil.getGPUs()[0]
        return {
            'used': gpu.memoryUsed,
            'total': gpu.memoryTotal,
            'free': gpu.memoryFree
        }

    def get_cpu_memory(self) -> Dict[str, float]:
        """Get current CPU memory usage"""
        memory = psutil.virtual_memory()
        return {
            'used': memory.used / (1024**3),  # GB
            'total': memory.total / (1024**3),  # GB
            'free': memory.available / (1024**3)  # GB
        }

    @contextmanager
    def memory_monitor(self, step_name: str = ""):
        """Context manager for monitoring memory usage"""
        before = self.get_gpu_memory()
        torch.cuda.empty_cache()

        try:
            yield
        finally:
            after = self.get_gpu_memory()
            torch.cuda.empty_cache()

            print(f"[{step_name}] GPU Memory: {before['used']:.1f}GB -> {after['used']:.1f}GB "
                  f"(Δ{after['used'] - before['used']:+.1f}GB)")

    def optimize_model_for_memory(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply memory optimizations to model

        Args:
            model: PyTorch model to optimize

        Returns:
            Optimized model
        """
        # Enable gradient checkpointing for transformer blocks
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        # Convert to half precision if needed
        # Note: This will be handled by the trainer

        return model

    def create_memory_efficient_dataloader(
        self,
        dataset,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2
    ):
        """Create memory-efficient DataLoader"""
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=True if num_workers > 0 else False
        )

        return dataloader

    def cleanup_memory(self):
        """Aggressive memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def log_memory_stats(self, step: int = 0):
        """Log comprehensive memory statistics"""
        gpu_mem = self.get_gpu_memory()
        cpu_mem = self.get_cpu_memory()

        print(f"Step {step} - Memory Stats:")
        print(f"  GPU: {gpu_mem['used']:.1f}/{gpu_mem['total']:.1f}GB "
              f"({gpu_mem['free']:.1f}GB free)")
        print(f"  CPU: {cpu_mem['used']:.1f}/{cpu_mem['total']:.1f}GB "
              f"({cpu_mem['free']:.1f}GB free)")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"  PyTorch CUDA: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


class GradientAccumulator:
    """
    Gradient accumulation for effective larger batch sizes
    """

    def __init__(self, accumulation_steps: int):
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
        self.scaled_loss = 0.0

    def accumulate(self, loss: torch.Tensor) -> bool:
        """
        Accumulate gradients

        Args:
            loss: Loss tensor to accumulate

        Returns:
            bool: True if should update parameters
        """
        # Scale loss by accumulation steps
        loss = loss / self.accumulation_steps

        # Backward pass
        loss.backward()

        self.step_count += 1

        # Check if should update
        should_update = (self.step_count % self.accumulation_steps == 0)

        if should_update:
            self.step_count = 0

        return should_update

    def reset(self):
        """Reset accumulator state"""
        self.step_count = 0
        self.scaled_loss = 0.0


class MixedPrecisionTrainer:
    """
    Mixed precision training with automatic scaling
    """

    def __init__(self, use_fp16: bool = True):
        self.use_fp16 = use_fp16 and torch.cuda.is_available()

        if self.use_fp16:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None

    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision"""
        if self.use_fp16:
            from torch.cuda.amp import autocast
            with autocast():
                yield
        else:
            yield

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training"""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer):
        """Perform optimizer step with scaling"""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients for clipping"""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)


def estimate_model_memory(model: torch.nn.Module, batch_size: int, seq_len: int) -> Dict[str, float]:
    """
    Estimate memory requirements for model

    Args:
        model: PyTorch model
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        Dict with memory estimates
    """
    # Rough estimation based on parameter count and activations
    param_count = sum(p.numel() for p in model.parameters())
    param_memory = param_count * 4  # float32

    # Estimate activation memory (rough approximation)
    # Transformer: roughly 2x parameters for activations per layer
    activation_memory = param_memory * 2 * len(list(model.modules()))

    # Batch size and sequence scaling
    total_memory = (param_memory + activation_memory) * batch_size * seq_len / (512 * 32)

    # Optimizer memory (AdamW: 2x parameters)
    optimizer_memory = param_memory * 2

    # Gradients memory
    gradient_memory = param_memory

    return {
        'parameters': param_memory / (1024**3),  # GB
        'activations': activation_memory / (1024**3),  # GB
        'gradients': gradient_memory / (1024**3),  # GB
        'optimizer': optimizer_memory / (1024**3),  # GB
        'total_estimate': total_memory / (1024**3)  # GB
    }


def create_optimized_config(gpu_memory_gb: float = 8.0) -> Dict[str, Any]:
    """
    Create optimized configuration based on GPU memory

    Args:
        gpu_memory_gb: Available GPU memory in GB

    Returns:
        Dict with optimized parameters
    """
    # Base configuration for 8GB RTX 2080 Super
    base_config = {
        'embed_dim': 512,
        'num_layers': 12,
        'num_heads': 8,
        'batch_size': 8,
        'seq_len': 512,
        'gradient_accumulation': 4,
        'fp16': True
    }

    # Scale down for less memory
    if gpu_memory_gb < 8.0:
        scale_factor = gpu_memory_gb / 8.0
        base_config['embed_dim'] = int(base_config['embed_dim'] * scale_factor)
        base_config['num_layers'] = max(6, int(base_config['num_layers'] * scale_factor))
        base_config['batch_size'] = max(1, int(base_config['batch_size'] * scale_factor))

    # Scale up for more memory
    elif gpu_memory_gb > 8.0:
        scale_factor = min(2.0, gpu_memory_gb / 8.0)
        base_config['embed_dim'] = int(base_config['embed_dim'] * scale_factor)
        base_config['num_layers'] = int(base_config['num_layers'] * scale_factor)
        base_config['batch_size'] = int(base_config['batch_size'] * scale_factor)

    return base_config
