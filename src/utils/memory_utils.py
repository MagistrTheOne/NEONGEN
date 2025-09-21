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
    Estimate memory requirements for model with improved accuracy

    Args:
        model: PyTorch model
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        Dict with memory estimates
    """
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    
    # Memory per parameter (bytes)
    param_memory_fp32 = param_count * 4  # float32
    param_memory_fp16 = param_count * 2  # float16
    
    # Estimate activation memory more accurately
    # For QTG model with complex numbers
    embed_dim = getattr(model, 'embed_dim', 512)
    num_layers = getattr(model, 'num_layers', 12)
    
    # Activation memory per layer (complex numbers = 2x real)
    activation_per_layer = batch_size * seq_len * embed_dim * 2 * 4  # complex64
    total_activation_memory = activation_per_layer * num_layers * 2  # forward + backward
    
    # Attention memory (quadratic in sequence length)
    attention_memory = batch_size * num_layers * seq_len * seq_len * 4  # float32
    
    # Total activation memory
    total_activation_memory += attention_memory
    
    # Optimizer memory (AdamW: 2x parameters for momentum and variance)
    optimizer_memory_fp32 = param_memory_fp32 * 2
    optimizer_memory_fp16 = param_memory_fp16 * 2
    
    # Gradients memory
    gradient_memory = param_memory_fp32
    
    # Total memory estimates
    total_memory_fp32 = param_memory_fp32 + total_activation_memory + optimizer_memory_fp32 + gradient_memory
    total_memory_fp16 = param_memory_fp16 + total_activation_memory + optimizer_memory_fp16 + gradient_memory
    
    return {
        'parameters_fp32': param_memory_fp32 / (1024**3),  # GB
        'parameters_fp16': param_memory_fp16 / (1024**3),  # GB
        'activations': total_activation_memory / (1024**3),  # GB
        'gradients': gradient_memory / (1024**3),  # GB
        'optimizer_fp32': optimizer_memory_fp32 / (1024**3),  # GB
        'optimizer_fp16': optimizer_memory_fp16 / (1024**3),  # GB
        'total_estimate_fp32': total_memory_fp32 / (1024**3),  # GB
        'total_estimate_fp16': total_memory_fp16 / (1024**3),  # GB
        'parameter_count': param_count,
        'embed_dim': embed_dim,
        'num_layers': num_layers
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


class AdvancedMemoryOptimizer:
    """
    Advanced memory optimization for QTG models
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.memory_stats = {}
    
    def optimize_model_architecture(self, model: torch.nn.Module, target_memory_gb: float) -> torch.nn.Module:
        """
        Optimize model architecture for target memory usage
        
        Args:
            model: QTG model to optimize
            target_memory_gb: Target memory usage in GB
            
        Returns:
            Optimized model
        """
        # Estimate current memory usage
        current_memory = estimate_model_memory(model, batch_size=1, seq_len=512)
        
        if current_memory['total_estimate_fp16'] <= target_memory_gb:
            return model
        
        # Calculate scaling factor
        scale_factor = target_memory_gb / current_memory['total_estimate_fp16']
        
        # Optimize model parameters
        if hasattr(model, 'embed_dim'):
            model.embed_dim = int(model.embed_dim * (scale_factor ** 0.5))
        
        if hasattr(model, 'num_layers'):
            model.num_layers = max(1, int(model.num_layers * (scale_factor ** 0.3)))
        
        if hasattr(model, 'num_heads'):
            model.num_heads = max(1, int(model.num_heads * (scale_factor ** 0.2)))
        
        return model
    
    def enable_activation_checkpointing(self, model: torch.nn.Module):
        """Enable activation checkpointing for memory efficiency"""
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
    
    def enable_cpu_offloading(self, model: torch.nn.Module):
        """Enable CPU offloading for large models"""
        # Move model to CPU and only move to GPU during forward pass
        model = model.cpu()
        
        # Wrap forward method to handle device movement
        original_forward = model.forward
        
        def cpu_offload_forward(*args, **kwargs):
            # Move to GPU for forward pass
            model.to(self.device)
            result = original_forward(*args, **kwargs)
            # Move back to CPU
            model.cpu()
            return result
        
        model.forward = cpu_offload_forward
        return model
    
    def optimize_attention_memory(self, model: torch.nn.Module, max_seq_len: int = 2048):
        """Optimize attention memory usage"""
        # Implement sparse attention or sliding window attention
        for module in model.modules():
            if hasattr(module, 'max_seq_len'):
                module.max_seq_len = min(module.max_seq_len, max_seq_len)
        
        return model
    
    def profile_memory_usage(self, model: torch.nn.Module, batch_size: int, seq_len: int):
        """Profile detailed memory usage"""
        if not torch.cuda.is_available():
            return {}
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create dummy input
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids)
        
        # Get memory stats
        memory_stats = {
            'peak_memory_allocated': torch.cuda.max_memory_allocated() / (1024**3),
            'peak_memory_reserved': torch.cuda.max_memory_reserved() / (1024**3),
            'current_memory_allocated': torch.cuda.memory_allocated() / (1024**3),
            'current_memory_reserved': torch.cuda.memory_reserved() / (1024**3)
        }
        
        torch.cuda.empty_cache()
        return memory_stats


def create_memory_efficient_model(
    base_model: torch.nn.Module,
    target_memory_gb: float,
    batch_size: int = 1,
    seq_len: int = 512
) -> torch.nn.Module:
    """
    Create memory-efficient version of model
    
    Args:
        base_model: Base QTG model
        target_memory_gb: Target memory usage in GB
        batch_size: Batch size for memory estimation
        seq_len: Sequence length for memory estimation
        
    Returns:
        Memory-optimized model
    """
    optimizer = AdvancedMemoryOptimizer()
    
    # Profile current memory usage
    current_memory = estimate_model_memory(base_model, batch_size, seq_len)
    
    if current_memory['total_estimate_fp16'] <= target_memory_gb:
        return base_model
    
    # Optimize model architecture
    optimized_model = optimizer.optimize_model_architecture(base_model, target_memory_gb)
    
    # Enable memory optimizations
    optimizer.enable_activation_checkpointing(optimized_model)
    optimizer.optimize_attention_memory(optimized_model, max_seq_len=seq_len)
    
    # Profile optimized memory usage
    optimized_memory = estimate_model_memory(optimized_model, batch_size, seq_len)
    
    print(f"Memory optimization results:")
    print(f"  Original: {current_memory['total_estimate_fp16']:.2f} GB")
    print(f"  Optimized: {optimized_memory['total_estimate_fp16']:.2f} GB")
    print(f"  Reduction: {(1 - optimized_memory['total_estimate_fp16'] / current_memory['total_estimate_fp16']) * 100:.1f}%")
    
    return optimized_model


def get_optimal_batch_size(
    model: torch.nn.Module,
    available_memory_gb: float,
    seq_len: int = 512,
    precision: str = 'fp16'
) -> int:
    """
    Find optimal batch size for given memory constraints
    
    Args:
        model: QTG model
        available_memory_gb: Available GPU memory in GB
        seq_len: Sequence length
        precision: Model precision ('fp16' or 'fp32')
        
    Returns:
        Optimal batch size
    """
    max_batch_size = 1
    
    # Binary search for optimal batch size
    low, high = 1, 128
    
    while low <= high:
        mid = (low + high) // 2
        memory_est = estimate_model_memory(model, mid, seq_len)
        
        memory_key = f'total_estimate_{precision}'
        if memory_est[memory_key] <= available_memory_gb * 0.9:  # 90% safety margin
            max_batch_size = mid
            low = mid + 1
        else:
            high = mid - 1
    
    return max_batch_size


def optimize_for_h200_gpu(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Optimize model specifically for H200 GPU (200GB VRAM)
    
    Args:
        model: QTG model to optimize
        
    Returns:
        Optimization recommendations
    """
    h200_memory_gb = 200.0
    
    # Estimate memory for different configurations
    configs = [
        {'batch_size': 1, 'seq_len': 2048, 'precision': 'fp16'},
        {'batch_size': 2, 'seq_len': 2048, 'precision': 'fp16'},
        {'batch_size': 4, 'seq_len': 1024, 'precision': 'fp16'},
        {'batch_size': 8, 'seq_len': 512, 'precision': 'fp16'},
    ]
    
    recommendations = []
    
    for config in configs:
        memory_est = estimate_model_memory(
            model, 
            config['batch_size'], 
            config['seq_len']
        )
        
        memory_usage = memory_est[f"total_estimate_{config['precision']}"]
        
        if memory_usage <= h200_memory_gb * 0.9:  # 90% safety margin
            recommendations.append({
                'batch_size': config['batch_size'],
                'seq_len': config['seq_len'],
                'precision': config['precision'],
                'memory_usage_gb': memory_usage,
                'memory_efficiency': memory_usage / h200_memory_gb * 100
            })
    
    # Sort by memory efficiency (higher is better)
    recommendations.sort(key=lambda x: x['memory_efficiency'], reverse=True)
    
    return {
        'gpu': 'H200',
        'total_memory_gb': h200_memory_gb,
        'recommendations': recommendations,
        'best_config': recommendations[0] if recommendations else None
    }
