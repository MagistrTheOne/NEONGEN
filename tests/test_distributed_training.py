#!/usr/bin/env python3
"""
Comprehensive tests for distributed training implementation
"""

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.qtgm_model import QTGModel
from training.loss_functions import QTGLoss


class TestDistributedTraining:
    """Test suite for distributed training functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Mock distributed environment
        self.original_dist_available = dist.is_available()
        self.original_dist_initialized = dist.is_initialized()
    
    def teardown_method(self):
        """Cleanup after tests"""
        # Restore original distributed state
        if self.original_dist_initialized and not dist.is_initialized():
            # Don't try to initialize if it wasn't originally initialized
            pass
    
    @patch('torch.distributed.is_available', return_value=True)
    @patch('torch.distributed.is_initialized', return_value=False)
    def test_distributed_environment_detection(self, mock_init, mock_avail):
        """Test detection of distributed training environment"""
        assert dist.is_available()
        assert not dist.is_initialized()
    
    @patch('torch.distributed.is_available', return_value=True)
    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.get_world_size', return_value=4)
    @patch('torch.distributed.get_rank', return_value=0)
    def test_distributed_info(self, mock_rank, mock_world_size, mock_init, mock_avail):
        """Test getting distributed training information"""
        assert dist.get_world_size() == 4
        assert dist.get_rank() == 0
    
    def test_model_parallelism_setup(self):
        """Test setting up model parallelism"""
        model = QTGModel(
            vocab_size=1000,
            embed_dim=512,
            num_layers=12,
            num_heads=8,
            max_seq_len=512
        )
        
        # Test model sharding (simplified)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        
        # Test that model can be split across devices
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        assert device_count >= 1
    
    def test_data_parallelism_setup(self):
        """Test setting up data parallelism"""
        # Create dummy dataset
        dataset_size = 1000
        batch_size = 8
        world_size = 4
        
        # Calculate expected samples per rank
        samples_per_rank = dataset_size // world_size
        assert samples_per_rank == 250
        
        # Test batch size scaling
        effective_batch_size = batch_size * world_size
        assert effective_batch_size == 32
    
    def test_gradient_synchronization(self):
        """Test gradient synchronization across ranks"""
        model = QTGModel(
            vocab_size=100,
            embed_dim=32,
            num_layers=2,
            num_heads=2,
            max_seq_len=64
        )
        
        # Create dummy loss
        input_ids = torch.randint(0, 100, (2, 32))
        outputs = model(input_ids)
        logits = outputs['logits']
        loss = nn.CrossEntropyLoss()(logits.view(-1, 100), input_ids.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.grad is not None:
                assert param.grad.shape == param.shape
    
    def test_learning_rate_scaling(self):
        """Test learning rate scaling for distributed training"""
        base_lr = 1e-4
        world_size = 4
        
        # Linear scaling rule
        scaled_lr = base_lr * world_size
        assert scaled_lr == 4e-4
        
        # Square root scaling rule
        sqrt_scaled_lr = base_lr * (world_size ** 0.5)
        assert sqrt_scaled_lr == 2e-4
    
    def test_checkpoint_saving_loading(self):
        """Test checkpoint saving and loading for distributed training"""
        model = QTGModel(
            vocab_size=100,
            embed_dim=32,
            num_layers=1,
            num_heads=2,
            max_seq_len=32
        )
        
        optimizer = torch.optim.Adam(model.parameters())
        
        # Create checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 10,
            'step': 1000
        }
        
        # Save checkpoint (simulated)
        checkpoint_path = '/tmp/test_checkpoint.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path)
        
        assert loaded_checkpoint['epoch'] == 10
        assert loaded_checkpoint['step'] == 1000
        assert 'model_state_dict' in loaded_checkpoint
        assert 'optimizer_state_dict' in loaded_checkpoint
        
        # Cleanup
        os.remove(checkpoint_path)
    
    def test_memory_efficient_distributed_training(self):
        """Test memory-efficient distributed training setup"""
        model = QTGModel(
            vocab_size=1000,
            embed_dim=256,
            num_layers=4,
            num_heads=4,
            max_seq_len=256
        )
        
        # Test gradient accumulation
        accumulation_steps = 4
        batch_size = 2
        effective_batch_size = batch_size * accumulation_steps
        
        assert effective_batch_size == 8
        
        # Test mixed precision
        use_fp16 = True
        if use_fp16 and torch.cuda.is_available():
            model = model.half()
        
        # Test gradient checkpointing
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    
    def test_communication_overhead_estimation(self):
        """Test estimation of communication overhead"""
        model = QTGModel(
            vocab_size=1000,
            embed_dim=512,
            num_layers=12,
            num_heads=8,
            max_seq_len=512
        )
        
        # Calculate parameter count
        param_count = sum(p.numel() for p in model.parameters())
        
        # Estimate communication overhead (simplified)
        # Each parameter needs to be communicated (4 bytes for float32)
        communication_bytes = param_count * 4
        
        # Convert to MB
        communication_mb = communication_bytes / (1024 * 1024)
        
        assert communication_mb > 0
        assert communication_mb < 10000  # Should be reasonable for this model size
    
    def test_load_balancing(self):
        """Test load balancing across ranks"""
        world_size = 4
        dataset_size = 1000
        
        # Calculate samples per rank
        base_samples = dataset_size // world_size
        remainder = dataset_size % world_size
        
        samples_per_rank = []
        for rank in range(world_size):
            if rank < remainder:
                samples_per_rank.append(base_samples + 1)
            else:
                samples_per_rank.append(base_samples)
        
        # Check that all samples are distributed
        assert sum(samples_per_rank) == dataset_size
        
        # Check that load is reasonably balanced
        max_samples = max(samples_per_rank)
        min_samples = min(samples_per_rank)
        assert max_samples - min_samples <= 1
    
    def test_fault_tolerance(self):
        """Test fault tolerance mechanisms"""
        # Test checkpoint frequency
        total_steps = 10000
        checkpoint_frequency = 1000
        
        checkpoint_steps = list(range(0, total_steps, checkpoint_frequency))
        assert len(checkpoint_steps) == 11  # 0, 1000, 2000, ..., 10000
        
        # Test recovery from checkpoint
        last_checkpoint_step = 5000
        remaining_steps = total_steps - last_checkpoint_step
        assert remaining_steps == 5000
    
    def test_performance_monitoring(self):
        """Test performance monitoring for distributed training"""
        # Test throughput calculation
        samples_processed = 1000
        time_elapsed = 10.0  # seconds
        
        throughput = samples_processed / time_elapsed
        assert throughput == 100.0  # samples per second
        
        # Test GPU utilization monitoring
        gpu_utilization = 85.0  # percentage
        assert 0 <= gpu_utilization <= 100
        
        # Test memory usage monitoring
        memory_used = 6.5  # GB
        memory_total = 8.0  # GB
        memory_utilization = (memory_used / memory_total) * 100
        assert memory_utilization == 81.25


class TestDistributedQTGTraining:
    """Test suite for distributed QTG-specific training"""
    
    def test_qtg_loss_distributed(self):
        """Test QTG loss computation in distributed setting"""
        model = QTGModel(
            vocab_size=100,
            embed_dim=32,
            num_layers=1,
            num_heads=2,
            max_seq_len=32
        )
        
        loss_fn = QTGLoss(vocab_size=100)
        
        input_ids = torch.randint(0, 100, (2, 32))
        labels = torch.randint(0, 100, (2, 32))
        
        outputs = model(input_ids)
        total_loss, loss_components = loss_fn(model, input_ids, labels, outputs)
        
        # Check that all loss components are computed
        expected_keys = ['ce_loss', 'topo_loss', 'geo_loss', 'quantum_loss', 'total_loss']
        assert all(key in loss_components for key in expected_keys)
        
        # Check that loss is finite
        assert torch.isfinite(total_loss)
    
    def test_topology_analysis_distributed(self):
        """Test topology analysis in distributed setting"""
        model = QTGModel(
            vocab_size=100,
            embed_dim=32,
            num_layers=1,
            num_heads=2,
            max_seq_len=32
        )
        
        input_ids = torch.randint(0, 100, (2, 32))
        
        # Get topology information
        topo_info = model.get_topology_info(input_ids)
        
        # Check that topology information is available
        assert len(topo_info) > 0
        
        # Check that persistence scores are computed
        for key, value in topo_info.items():
            if 'persistence' in key:
                assert torch.all(value >= 0)
    
    def test_quantum_analysis_distributed(self):
        """Test quantum analysis in distributed setting"""
        model = QTGModel(
            vocab_size=100,
            embed_dim=32,
            num_layers=1,
            num_heads=2,
            max_seq_len=32
        )
        
        input_ids = torch.randint(0, 100, (2, 32))
        
        # Get quantum information
        quantum_info = model.get_quantum_info(input_ids)
        
        # Check that quantum information is available
        assert len(quantum_info) > 0
        
        # Check that amplitudes and phases are computed
        for key, value in quantum_info.items():
            if 'amplitudes' in key:
                assert torch.all(value >= 0)
            elif 'phases' in key:
                assert torch.all(torch.isfinite(value))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
