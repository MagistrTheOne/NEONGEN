#!/usr/bin/env python3
"""
Comprehensive tests for memory optimization utilities
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.memory_utils import (
    MemoryOptimizer, 
    GradientAccumulator, 
    MixedPrecisionTrainer,
    estimate_model_memory,
    create_optimized_config
)
from models.qtgm_model import QTGModel


class TestMemoryOptimizer:
    """Test suite for MemoryOptimizer"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.memory_optimizer = MemoryOptimizer(device='cpu')  # Use CPU for testing
    
    def test_gpu_memory_detection(self):
        """Test GPU memory detection"""
        with patch('torch.cuda.is_available', return_value=False):
            memory = self.memory_optimizer.get_gpu_memory()
            assert memory['used'] == 0.0
            assert memory['total'] == 0.0
            assert memory['free'] == 0.0
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('GPUtil.getGPUs')
    def test_gpu_memory_with_cuda(self, mock_get_gpus):
        """Test GPU memory detection with CUDA available"""
        # Mock GPU info
        mock_gpu = MagicMock()
        mock_gpu.memoryUsed = 2048  # MB
        mock_gpu.memoryTotal = 8192  # MB
        mock_gpu.memoryFree = 6144  # MB
        mock_get_gpus.return_value = [mock_gpu]
        
        memory = self.memory_optimizer.get_gpu_memory()
        assert memory['used'] == 2048
        assert memory['total'] == 8192
        assert memory['free'] == 6144
    
    def test_cpu_memory_detection(self):
        """Test CPU memory detection"""
        memory = self.memory_optimizer.get_cpu_memory()
        
        assert 'used' in memory
        assert 'total' in memory
        assert 'free' in memory
        assert memory['used'] >= 0
        assert memory['total'] > 0
        assert memory['free'] >= 0
    
    def test_memory_monitor_context(self):
        """Test memory monitoring context manager"""
        with patch.object(self.memory_optimizer, 'get_gpu_memory') as mock_get_memory:
            mock_get_memory.side_effect = [
                {'used': 1000, 'total': 8000, 'free': 7000},  # Before
                {'used': 2000, 'total': 8000, 'free': 6000}   # After
            ]
            
            with self.memory_optimizer.memory_monitor("test_step"):
                pass  # Do nothing, just test the context manager
    
    def test_optimize_model_for_memory(self):
        """Test model optimization for memory"""
        model = nn.Linear(10, 5)
        optimized_model = self.memory_optimizer.optimize_model_for_memory(model)
        
        assert optimized_model is not None
        assert isinstance(optimized_model, nn.Module)
    
    def test_create_memory_efficient_dataloader(self):
        """Test creating memory-efficient DataLoader"""
        # Create dummy dataset
        dataset = torch.utils.data.TensorDataset(torch.randn(100, 10))
        
        dataloader = self.memory_optimizer.create_memory_efficient_dataloader(
            dataset, batch_size=4, num_workers=0  # Use 0 workers for testing
        )
        
        assert isinstance(dataloader, torch.utils.data.DataLoader)
        assert dataloader.batch_size == 4
    
    def test_cleanup_memory(self):
        """Test memory cleanup"""
        # Should not raise any exceptions
        self.memory_optimizer.cleanup_memory()
    
    def test_log_memory_stats(self):
        """Test memory statistics logging"""
        with patch('builtins.print') as mock_print:
            self.memory_optimizer.log_memory_stats(step=100)
            mock_print.assert_called()


class TestGradientAccumulator:
    """Test suite for GradientAccumulator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.accumulator = GradientAccumulator(accumulation_steps=4)
    
    def test_initialization(self):
        """Test accumulator initialization"""
        assert self.accumulator.accumulation_steps == 4
        assert self.accumulator.step_count == 0
        assert self.accumulator.scaled_loss == 0.0
    
    def test_accumulation_cycle(self):
        """Test complete accumulation cycle"""
        # Create dummy loss
        loss = torch.tensor(1.0, requires_grad=True)
        
        # First 3 steps should not trigger update
        for i in range(3):
            should_update = self.accumulator.accumulate(loss)
            assert should_update == False
            assert self.accumulator.step_count == i + 1
        
        # 4th step should trigger update
        should_update = self.accumulator.accumulate(loss)
        assert should_update == True
        assert self.accumulator.step_count == 0  # Reset
    
    def test_reset(self):
        """Test accumulator reset"""
        self.accumulator.step_count = 3
        self.accumulator.reset()
        
        assert self.accumulator.step_count == 0
        assert self.accumulator.scaled_loss == 0.0


class TestMixedPrecisionTrainer:
    """Test suite for MixedPrecisionTrainer"""
    
    def test_initialization_with_fp16(self):
        """Test initialization with FP16 enabled"""
        with patch('torch.cuda.is_available', return_value=True):
            trainer = MixedPrecisionTrainer(use_fp16=True)
            assert trainer.use_fp16 == True
            assert trainer.scaler is not None
    
    def test_initialization_without_fp16(self):
        """Test initialization without FP16"""
        trainer = MixedPrecisionTrainer(use_fp16=False)
        assert trainer.use_fp16 == False
        assert trainer.scaler is None
    
    def test_autocast_context(self):
        """Test autocast context manager"""
        trainer = MixedPrecisionTrainer(use_fp16=False)
        
        with trainer.autocast():
            pass  # Should not raise any exceptions
    
    def test_scale_loss(self):
        """Test loss scaling"""
        trainer = MixedPrecisionTrainer(use_fp16=False)
        loss = torch.tensor(1.0)
        
        scaled_loss = trainer.scale_loss(loss)
        assert scaled_loss is loss  # Should return same tensor when no scaling
    
    def test_step_without_scaler(self):
        """Test optimizer step without scaler"""
        trainer = MixedPrecisionTrainer(use_fp16=False)
        optimizer = torch.optim.Adam([torch.tensor(1.0, requires_grad=True)])
        
        # Should not raise any exceptions
        trainer.step(optimizer)
    
    def test_unscale_gradients(self):
        """Test gradient unscaling"""
        trainer = MixedPrecisionTrainer(use_fp16=False)
        optimizer = torch.optim.Adam([torch.tensor(1.0, requires_grad=True)])
        
        # Should not raise any exceptions
        trainer.unscale_gradients(optimizer)


class TestMemoryEstimation:
    """Test suite for memory estimation functions"""
    
    def test_estimate_model_memory(self):
        """Test model memory estimation"""
        model = QTGModel(
            vocab_size=1000,
            embed_dim=128,
            num_layers=2,
            num_heads=4,
            max_seq_len=256
        )
        
        memory_est = estimate_model_memory(model, batch_size=4, seq_len=32)
        
        required_keys = ['parameters', 'activations', 'gradients', 'optimizer', 'total_estimate']
        assert all(key in memory_est for key in required_keys)
        
        # All estimates should be positive
        for key, value in memory_est.items():
            assert value > 0, f"{key} should be positive"
    
    def test_create_optimized_config(self):
        """Test creating optimized configuration"""
        config = create_optimized_config(gpu_memory_gb=8.0)
        
        required_keys = ['embed_dim', 'num_layers', 'num_heads', 'batch_size', 'seq_len', 'fp16']
        assert all(key in config for key in required_keys)
        
        # Test scaling for different GPU memory sizes
        config_small = create_optimized_config(gpu_memory_gb=4.0)
        config_large = create_optimized_config(gpu_memory_gb=16.0)
        
        # Smaller GPU should have smaller parameters
        assert config_small['embed_dim'] <= config['embed_dim']
        assert config_small['num_layers'] <= config['num_layers']
        assert config_small['batch_size'] <= config['batch_size']
        
        # Larger GPU should have larger parameters
        assert config_large['embed_dim'] >= config['embed_dim']
        assert config_large['num_layers'] >= config['num_layers']
        assert config_large['batch_size'] >= config['batch_size']


class TestMemoryIntegration:
    """Integration tests for memory optimization"""
    
    def test_full_training_cycle_memory(self):
        """Test memory optimization in full training cycle"""
        # Create model
        model = QTGModel(
            vocab_size=100,
            embed_dim=32,
            num_layers=1,
            num_heads=2,
            max_seq_len=64
        )
        
        # Create optimizer and loss
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = nn.CrossEntropyLoss()
        
        # Create memory optimizer
        memory_optimizer = MemoryOptimizer(device='cpu')
        
        # Create mixed precision trainer
        mp_trainer = MixedPrecisionTrainer(use_fp16=False)
        
        # Create gradient accumulator
        accumulator = GradientAccumulator(accumulation_steps=2)
        
        # Training step
        input_ids = torch.randint(0, 100, (2, 32))
        labels = torch.randint(0, 100, (2, 32))
        
        with memory_optimizer.memory_monitor("training_step"):
            outputs = model(input_ids)
            logits = outputs['logits']
            loss = loss_fn(logits.view(-1, 100), labels.view(-1))
            
            scaled_loss = mp_trainer.scale_loss(loss)
            should_update = accumulator.accumulate(scaled_loss)
            
            if should_update:
                mp_trainer.step(optimizer)
                accumulator.reset()
        
        # Should complete without errors
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
