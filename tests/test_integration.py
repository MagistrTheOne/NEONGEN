import torch
import pytest
import tempfile
import os
import json
from pathlib import Path

# Import our modules
from src.models.qembed import QEmbed
from src.models.topofilter import TopoFilter
from src.models.geoattention import GeoAttention
from src.models.qtgm_model import QTGModel, QTGTransformerBlock
from src.training.loss_functions import QTGLoss, QTGTrainer
from src.training.data_loader import QTGTextDataset, MockQTGTokenizer
from src.utils.memory_utils import MemoryOptimizer, estimate_model_memory

class TestQTGIntegration:
    """Integration tests for QTG model components"""

    def test_qembed_topofilter_integration(self):
        """Test QEmbed and TopoFilter working together"""
        vocab_size = 1000
        embed_dim = 128

        # Initialize components
        qembed = QEmbed(vocab_size, embed_dim)
        topofilter = TopoFilter()

        # Create test input
        batch_size = 4
        seq_len = 16
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        x_onehot = torch.nn.functional.one_hot(x, num_classes=vocab_size).float()

        # Forward pass through both layers
        embeddings = qembed(x_onehot)
        filtered = topofilter(embeddings)

        # Check output properties
        assert embeddings.shape == (batch_size, seq_len, embed_dim)
        assert filtered.shape == (batch_size, seq_len, embed_dim)
        assert embeddings.dtype == torch.complex64
        assert filtered.dtype == torch.complex64

    def test_geoattention_integration(self):
        """Test GeoAttention standalone functionality"""
        embed_dim = 128
        num_heads = 8

        geo_attn = GeoAttention(embed_dim, num_heads)

        batch_size = 2
        seq_len = 16

        # Create complex embeddings
        embeddings = torch.complex(
            torch.randn(batch_size, seq_len, embed_dim),
            torch.randn(batch_size, seq_len, embed_dim)
        )

        # Forward pass
        output = geo_attn(embeddings)

        # Check output
        assert output.shape == embeddings.shape
        assert output.dtype == embeddings.dtype

        # Test attention map
        attention_map = geo_attn.get_attention_map(embeddings)
        expected_shape = (batch_size, num_heads, seq_len, seq_len)
        assert attention_map.shape == expected_shape

    def test_transformer_block_integration(self):
        """Test complete QTG transformer block"""
        embed_dim = 128
        num_heads = 8

        block = QTGTransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            persistence_threshold=0.1,
            tau=1.0,
            gamma=1.0,
            dropout=0.1
        )

        batch_size = 2
        seq_len = 12

        # Create complex input
        x = torch.complex(
            torch.randn(batch_size, seq_len, embed_dim),
            torch.randn(batch_size, seq_len, embed_dim)
        )

        # Forward pass
        output = block(x)

        # Check output
        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_full_model_integration(self):
        """Test complete QTG model"""
        vocab_size = 1000
        embed_dim = 64  # Small for testing
        num_layers = 2
        num_heads = 4
        max_seq_len = 32

        model = QTGModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len
        )

        batch_size = 2
        seq_len = 16

        # Create input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward pass
        outputs = model(input_ids)

        # Check outputs
        assert 'logits' in outputs
        assert 'last_hidden_state' in outputs
        assert 'attention_maps' in outputs

        assert outputs['logits'].shape == (batch_size, seq_len, vocab_size)
        assert outputs['last_hidden_state'].shape == (batch_size, seq_len, embed_dim)
        assert len(outputs['attention_maps']) == num_layers

    def test_loss_function_integration(self):
        """Test QTG loss function with model"""
        vocab_size = 500
        embed_dim = 64

        # Create model and loss
        model = QTGModel(vocab_size=vocab_size, embed_dim=embed_dim, num_layers=1, num_heads=4)
        loss_fn = QTGLoss(vocab_size=vocab_size)

        batch_size = 2
        seq_len = 8

        # Create test data
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward pass
        outputs = model(input_ids)

        # Compute loss
        total_loss, loss_components = loss_fn(model, input_ids, labels, outputs)

        # Check loss components
        expected_keys = ['ce_loss', 'topo_loss', 'geo_loss', 'quantum_loss', 'total_loss']
        assert all(key in loss_components for key in expected_keys)
        assert all(isinstance(loss_components[key], float) for key in expected_keys)

        # Total loss should be reasonable
        assert total_loss.item() > 0
        assert not torch.isnan(total_loss)

    def test_trainer_integration(self):
        """Test QTG trainer with model and loss"""
        vocab_size = 300
        embed_dim = 32

        # Create components
        model = QTGModel(vocab_size=vocab_size, embed_dim=embed_dim, num_layers=1, num_heads=2)
        loss_fn = QTGLoss(vocab_size=vocab_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        trainer = QTGTrainer(model, optimizer, loss_fn, 'cpu')

        batch_size = 1
        seq_len = 4

        # Create test data
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Training step
        loss_components = trainer.training_step(input_ids, labels)

        # Check results
        assert isinstance(loss_components, dict)
        assert 'total_loss' in loss_components

    def test_data_loader_integration(self):
        """Test data loading pipeline"""
        # Create temporary dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = os.path.join(temp_dir, "test_data.jsonl")

            # Create mock data
            data = [
                {"text": "The quick brown fox jumps over the lazy dog.", "id": 0},
                {"text": "Machine learning is a powerful tool.", "id": 1},
                {"text": "Quantum computing uses quantum mechanics.", "id": 2}
            ]

            with open(dataset_path, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')

            # Create dataset
            tokenizer = MockQTGTokenizer(vocab_size=1000)
            dataset = QTGTextDataset(
                data_path=dataset_path,
                tokenizer=tokenizer,
                max_length=16,
                stride=8
            )

            # Check dataset
            assert len(dataset) > 0
            assert 'input_ids' in dataset[0]
            assert 'labels' in dataset[0]
            assert 'text' in dataset[0]

    def test_memory_estimation(self):
        """Test memory estimation utilities"""
        vocab_size = 1000
        embed_dim = 128
        num_layers = 2

        model = QTGModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=8
        )

        batch_size = 4
        seq_len = 16

        memory_est = estimate_model_memory(model, batch_size, seq_len)

        # Check that all memory components are estimated
        expected_keys = ['parameters', 'activations', 'gradients', 'optimizer', 'total_estimate']
        assert all(key in memory_est for key in expected_keys)

        # Memory estimates should be positive
        assert all(memory_est[key] > 0 for key in expected_keys)

    def test_generation_integration(self):
        """Test text generation with QTG model"""
        vocab_size = 200
        embed_dim = 64

        model = QTGModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=1,
            num_heads=4
        )

        # Create trainer
        loss_fn = QTGLoss(vocab_size=vocab_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = QTGTrainer(model, optimizer, loss_fn, 'cpu')

        # Create prompt
        prompt_ids = torch.randint(0, vocab_size, (1, 5))

        # Generate
        generated = trainer.generate_sample(prompt_ids, max_length=10)

        # Check generation
        assert generated.shape[0] == prompt_ids.shape[0]  # Same batch size
        assert generated.shape[1] == prompt_ids.shape[1] + 10  # Prompt + generated tokens

    def test_topological_analysis(self):
        """Test topological analysis functionality"""
        vocab_size = 300
        embed_dim = 64

        model = QTGModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=1,
            num_heads=4
        )

        batch_size = 1
        seq_len = 8
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Get topological info
        topo_info = model.get_topology_info(input_ids)

        # Check that topological information is returned
        assert isinstance(topo_info, dict)
        assert len(topo_info) > 0

        # Check specific layer information
        expected_keys = ['layer_0_persistence', 'layer_0_filter_mask']
        assert all(key in topo_info for key in expected_keys)

    def test_quantum_analysis(self):
        """Test quantum analysis functionality"""
        vocab_size = 200
        embed_dim = 32

        model = QTGModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=1,
            num_heads=2
        )

        batch_size = 1
        seq_len = 6
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Get quantum info
        quantum_info = model.get_quantum_info(input_ids)

        # Check quantum information
        assert isinstance(quantum_info, dict)
        assert len(quantum_info) > 0

        # Check specific layer information
        expected_keys = ['layer_0_amplitudes', 'layer_0_phases']
        assert all(key in quantum_info for key in expected_keys)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
