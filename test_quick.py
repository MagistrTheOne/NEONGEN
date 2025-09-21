#!/usr/bin/env python3
"""
Quick test script to verify QTG model components work correctly
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    try:
        from src.models.qembed import QEmbed
        from src.models.topofilter import TopoFilter
        from src.models.geoattention import GeoAttention
        from src.models.qtgm_model import QTGModel, QTGTransformerBlock
        from src.training.loss_functions import QTGLoss, QTGTrainer
        from src.training.data_loader import QTGTextDataset, MockQTGTokenizer
        from src.utils.memory_utils import MemoryOptimizer, estimate_model_memory
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_components():
    """Test basic component functionality"""
    print("Testing basic components...")

    try:
        from src.models.qembed import QEmbed
        from src.models.topofilter import TopoFilter
        from src.models.geoattention import GeoAttention

        # Test QEmbed with simple one-hot input
        vocab_size, embed_dim = 50, 32
        qembed = QEmbed(vocab_size, embed_dim)

        # Create simple one-hot input
        x = torch.eye(vocab_size).unsqueeze(0)  # (1, vocab_size, vocab_size)
        out = qembed(x)
        assert out.shape == (1, vocab_size, embed_dim), f"QEmbed shape mismatch: {out.shape}"
        assert out.dtype == torch.complex64, f"QEmbed dtype mismatch: {out.dtype}"

        # Test TopoFilter
        topofilter = TopoFilter()
        filtered = topofilter(out)
        assert filtered.shape == out.shape, f"TopoFilter shape mismatch: {filtered.shape}"
        assert filtered.dtype == torch.complex64, f"TopoFilter dtype mismatch: {filtered.dtype}"

        # Test GeoAttention
        geo_attn = GeoAttention(embed_dim, num_heads=2)
        attended = geo_attn(filtered)
        assert attended.shape == filtered.shape, f"GeoAttention shape mismatch: {attended.shape}"
        assert attended.dtype == torch.complex64, f"GeoAttention dtype mismatch: {attended.dtype}"

        print("✅ Basic components test passed")
        return True
    except Exception as e:
        print(f"❌ Basic components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test QTG model creation and forward pass"""
    print("Testing model creation...")

    try:
        from src.models.qtgm_model import QTGModel

        vocab_size = 100
        embed_dim = 32
        num_layers = 1
        num_heads = 2
        max_seq_len = 16

        model = QTGModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=0.0  # Disable dropout for testing
        )

        # Test forward pass with small batch
        batch_size = 1
        seq_len = 8
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        outputs = model(input_ids)

        # Check outputs
        assert 'logits' in outputs
        assert 'last_hidden_state' in outputs
        assert 'attention_maps' in outputs

        assert outputs['logits'].shape == (batch_size, seq_len, vocab_size)
        assert outputs['last_hidden_state'].shape == (batch_size, seq_len, embed_dim)
        assert len(outputs['attention_maps']) == num_layers

        print("✅ Model creation test passed")
        return True
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_estimation():
    """Test memory estimation"""
    print("Testing memory estimation...")

    try:
        from src.models.qtgm_model import QTGModel
        from src.utils.memory_utils import estimate_model_memory

        model = QTGModel(vocab_size=1000, embed_dim=128, num_layers=2, num_heads=4)
        mem_est = estimate_model_memory(model, batch_size=4, seq_len=32)

        required_keys = ['parameters', 'activations', 'gradients', 'optimizer', 'total_estimate']
        assert all(key in mem_est for key in required_keys), "Missing memory estimation keys"

        print("✅ Memory estimation test passed")
        print(".1f")
        return True
    except Exception as e:
        print(f"❌ Memory estimation test failed: {e}")
        return False

def test_loss_function():
    """Test QTG loss function"""
    print("Testing loss function...")

    try:
        from src.models.qtgm_model import QTGModel
        from src.training.loss_functions import QTGLoss

        vocab_size = 100
        model = QTGModel(vocab_size=vocab_size, embed_dim=32, num_layers=1, num_heads=2, dropout=0.0)
        loss_fn = QTGLoss(vocab_size=vocab_size)

        batch_size = 2
        seq_len = 8

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        outputs = model(input_ids)
        total_loss, loss_components = loss_fn(model, input_ids, labels, outputs)

        expected_keys = ['ce_loss', 'topo_loss', 'geo_loss', 'quantum_loss', 'total_loss']
        assert all(key in loss_components for key in expected_keys), "Missing loss components"

        assert not torch.isnan(total_loss), "Loss is NaN"
        assert total_loss.item() > 0, "Loss is not positive"

        print("✅ Loss function test passed")
        return True
    except Exception as e:
        print(f"❌ Loss function test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 QTG Model Quick Test Suite")
    print("=" * 50)

    tests = [
        test_imports,
        test_basic_components,
        test_model_creation,
        test_memory_estimation,
        test_loss_function
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! QTG model is ready for training.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
