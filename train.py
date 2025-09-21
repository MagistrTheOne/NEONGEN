#!/usr/bin/env python3
"""
Main training script for QTG (Quantum-Topological-Geometric) Model
Optimized for RTX 2080 Super (8GB VRAM)
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from training.trainer import QTGTrainingPipeline, quick_memory_test
from training.data_loader import create_mock_dataset, load_qtg_dataset, create_train_val_split

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train QTG Model")

    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to training data (JSONL format)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--memory_test",
        action="store_true",
        help="Run memory test and exit"
    )

    parser.add_argument(
        "--create_mock_data",
        action="store_true",
        help="Create mock dataset for testing"
    )

    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume training from checkpoint"
    )

    return parser.parse_args()

def setup_directories(output_dir: str):
    """Setup output directories"""
    dirs = [
        output_dir,
        f"{output_dir}/checkpoints",
        f"{output_dir}/logs",
        f"{output_dir}/samples"
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def main():
    """Main training function"""
    args = parse_args()

    # Setup directories
    setup_directories(args.output_dir)

    # Memory test
    if args.memory_test:
        print("Running memory test...")
        memory_ok = quick_memory_test(args.config)
        if memory_ok:
            print("✅ Memory test passed!")
            return
        else:
            print("❌ Memory test failed!")
            return

    # Create mock data if requested
    if args.create_mock_data:
        print("Creating mock dataset...")
        data_path = create_mock_dataset(
            num_samples=10000,
            max_length=512,
            output_path="data/mock_dataset.jsonl"
        )
        args.data_path = data_path

    # Check data path
    if not args.data_path:
        print("❌ No data path specified. Use --data_path or --create_mock_data")
        return

    if not os.path.exists(args.data_path):
        print(f"❌ Data file not found: {args.data_path}")
        return

    # Create train/val split
    print("Creating train/validation split...")
    train_path, val_path = create_train_val_split(args.data_path, train_ratio=0.9)

    # Load datasets
    print("Loading datasets...")
    train_dataset = load_qtg_dataset(
        data_path=train_path,
        max_length=512,
        stride=256,
        cache_dir="data/cache"
    )

    val_dataset = load_qtg_dataset(
        data_path=val_path,
        max_length=512,
        stride=256,
        cache_dir="data/cache"
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    # Create training pipeline
    print("Initializing training pipeline...")
    trainer = QTGTrainingPipeline(args.config)

    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

    # Get model info
    model_info = trainer.get_model_info()
    print("Model Information:")
    print(f"  Parameters: {model_info['total_parameters']:,}")
    print(f"  Model size: {model_info['model_size_mb']:.1f} MB")
    print(f"  GPU Memory: {model_info['gpu_memory_used_gb']:.1f}/{model_info['gpu_memory_total_gb']:.1f} GB")

    # Start training
    print("Starting training...")
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=args.num_epochs,
        checkpoint_dir=f"{args.output_dir}/checkpoints"
    )

    print("Training completed!")

if __name__ == "__main__":
    main()
