import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
from tqdm import tqdm
import json

from models.qtgm_model import QTGModel
from training.loss_functions import QTGLoss, QTGTrainer
from utils.memory_utils import MemoryOptimizer, GradientAccumulator, MixedPrecisionTrainer

class QTGTrainingPipeline:
    """
    Complete training pipeline for QTG models
    Optimized for RTX 2080 Super with memory management
    """

    def __init__(self, config_path: str):
        """
        Initialize training pipeline

        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_optimizer = MemoryOptimizer(str(self.device))

        # Setup logging
        self._setup_logging()

        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.trainer = None
        self.memory_trainer = None

        self._init_components()

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _init_components(self):
        """Initialize all training components"""
        model_config = self.config['model']
        training_config = self.config['training']
        memory_config = self.config['memory']

        # Initialize model
        self.logger.info("Initializing QTG Model...")
        self.model = QTGModel(
            vocab_size=model_config['vocab_size'],
            embed_dim=model_config['embed_dim'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            max_seq_len=model_config['max_seq_len'],
            persistence_threshold=model_config['persistence_threshold'],
            tau=model_config['tau'],
            gamma=model_config['gamma'],
            dropout=model_config['dropout']
        )

        # Memory optimization
        self.logger.info("Applying memory optimizations...")
        self.model = self.memory_optimizer.optimize_model_for_memory(self.model)
        self.model.to(self.device)

        # Initialize optimizer
        self.logger.info("Initializing optimizer...")
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=training_config['lr'],
            weight_decay=training_config['weight_decay'],
            betas=(training_config['beta1'], training_config['beta2'])
        )

        # Initialize loss function
        self.logger.info("Initializing loss function...")
        self.loss_fn = QTGLoss(
            vocab_size=model_config['vocab_size'],
            lambda_topo=training_config['lambda_topo'],
            lambda_geo=training_config['lambda_geo'],
            lambda_quantum=training_config['lambda_quantum']
        )

        # Initialize specialized trainer
        self.logger.info("Initializing QTG trainer...")
        self.trainer = QTGTrainer(self.model, self.optimizer, self.loss_fn, str(self.device))

        # Initialize mixed precision training
        self.logger.info("Initializing mixed precision training...")
        self.memory_trainer = MixedPrecisionTrainer(use_fp16=memory_config['fp16'])

        # Initialize gradient accumulator
        self.grad_accumulator = GradientAccumulator(memory_config['gradient_accumulation_steps'])

        # Initialize scheduler
        self.logger.info("Initializing learning rate scheduler...")
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=training_config['total_steps'],
            eta_min=training_config['lr'] * 0.1
        )

        # Log memory usage
        self.memory_optimizer.log_memory_stats(step=0)

    def create_dataloader(self, dataset, is_training: bool = True) -> DataLoader:
        """Create optimized DataLoader"""
        data_config = self.config['data']
        memory_config = self.config['memory']

        batch_size = data_config['batch_size'] if is_training else self.config['evaluation']['eval_batch_size']

        return self.memory_optimizer.create_memory_efficient_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=memory_config.get('num_workers', 4),
            pin_memory=memory_config.get('use_pinned_memory', True),
            prefetch_factor=memory_config.get('prefetch_factor', 2)
        )

    def train_epoch(self, train_dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        total_losses = {
            'ce_loss': 0.0,
            'topo_loss': 0.0,
            'geo_loss': 0.0,
            'quantum_loss': 0.0,
            'total_loss': 0.0
        }

        num_batches = len(train_dataloader)

        with tqdm(train_dataloader, desc=f"Epoch {epoch}") as pbar:
            for step, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Mixed precision training
                with self.memory_trainer.autocast():
                    loss_components = self.trainer.training_step(input_ids, labels)

                # Gradient accumulation
                should_update = self.grad_accumulator.accumulate(
                    torch.tensor(loss_components['total_loss'], device=self.device)
                )

                if should_update:
                    # Gradient clipping
                    self.memory_trainer.unscale_gradients(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # Optimizer step
                    self.memory_trainer.step(self.optimizer)
                    self.scheduler.step()

                    # Reset gradients
                    self.optimizer.zero_grad()

                # Accumulate losses
                for key in total_losses:
                    total_losses[key] += loss_components[key]

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_components['total_loss']:.4f}",
                    'ce': f"{loss_components['ce_loss']:.4f}",
                    'topo': f"{loss_components['topo_loss']:.4f}"
                })

                # Memory cleanup
                if step % self.config['memory']['empty_cache_steps'] == 0:
                    self.memory_optimizer.cleanup_memory()

        # Average losses
        for key in total_losses:
            total_losses[key] /= num_batches

        return total_losses

    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Run validation"""
        self.model.eval()

        total_losses = {
            'ce_loss': 0.0,
            'topo_loss': 0.0,
            'geo_loss': 0.0,
            'quantum_loss': 0.0,
            'total_loss': 0.0
        }

        num_batches = len(val_dataloader)

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                loss_components = self.trainer.validation_step(input_ids, labels)

                for key in total_losses:
                    total_losses[key] += loss_components[key]

        # Average losses
        for key in total_losses:
            total_losses[key] /= num_batches

        return total_losses

    def generate_samples(self, num_samples: int = 5, max_length: int = 50) -> list:
        """Generate sample texts for monitoring"""
        self.model.eval()

        samples = []
        eval_config = self.config['evaluation']

        for i in range(num_samples):
            # Create random prompt
            prompt_length = torch.randint(5, 20, (1,))
            prompt_ids = torch.randint(0, self.config['model']['vocab_size'], (1, prompt_length.item()))

            prompt_ids = prompt_ids.to(self.device)

            # Generate
            generated = self.trainer.generate_sample(
                input_ids=prompt_ids,
                max_length=max_length,
                temperature=eval_config['temperature']
            )

            # Convert to list for JSON serialization
            sample = {
                'prompt': prompt_ids.squeeze().cpu().tolist(),
                'generated': generated.squeeze().cpu().tolist(),
                'prompt_text': f"Token sequence of length {prompt_length.item()}",
                'generated_text': f"Generated sequence of length {len(generated.squeeze())}"
            }
            samples.append(sample)

        return samples

    def save_checkpoint(self, epoch: int, losses: Dict[str, float], path: str = "checkpoints"):
        """Save model checkpoint"""
        Path(path).mkdir(exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'losses': losses,
            'config': self.config
        }

        checkpoint_path = f"{path}/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save latest
        torch.save(checkpoint, f"{path}/checkpoint_latest.pt")

        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return 0

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        losses = checkpoint.get('losses', {})

        self.logger.info(f"Checkpoint loaded from epoch {epoch}")
        return epoch

    def train(
        self,
        train_dataset,
        val_dataset=None,
        num_epochs: int = 10,
        checkpoint_dir: str = "checkpoints"
    ):
        """Main training loop"""
        self.logger.info("Starting QTG model training...")

        # Create dataloaders
        train_dataloader = self.create_dataloader(train_dataset, is_training=True)
        val_dataloader = self.create_dataloader(val_dataset, is_training=False) if val_dataset else None

        # Training state
        best_val_loss = float('inf')
        global_step = 0

        # Training loop
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Train epoch
            train_losses = self.train_epoch(train_dataloader, epoch)
            self.logger.info(f"Train losses: {train_losses}")

            # Validate
            if val_dataloader:
                val_losses = self.validate(val_dataloader)
                self.logger.info(f"Validation losses: {val_losses}")

                # Save best model
                if val_losses['total_loss'] < best_val_loss:
                    best_val_loss = val_losses['total_loss']
                    self.save_checkpoint(epoch, val_losses, checkpoint_dir)
            else:
                # Save periodic checkpoints
                if (epoch + 1) % 5 == 0:
                    self.save_checkpoint(epoch, train_losses, checkpoint_dir)

            # Generate samples for monitoring
            if (epoch + 1) % 2 == 0:
                samples = self.generate_samples(num_samples=3)
                self.logger.info(f"Sample generation at epoch {epoch + 1}")

            # Memory stats
            self.memory_optimizer.log_memory_stats(step=epoch)

        self.logger.info("Training completed!")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        memory_stats = self.memory_optimizer.get_gpu_memory()

        return {
            'total_parameters': param_count,
            'trainable_parameters': trainable_params,
            'model_size_mb': param_count * 4 / (1024**2),  # float32
            'gpu_memory_used_gb': memory_stats['used'],
            'gpu_memory_total_gb': memory_stats['total'],
            'config': self.config
        }


def create_training_pipeline(config_path: str) -> QTGTrainingPipeline:
    """
    Factory function for creating training pipeline

    Args:
        config_path (str): Path to configuration file

    Returns:
        QTGTrainingPipeline: Configured training pipeline
    """
    return QTGTrainingPipeline(config_path)


def quick_memory_test(config_path: str = "config/model_config.yaml") -> bool:
    """
    Quick test to check if model fits in memory

    Args:
        config_path (str): Path to configuration file

    Returns:
        bool: True if model fits in memory
    """
    try:
        pipeline = QTGTrainingPipeline(config_path)
        info = pipeline.get_model_info()

        memory_ok = info['gpu_memory_used_gb'] < 7.0  # Leave 1GB buffer

        print(f"Model loaded successfully: {info['total_parameters']:,} parameters")
        print(f"GPU Memory: {info['gpu_memory_used_gb']:.1f}/{info['gpu_memory_total_gb']:.1f}GB")
        print(f"Memory OK: {memory_ok}")

        return memory_ok

    except Exception as e:
        print(f"Memory test failed: {e}")
        return False
