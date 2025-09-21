#!/usr/bin/env python3
"""
Distributed training script for NEON QTG models
Usage: torchrun --nproc_per_node=4 train_distributed.py --config config/neon_qtg_7b.yaml
"""

import argparse
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import os
import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

from training.distributed_trainer import (
    DistributedQTGTrainer,
    distributed_training_context,
    create_distributed_sampler,
    estimate_distributed_training_time
)
from training.loss_functions import QTGLoss
from models.qtgm_model import QTGModel
from training.data_loader import QTGTextDataset, MockQTGTokenizer


def load_config(config_path: str) -> dict:
    """Load training configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict) -> QTGModel:
    """Create QTG model from configuration"""
    model_config = config['model']
    
    model = QTGModel(
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
    
    return model


def create_optimizer(model: QTGModel, config: dict) -> torch.optim.Optimizer:
    """Create optimizer from configuration"""
    training_config = config['training']
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config['lr'],
        weight_decay=training_config['weight_decay'],
        betas=(training_config['beta1'], training_config['beta2'])
    )
    
    return optimizer


def create_loss_function(config: dict) -> QTGLoss:
    """Create loss function from configuration"""
    training_config = config['training']
    
    loss_fn = QTGLoss(
        vocab_size=config['model']['vocab_size'],
        lambda_topo=training_config['lambda_topo'],
        lambda_geo=training_config['lambda_geo'],
        lambda_quantum=training_config['lambda_quantum']
    )
    
    return loss_fn


def create_dataloader(config: dict, rank: int, world_size: int) -> DataLoader:
    """Create distributed dataloader"""
    data_config = config['data']
    memory_config = config['memory']
    
    # Create dataset
    dataset = QTGTextDataset(
        data_path=data_config.get('data_path', 'data/train.jsonl'),
        tokenizer=MockQTGTokenizer(vocab_size=config['model']['vocab_size']),
        max_length=data_config['max_length'],
        stride=data_config['stride']
    )
    
    # Create distributed sampler
    sampler = create_distributed_sampler(
        dataset,
        world_size=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=memory_config['batch_size'],
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    return dataloader


def train_worker(rank: int, world_size: int, config: dict, args):
    """Training worker function"""
    
    with distributed_training_context(rank, world_size):
        # Setup device
        device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        torch.cuda.set_device(rank)
        
        # Create model
        model = create_model(config)
        
        # Create optimizer
        optimizer = create_optimizer(model, config)
        
        # Create loss function
        loss_fn = create_loss_function(config)
        
        # Create distributed trainer
        trainer = DistributedQTGTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            world_size=world_size,
            rank=rank,
            use_mixed_precision=config['memory'].get('fp16', True),
            use_gradient_checkpointing=config['memory'].get('gradient_checkpointing', True)
        )
        
        # Create dataloader
        train_dataloader = create_dataloader(config, rank, world_size)
        
        # Load checkpoint if resuming
        start_epoch = 0
        start_step = 0
        if args.resume_from:
            checkpoint = trainer.load_checkpoint(args.resume_from)
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']
        
        # Training loop
        trainer.log(f"Starting distributed training on rank {rank}")
        trainer.log(f"World size: {world_size}")
        trainer.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Estimate training time
        model_size = sum(p.numel() for p in model.parameters())
        estimated_time = estimate_distributed_training_time(
            model_size=model_size,
            world_size=world_size,
            batch_size=config['memory']['batch_size'],
            sequence_length=config['data']['max_length'],
            steps=config['training']['total_steps']
        )
        trainer.log(f"Estimated training time: {estimated_time:.1f} hours")
        
        # Training epochs
        total_steps = config['training']['total_steps']
        gradient_accumulation_steps = config['memory']['gradient_accumulation_steps']
        eval_steps = config['evaluation']['eval_steps']
        save_steps = config['evaluation']['save_steps']
        
        step_count = start_step
        epoch_count = start_epoch
        
        while step_count < total_steps:
            epoch_count += 1
            train_dataloader.sampler.set_epoch(epoch_count)
            
            for batch_idx, batch in enumerate(train_dataloader):
                if step_count >= total_steps:
                    break
                
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                # Training step
                loss_components = trainer.training_step(
                    input_ids=input_ids,
                    labels=labels,
                    gradient_accumulation_steps=gradient_accumulation_steps
                )
                
                step_count += 1
                
                # Logging
                if step_count % config['logging']['log_steps'] == 0:
                    trainer.log(f"Step {step_count}/{total_steps}")
                    trainer.log(f"Loss: {loss_components['total_loss']:.4f}")
                    trainer.log(f"CE Loss: {loss_components['ce_loss']:.4f}")
                    trainer.log(f"Topo Loss: {loss_components['topo_loss']:.4f}")
                    trainer.log(f"Geo Loss: {loss_components['geo_loss']:.4f}")
                    trainer.log(f"Quantum Loss: {loss_components['quantum_loss']:.4f}")
                
                # Evaluation
                if step_count % eval_steps == 0:
                    trainer.log("Running evaluation...")
                    # Add validation logic here
                
                # Save checkpoint
                if step_count % save_steps == 0:
                    checkpoint_path = f"outputs/checkpoints/checkpoint_step_{step_count}.pt"
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    
                    is_best = loss_components['total_loss'] < trainer.best_loss
                    if is_best:
                        trainer.best_loss = loss_components['total_loss']
                    
                    trainer.save_checkpoint(
                        checkpoint_path=checkpoint_path,
                        epoch=epoch_count,
                        step=step_count,
                        is_best=is_best
                    )
                
                # Generate sample
                if step_count % (save_steps * 2) == 0:
                    sample_input = input_ids[:1, :10]  # First sample, first 10 tokens
                    generated = trainer.generate_sample(
                        input_ids=sample_input,
                        max_length=50,
                        temperature=0.8
                    )
                    trainer.log(f"Generated sample: {generated[0].tolist()}")
        
        trainer.log(f"Training completed on rank {rank}")
        trainer.log(f"Final step count: {step_count}")
        trainer.log(f"Best loss: {trainer.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Distributed QTG Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume_from", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--world_size", type=int, default=None, help="Number of processes")
    parser.add_argument("--rank", type=int, default=None, help="Process rank")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get world size and rank from environment if not provided
    world_size = args.world_size or int(os.environ.get('WORLD_SIZE', 1))
    rank = args.rank or int(os.environ.get('RANK', 0))
    
    if world_size == 1:
        # Single GPU training
        train_worker(0, 1, config, args)
    else:
        # Multi-GPU training
        mp.spawn(
            train_worker,
            args=(world_size, config, args),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    main()
