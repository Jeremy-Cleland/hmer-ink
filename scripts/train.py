"""
Training script for HMER-Ink models.
"""

import os
import sys
import time
import logging
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hmer.config import load_config, get_optimizer, get_scheduler, get_device
from hmer.data.dataset import HMERDataset
from hmer.data.transforms import get_train_transforms, get_eval_transforms
from hmer.utils.tokenizer import LaTeXTokenizer
from hmer.models import create_model
from hmer.utils.metrics import compute_metrics


def train_epoch(model: nn.Module, 
               dataloader: DataLoader, 
               optimizer: optim.Optimizer,
               criterion: nn.Module,
               device: torch.device,
               epoch: int,
               use_amp: bool = False) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    # Create scaler for AMP
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch in pbar:
        # Get batch data
        input_seq = batch['input'].to(device)
        target_seq = batch['target'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with AMP if enabled
        if use_amp:
            with torch.cuda.amp.autocast():
                # Forward pass
                logits = model(input_seq, target_seq[:, :-1], input_lengths, target_lengths - 1)
                
                # Calculate loss
                # Reshape logits to [batch_size * seq_len, vocab_size]
                logits_flat = logits.reshape(-1, logits.shape[-1])
                
                # Reshape targets to [batch_size * seq_len]
                targets_flat = target_seq[:, 1:].reshape(-1)
                
                # Calculate loss
                loss = criterion(logits_flat, targets_flat)
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass
            logits = model(input_seq, target_seq[:, :-1], input_lengths, target_lengths - 1)
            
            # Calculate loss
            # Reshape logits to [batch_size * seq_len, vocab_size]
            logits_flat = logits.reshape(-1, logits.shape[-1])
            
            # Reshape targets to [batch_size * seq_len]
            targets_flat = target_seq[:, 1:].reshape(-1)
            
            # Calculate loss
            loss = criterion(logits_flat, targets_flat)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        batch_count += 1
        
        # Update progress bar
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    # Calculate average loss
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    
    return {
        'loss': avg_loss
    }


def validate(model: nn.Module, 
            dataloader: DataLoader, 
            criterion: nn.Module,
            tokenizer: LaTeXTokenizer,
            device: torch.device,
            beam_size: int = 4) -> Dict[str, float]:
    """
    Validate model performance.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        tokenizer: Tokenizer for decoding predictions
        device: Device to run on
        beam_size: Beam size for generation
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    batch_count = 0
    
    all_predictions = []
    all_targets = []
    
    # Progress bar
    pbar = tqdm(dataloader, desc="Validation")
    
    with torch.no_grad():
        for batch in pbar:
            # Get batch data
            input_seq = batch['input'].to(device)
            target_seq = batch['target'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            target_lengths = batch['target_lengths'].to(device)
            labels = batch['labels']
            
            # Forward pass for loss calculation
            logits = model(input_seq, target_seq[:, :-1], input_lengths, target_lengths - 1)
            
            # Calculate loss
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = target_seq[:, 1:].reshape(-1)
            loss = criterion(logits_flat, targets_flat)
            
            # Generate predictions with beam search
            beam_results, _ = model.generate(
                input_seq, input_lengths, max_length=128, beam_size=beam_size
            )
            
            # Decode predictions
            batch_predictions = []
            for beams in beam_results:
                # Take the top beam result (best prediction)
                top_beam = beams[0]
                prediction = tokenizer.decode(top_beam, skip_special_tokens=True)
                batch_predictions.append(prediction)
            
            # Add to lists for metric calculation
            all_predictions.extend(batch_predictions)
            all_targets.extend(labels)
            
            # Update statistics
            total_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    # Calculate average loss
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    
    # Calculate metrics
    metrics = compute_metrics(all_predictions, all_targets)
    metrics['loss'] = avg_loss
    
    return metrics


def train(config_path: str, 
         checkpoint_path: Optional[str] = None, 
         output_dir: Optional[str] = None) -> nn.Module:
    """
    Train a HMER model.
    
    Args:
        config_path: Path to configuration file
        checkpoint_path: Path to checkpoint for resuming training
        output_dir: Directory to save outputs
        
    Returns:
        Trained model
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set up output directory
    if output_dir is None:
        output_dir = config['output'].get('checkpoint_dir', 'outputs/checkpoints')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_dir = config['output'].get('log_dir', 'outputs/logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    
    # Get training parameters
    train_config = config['training']
    
    batch_size = train_config.get('batch_size', 32)
    num_epochs = train_config.get('num_epochs', 50)
    early_stopping = train_config.get('early_stopping_patience', 5)
    use_amp = train_config.get('use_amp', True)
    num_workers = train_config.get('num_workers', 4)
    grad_accum_steps = train_config.get('gradient_accumulation_steps', 1)
    save_every = train_config.get('save_every_n_epochs', 1)
    
    # Get device
    device = get_device(train_config)
    logging.info(f"Using device: {device}")
    
    # Set up data directories
    data_config = config['data']
    data_dir = data_config.get('data_dir', 'data')
    train_dirs = data_config.get('train_dirs', ['train', 'synthetic'])
    valid_dir = data_config.get('valid_dir', 'valid')
    
    # Create tokenizer
    tokenizer = LaTeXTokenizer()
    
    # Load or create vocabulary
    vocab_path = os.path.join(output_dir, 'vocab.json')
    if os.path.exists(vocab_path):
        logging.info(f"Loading vocabulary from {vocab_path}")
        tokenizer.load_vocab(vocab_path)
    else:
        logging.info("Building vocabulary from training data")
        
        # Get all LaTeX expressions from training data
        latex_expressions = []
        for dir_name in train_dirs:
            dir_path = os.path.join(data_dir, dir_name)
            if not os.path.exists(dir_path):
                continue
                
            # Sample a subset of files for vocabulary building
            import random
            file_list = [f for f in os.listdir(dir_path) if f.endswith('.inkml')]
            if len(file_list) > 10000:  # Limit to 10,000 files for efficiency
                file_list = random.sample(file_list, 10000)
            
            # Parse each file and extract labels
            from hmer.data.inkml import InkmlParser
            parser = InkmlParser()
            for filename in tqdm(file_list, desc=f"Building vocab from {dir_name}"):
                file_path = os.path.join(dir_path, filename)
                try:
                    data = parser.parse_inkml(file_path)
                    # Use normalized label if available, otherwise use label
                    label = data.get('normalized_label', '') or data.get('label', '')
                    if label:
                        latex_expressions.append(label)
                except Exception as e:
                    logging.warning(f"Error parsing {file_path}: {e}")
        
        # Build vocabulary
        tokenizer.build_vocab_from_data(latex_expressions, min_freq=5)
        
        # Save vocabulary
        tokenizer.save_vocab(vocab_path)
    
    logging.info(f"Vocabulary size: {len(tokenizer)}")
    
    # Create datasets
    logging.info("Creating datasets")
    
    # Get transforms
    train_transform = get_train_transforms(data_config)
    eval_transform = get_eval_transforms(data_config)
    
    # Normalization parameters
    normalize = True
    x_range = data_config.get('normalization', {}).get('x_range', (-1, 1))
    y_range = data_config.get('normalization', {}).get('y_range', (-1, 1))
    time_range = data_config.get('normalization', {}).get('time_range', (0, 1))
    
    # Create training dataset
    train_dataset = HMERDataset(
        data_dir=data_dir,
        split_dirs=train_dirs,
        tokenizer=tokenizer,
        max_seq_length=data_config.get('max_seq_length', 512),
        max_token_length=data_config.get('max_token_length', 128),
        transform=train_transform,
        normalize=normalize,
        use_relative_coords=True,
        x_range=x_range,
        y_range=y_range,
        time_range=time_range,
    )
    
    # Create validation dataset
    valid_dataset = HMERDataset(
        data_dir=data_dir,
        split_dirs=[valid_dir],
        tokenizer=tokenizer,
        max_seq_length=data_config.get('max_seq_length', 512),
        max_token_length=data_config.get('max_token_length', 128),
        transform=eval_transform,
        normalize=normalize,
        use_relative_coords=True,
        x_range=x_range,
        y_range=y_range,
        time_range=time_range,
    )
    
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(valid_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=valid_dataset.collate_fn
    )
    
    # Create model
    logging.info("Creating model")
    
    model = create_model(config['model'], len(tokenizer))
    model = model.to(device)
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(train_config, model.parameters())
    scheduler = get_scheduler(train_config, optimizer)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token (index 0)
    
    # Initialize training state
    start_epoch = 0
    best_metric = float('inf')
    
    # Load checkpoint if provided
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model'])
        
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'scheduler' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_metric = checkpoint.get('best_metric', float('inf'))
        
        logging.info(f"Resumed from epoch {start_epoch} with best metric {best_metric:.4f}")
    
    # Set up tensorboard if requested
    if config['output'].get('tensorboard', False):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
    else:
        writer = None
    
    # Set up wandb if requested
    if config['output'].get('use_wandb', False):
        import wandb
        wandb.init(
            project=config['output'].get('project_name', 'hmer-ink'),
            config=config
        )
    
    # Training loop
    logging.info("Starting training")
    
    no_improvement = 0
    
    for epoch in range(start_epoch, num_epochs):
        # Train for one epoch
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, use_amp
        )
        
        # Validate
        val_metrics = validate(
            model, valid_loader, criterion, tokenizer, device, 
            beam_size=config['evaluation'].get('beam_size', 4)
        )
        
        # Log metrics
        logging.info(f"Epoch {epoch}:")
        logging.info(f"  Train Loss: {train_metrics['loss']:.4f}")
        logging.info(f"  Valid Loss: {val_metrics['loss']:.4f}")
        logging.info(f"  Expression Recognition Rate: {val_metrics['expression_recognition_rate']:.4f}")
        logging.info(f"  Symbol Accuracy: {val_metrics['symbol_accuracy']:.4f}")
        logging.info(f"  Edit Distance: {val_metrics['edit_distance']:.4f}")
        
        # Update tensorboard
        if writer is not None:
            writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f"Valid/{key}", value, epoch)
        
        # Update wandb
        if config['output'].get('use_wandb', False):
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                **{f"val_{k}": v for k, v in val_metrics.items()}
            })
        
        # Update scheduler if using ReduceLROnPlateau
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Save model
        if epoch % save_every == 0 or epoch == num_epochs - 1:
            checkpoint_filename = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
            model.save_checkpoint(
                checkpoint_filename, epoch, optimizer, scheduler, val_metrics['loss']
            )
            logging.info(f"Saved checkpoint to {checkpoint_filename}")
        
        # Check for best model
        metric_name = 'loss'  # We can change this to ERR or another metric
        current_metric = val_metrics[metric_name]
        
        if current_metric < best_metric:
            best_metric = current_metric
            best_checkpoint = os.path.join(output_dir, "best_model.pt")
            model.save_checkpoint(
                best_checkpoint, epoch, optimizer, scheduler, best_metric
            )
            logging.info(f"New best model with {metric_name} = {best_metric:.4f}")
            no_improvement = 0
        else:
            no_improvement += 1
        
        # Early stopping
        if no_improvement >= early_stopping:
            logging.info(f"Early stopping after {epoch + 1} epochs")
            break
    
    # Close tensorboard writer
    if writer is not None:
        writer.close()
    
    # Close wandb
    if config['output'].get('use_wandb', False):
        wandb.finish()
    
    # Load best model for return
    best_checkpoint = os.path.join(output_dir, "best_model.pt")
    if os.path.exists(best_checkpoint):
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
        logging.info(f"Loaded best model with {metric_name} = {checkpoint.get('best_metric', -1):.4f}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train HMER-Ink model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for resuming training")
    parser.add_argument("--output_dir", type=str, help="Directory to save outputs")
    
    args = parser.parse_args()
    
    train(args.config, args.checkpoint, args.output_dir)