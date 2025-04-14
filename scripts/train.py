"""
Training script for HMER-Ink models.
"""

import logging
import os
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hmer.config import get_device, get_optimizer, get_scheduler, load_config
from hmer.data.dataset import HMERDataset
from hmer.data.transforms import get_eval_transforms, get_train_transforms
from hmer.models import create_model
from hmer.utils.metrics import compute_metrics
from hmer.utils.tokenizer import LaTeXTokenizer


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_amp: bool = False,
    grad_accum_steps: int = 1,
    clip_grad_norm: float = 0.0,
    benchmark_mode: bool = False,
) -> Dict[str, float]:
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
        grad_accum_steps: Number of steps to accumulate gradients
        clip_grad_norm: Value for gradient clipping (0 to disable)
        benchmark_mode: Whether to use benchmark mode for faster training

    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    batch_count = 0

    # Enable benchmark mode if requested (faster on MPS)
    if benchmark_mode and device.type == "mps":
        torch.backends.cudnn.benchmark = True

    # Import correct AMP implementation
    from torch.amp import GradScaler, autocast

    # Create device-appropriate scaler for AMP
    scaler = None
    if use_amp:
        if device.type in ["cuda", "mps"]:
            scaler = GradScaler()
        else:  # CPU
            logging.info("AMP not supported on CPU, training without it")
            use_amp = False

    # Enable graph mode for MPS if available
    if device.type == "mps" and hasattr(torch.backends.mps, "enable_graph_mode"):
        torch.backends.mps.enable_graph_mode(True)

    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    # For gradient accumulation
    optimizer.zero_grad()
    accum_count = 0

    for batch_idx, batch in enumerate(pbar):
        # Get batch data
        input_seq = batch["input"].to(device)
        target_seq = batch["target"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        # Forward pass with AMP if enabled
        if use_amp:
            with autocast(device_type=device.type):
                # Forward pass
                logits = model(
                    input_seq, target_seq[:, :-1], input_lengths, target_lengths - 1
                )

                # Calculate loss
                # Reshape logits to [batch_size * seq_len, vocab_size]
                logits_flat = logits.reshape(-1, logits.shape[-1])

                # Reshape targets to [batch_size * seq_len]
                targets_flat = target_seq[:, 1:].reshape(-1)

                # Calculate loss and normalize by gradient accumulation steps
                loss = criterion(logits_flat, targets_flat) / grad_accum_steps

            # Backward pass with scaler
            scaler.scale(loss).backward()
        else:
            # Forward pass (without AMP)
            logits = model(
                input_seq, target_seq[:, :-1], input_lengths, target_lengths - 1
            )

            # Calculate loss
            # Reshape logits to [batch_size * seq_len, vocab_size]
            logits_flat = logits.reshape(-1, logits.shape[-1])

            # Reshape targets to [batch_size * seq_len]
            targets_flat = target_seq[:, 1:].reshape(-1)

            # Calculate loss and normalize by gradient accumulation steps
            loss = criterion(logits_flat, targets_flat) / grad_accum_steps

            # Backward pass
            loss.backward()

        # Update statistics (use the full loss value for reporting)
        total_loss += loss.item() * grad_accum_steps
        batch_count += 1
        accum_count += 1

        # Update progress bar
        pbar.set_postfix(loss=f"{loss.item() * grad_accum_steps:.4f}")

        # Step optimizer after accumulating gradients
        if accum_count == grad_accum_steps or batch_idx == len(dataloader) - 1:
            # Apply gradient clipping if configured
            if clip_grad_norm > 0:
                if use_amp:
                    # Unscale before clipping
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            # Step optimizer
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # Reset gradients and accumulation counter
            optimizer.zero_grad()
            accum_count = 0

    # Calculate average loss
    avg_loss = total_loss / batch_count if batch_count > 0 else 0

    return {"loss": avg_loss}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    tokenizer: LaTeXTokenizer,
    device: torch.device,
    beam_size: int = 4,
    fast_mode: bool = True,
    max_samples: int = 100,  # Limit validation samples in fast mode
) -> Dict[str, float]:
    """
    Validate model performance.

    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        tokenizer: Tokenizer for decoding predictions
        device: Device to run on
        beam_size: Beam size for generation
        fast_mode: Whether to use fast validation mode (smaller beam size & limited samples)
        max_samples: Maximum number of samples to validate in fast mode

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    batch_count = 0

    all_predictions = []
    all_targets = []

    # In fast mode, use smaller beam size during training to speed up validation
    if fast_mode:
        effective_beam_size = min(2, beam_size)  # Use at most beam size 2 in fast mode
        max_gen_length = 64  # Shorter max length for faster generation
    else:
        effective_beam_size = beam_size
        max_gen_length = 128

    # Progress bar
    pbar = tqdm(dataloader, desc="Validation")

    sample_count = 0
    with torch.no_grad():
        for batch in pbar:
            # Get batch data
            input_seq = batch["input"].to(device)
            target_seq = batch["target"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            labels = batch["labels"]

            # Forward pass for loss calculation
            logits = model(
                input_seq, target_seq[:, :-1], input_lengths, target_lengths - 1
            )

            # Calculate loss
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = target_seq[:, 1:].reshape(-1)
            loss = criterion(logits_flat, targets_flat)

            # In fast mode, limit number of samples for beam search
            if fast_mode and sample_count >= max_samples:
                # Still update loss but skip beam search
                total_loss += loss.item()
                batch_count += 1
                continue

            # For large batches in fast mode, only use part of the batch for generation
            if fast_mode and input_seq.size(0) > 8:
                # Use a smaller subset of the batch for generation
                gen_indices = list(range(min(8, input_seq.size(0))))
                gen_input_seq = input_seq[gen_indices]
                gen_input_lengths = input_lengths[gen_indices]
                gen_labels = [labels[i] for i in gen_indices]
            else:
                gen_input_seq = input_seq
                gen_input_lengths = input_lengths
                gen_labels = labels

            # Generate predictions with beam search (faster settings in fast mode)
            beam_results, _ = model.generate(
                gen_input_seq,
                gen_input_lengths,
                max_length=max_gen_length,
                beam_size=effective_beam_size,
                fast_mode=True,  # Enable fast generation mode
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
            all_targets.extend(gen_labels)
            sample_count += len(batch_predictions)

            # Update statistics
            total_loss += loss.item()
            batch_count += 1

            # Update progress bar
            pbar.set_postfix(loss=f"{loss.item():.4f}", samples=sample_count)

    # Calculate average loss
    avg_loss = total_loss / batch_count if batch_count > 0 else 0

    # Calculate metrics
    metrics = compute_metrics(all_predictions, all_targets)
    metrics["loss"] = avg_loss
    metrics["num_samples"] = sample_count

    return metrics


def train(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> nn.Module:
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

    # Apply MPS configuration if specified (these map to environment variables)
    mps_config = config.get("mps_configuration", {})
    if mps_config:
        # Set MPS environment variables
        if mps_config.get("enable_mps_fallback", True):
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        if mps_config.get("verbose", True):
            os.environ["PYTORCH_MPS_VERBOSE"] = "1"

        high_watermark = mps_config.get("high_watermark_ratio", 0.0)
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(high_watermark)

        if mps_config.get("prefer_channels_last", True):
            os.environ["PYTORCH_PREFER_CHANNELS_LAST"] = "1"
        if mps_config.get("enable_early_graph_capture", True):
            os.environ["PYTORCH_MPS_ENABLE_EARLY_GRAPH_CAPTURE"] = "1"
        if mps_config.get("separate_device_alloc", True):
            os.environ["PYTORCH_MPS_SEPARATE_DEVICE_ALLOC"] = "1"
        if mps_config.get("use_system_allocator", True):
            os.environ["PYTORCH_MPS_USE_SYSTEM_ALLOCATOR"] = "1"

        logging.info("Applied MPS-specific optimizations from config")

    # Set up output directory
    if output_dir is None:
        output_dir = config["output"].get("checkpoint_dir", "outputs/checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging
    log_dir = config["output"].get("log_dir", "outputs/logs")
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler(),
        ],
    )

    # Get training parameters
    train_config = config["training"]

    batch_size = train_config.get("batch_size", 32)
    num_epochs = train_config.get("num_epochs", 50)
    early_stopping = train_config.get("early_stopping_patience", 5)
    use_amp = train_config.get("use_amp", True)
    num_workers = train_config.get("num_workers", 4)
    grad_accum_steps = train_config.get("gradient_accumulation_steps", 1)
    save_every = train_config.get("save_every_n_epochs", 1)

    # Get device
    device = get_device(train_config)
    logging.info(f"Using device: {device}")

    # Set up data directories
    data_config = config["data"]
    data_dir = data_config.get("data_dir", "data")
    train_dirs = data_config.get("train_dirs", ["train", "synthetic"])
    valid_dir = data_config.get("valid_dir", "valid")

    # Create tokenizer
    tokenizer = LaTeXTokenizer()

    # Load or create vocabulary
    vocab_path = os.path.join(output_dir, "vocab.json")
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

            file_list = [f for f in os.listdir(dir_path) if f.endswith(".inkml")]
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
                    label = data.get("normalized_label", "") or data.get("label", "")
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
    x_range = data_config.get("normalization", {}).get("x_range", (-1, 1))
    y_range = data_config.get("normalization", {}).get("y_range", (-1, 1))
    time_range = data_config.get("normalization", {}).get("time_range", (0, 1))

    # Create training dataset
    train_dataset = HMERDataset(
        data_dir=data_dir,
        split_dirs=train_dirs,
        tokenizer=tokenizer,
        max_seq_length=data_config.get("max_seq_length", 512),
        max_token_length=data_config.get("max_token_length", 128),
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
        max_seq_length=data_config.get("max_seq_length", 512),
        max_token_length=data_config.get("max_token_length", 128),
        transform=eval_transform,
        normalize=normalize,
        use_relative_coords=True,
        x_range=x_range,
        y_range=y_range,
        time_range=time_range,
    )

    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(valid_dataset)}")

    # Create data loaders with optimized settings
    prefetch_factor = config.get("mps_options", {}).get("prefetch_factor", 2)
    persistent_workers = config.get("mps_options", {}).get("persistent_workers", False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=train_dataset.collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=valid_dataset.collate_fn,
    )

    # Create model
    logging.info("Creating model")

    model = create_model(config["model"], len(tokenizer))
    model = model.to(device)

    # Create optimizer and scheduler
    optimizer = get_optimizer(train_config, model.parameters())
    scheduler = get_scheduler(train_config, optimizer)

    # Create loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token (index 0)

    # Initialize training state
    start_epoch = 0
    best_metric = float("inf")

    # Load checkpoint if provided
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model"])

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        if "scheduler" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])

        start_epoch = checkpoint.get("epoch", 0) + 1
        best_metric = checkpoint.get("best_metric", float("inf"))

        logging.info(
            f"Resumed from epoch {start_epoch} with best metric {best_metric:.4f}"
        )

    # Set up tensorboard if requested
    if config["output"].get("tensorboard", False):
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
    else:
        writer = None

    # Set up wandb if requested
    if config["output"].get("use_wandb", False):
        import wandb

        # Ensure log_dir is defined (it should be from earlier setup)
        log_dir = config["output"].get("log_dir", "outputs/logs")
        os.makedirs(log_dir, exist_ok=True)  # Ensure it exists

        wandb.init(
            project=config["output"].get("project_name", "hmer-ink"),
            config=config,
            dir=log_dir,  # Specify the directory for wandb runs
        )

    # Training loop
    logging.info("Starting training")

    no_improvement = 0

    for epoch in range(start_epoch, num_epochs):
        # Train for one epoch
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            use_amp=use_amp,
            grad_accum_steps=grad_accum_steps,
            clip_grad_norm=train_config.get("clip_grad_norm", 0.0),
            benchmark_mode=config.get("mps_options", {}).get("benchmark_mode", False),
        )

        # Validate
        val_metrics = validate(
            model,
            valid_loader,
            criterion,
            tokenizer,
            device,
            beam_size=config["evaluation"].get("beam_size", 2),
            fast_mode=True,  # Use fast mode during training
            max_samples=50,  # Limit to 50 samples for faster validation
        )

        # Log metrics
        logging.info(f"Epoch {epoch}:")
        logging.info(f"  Train Loss: {train_metrics['loss']:.4f}")
        logging.info(f"  Valid Loss: {val_metrics['loss']:.4f}")
        logging.info(
            f"  Expression Recognition Rate: {val_metrics['expression_recognition_rate']:.4f}"
        )
        logging.info(f"  Symbol Accuracy: {val_metrics['symbol_accuracy']:.4f}")
        logging.info(f"  Edit Distance: {val_metrics['edit_distance']:.4f}")

        # Update tensorboard
        if writer is not None:
            writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f"Valid/{key}", value, epoch)

        # Update wandb
        if config["output"].get("use_wandb", False):
            # Collect error examples from validation
            error_examples = []

            # Only proceed if validation was actually performed
            if hasattr(val_metrics, "get") and val_metrics.get("num_samples", 0) > 0:
                        # Try to get a few error examples from the validation run
                if 'all_predictions' in vars() and 'all_targets' in vars() and len(all_predictions) > 0:
                    num_examples = min(5, len(all_predictions))
                    for i in range(num_examples):
                        error_examples.append({
                            "prediction": all_predictions[i],
                            "target": all_targets[i],
                            "cer": val_metrics.get("edit_distance", 0)
                        })

            # Log to wandb
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
            )

            # Also log to our training monitor 
            if config["output"].get("record_metrics", False):
                try:
                    metrics_dir = config["output"].get("metrics_dir", "outputs/training_metrics")
                    os.makedirs(metrics_dir, exist_ok=True)
                    
                    from scripts.training_monitor import capture_training_metrics

                    # Create metrics dict with train and val metrics
                    monitor_metrics = {
                        "epoch": epoch,
                        "train_loss": train_metrics["loss"],
                        "global_step": epoch * len(train_loader),
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                    }

                    # Log to our monitor
                    capture_training_metrics(monitor_metrics, error_examples, metrics_dir)
                    logging.info(f"Recorded metrics to {metrics_dir}")
                except ImportError:
                    logging.info(
                        "Training monitor not found. Metrics only logged to wandb."
                    )

        # Update scheduler if using ReduceLROnPlateau
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        # Save model
        if epoch % save_every == 0 or epoch == num_epochs - 1:
            checkpoint_filename = os.path.join(
                output_dir, f"checkpoint_epoch_{epoch}.pt"
            )
            model.save_checkpoint(
                checkpoint_filename, epoch, optimizer, scheduler, val_metrics["loss"]
            )
            logging.info(f"Saved checkpoint to {checkpoint_filename}")

        # Check for best model
        metric_name = "loss"  # We can change this to ERR or another metric
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
    if config["output"].get("use_wandb", False):
        wandb.finish()

    # Load best model for return
    best_checkpoint = os.path.join(output_dir, "best_model.pt")
    if os.path.exists(best_checkpoint):
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        logging.info(
            f"Loaded best model with {metric_name} = {checkpoint.get('best_metric', -1):.4f}"
        )

    return model


if __name__ == "__main__":
    import typer
    
    def main(
        config: str = typer.Argument(..., help="Path to configuration file"),
        checkpoint: Optional[str] = typer.Option(None, help="Path to checkpoint for resuming training"),
        output_dir: Optional[str] = typer.Option(None, help="Directory to save outputs"),
    ):
        """Train HMER-Ink model."""
        train(config, checkpoint, output_dir)
    
    typer.run(main)
