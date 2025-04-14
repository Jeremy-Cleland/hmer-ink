"""
Model expansion script for HMER-Ink.

This script implements progressive model expansion, allowing a smaller trained model
to be expanded to a larger architecture while preserving learned weights.
"""

import os
import sys
import typer
from typing import Optional

import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hmer.config import load_config
from hmer.models import HMERModel, get_encoder, get_decoder


def expand_model(
    src_checkpoint: str,
    dst_checkpoint: str,
    config_path: str,
    device: Optional[str] = None,
):
    """
    Expand a model from a smaller architecture to a larger one.

    Args:
        src_checkpoint: Path to source checkpoint (smaller model)
        dst_checkpoint: Path to save the expanded model
        config_path: Path to configuration file for the expanded model
        device: Device to use for loading model
    """
    print(f"Expanding model: {src_checkpoint} → {dst_checkpoint}")
    print(f"Using configuration: {config_path}")

    # Load configuration
    config = load_config(config_path)
    print("Loaded expanded model configuration")

    # Set device
    if device is None:
        device = config["training"].get("device", "cpu")
    print(f"Using device: {device}")

    # Load the smaller model
    print("Loading source model...")
    small_model, checkpoint = HMERModel.load_checkpoint(
        src_checkpoint, map_location=device
    )
    print("Source model loaded successfully")

    # Extract vocabulary size
    vocab_size = small_model.decoder.output_projection.weight.shape[0]
    print(f"Vocabulary size: {vocab_size}")

    # Create the larger model with new configuration
    print("Creating expanded model...")
    # Pass only the configuration needed by get_encoder and get_decoder
    encoder = get_encoder({"encoder": config["model"]["encoder"]})
    decoder = get_decoder({"decoder": config["model"]["decoder"]}, vocab_size)
    large_model = HMERModel(encoder, decoder, config)
    print("Expanded model created successfully")

    # Print model sizes for comparison
    small_params = sum(p.numel() for p in small_model.parameters())
    large_params = sum(p.numel() for p in large_model.parameters())
    print(f"Small model parameters: {small_params:,}")
    print(f"Large model parameters: {large_params:,}")
    print(
        f"Parameter increase: {large_params - small_params:,} (+{(large_params / small_params - 1) * 100:.1f}%)"
    )

    # Copy matching weights from small to large model
    print("Transferring weights from small to large model...")
    large_state_dict = large_model.state_dict()
    small_state_dict = small_model.state_dict()

    # Keep track of transferred parameters
    transferred = 0
    total_params = len(small_state_dict)

    # Dictionary to store shape mismatches for reporting
    shape_mismatches = {}

    for name, param in small_state_dict.items():
        if name in large_state_dict:
            if large_state_dict[name].shape == param.shape:
                # Direct transfer for matching shapes
                large_state_dict[name] = param
                transferred += 1
            else:
                # Store shape mismatch for later reporting
                shape_mismatches[name] = (param.shape, large_state_dict[name].shape)

                # For embedding layers, we can partially copy
                if "embedding" in name or "output_projection" in name:
                    if len(param.shape) == 2 and len(large_state_dict[name].shape) == 2:
                        # Copy weights for dimensions that match
                        min_dim0 = min(param.shape[0], large_state_dict[name].shape[0])
                        min_dim1 = min(param.shape[1], large_state_dict[name].shape[1])
                        large_state_dict[name][:min_dim0, :min_dim1] = param[
                            :min_dim0, :min_dim1
                        ]
                        transferred += 0.5  # Count as partial transfer

    # Print transfer statistics
    print(f"Transferred {transferred}/{total_params} parameter tensors")
    if shape_mismatches:
        print(f"Shape mismatches ({len(shape_mismatches)} tensors):")
        for name, (small_shape, large_shape) in shape_mismatches.items():
            print(f"  - {name}: {small_shape} → {large_shape}")

    # Load the modified state dict into the large model
    large_model.load_state_dict(large_state_dict, strict=False)
    print("Weight transfer complete")

    # Save the expanded model
    print(f"Saving expanded model to {dst_checkpoint}")
    # Only create directories if there's actually a path to create
    dst_dir = os.path.dirname(dst_checkpoint)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)

    # Preserve training state from the original checkpoint
    save_dict = {
        "model": large_model.state_dict(),
        "config": config,
        "epoch": checkpoint.get("epoch", 0),
    }

    # If optimizer and scheduler states exist, note that we're not transferring them
    if "optimizer" in checkpoint:
        print(
            "Note: Optimizer state from source model cannot be transferred due to parameter count differences"
        )
    if "scheduler" in checkpoint:
        print("Note: Scheduler state from source model cannot be transferred")

    # Save best metric if available
    if "best_metric" in checkpoint:
        save_dict["best_metric"] = checkpoint["best_metric"]
        print(f"Preserved best metric: {checkpoint['best_metric']}")

    # Save the checkpoint
    torch.save(save_dict, dst_checkpoint)
    print("Expanded model saved successfully")

    return large_model


def main(
    src_checkpoint: str = typer.Argument(
        ..., help="Path to source checkpoint (smaller model)"
    ),
    dst_checkpoint: str = typer.Argument(..., help="Path to save the expanded model"),
    config: str = typer.Option(
        "configs/fast_expanded.yaml",
        "--config",
        "-c",
        help="Configuration for expanded model",
    ),
    device: str = typer.Option(
        "", "--device", "-d", help="Device to use (default: from config)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Only show model comparison without saving"
    ),
):
    """Expand a model from a smaller architecture to a larger one, preserving learned weights."""
    if dry_run:
        # Load models but don't save
        print("Dry run mode: Only showing model comparison")
        small_model, _ = HMERModel.load_checkpoint(src_checkpoint)

        # Load configuration
        expanded_config = load_config(config)

        # Get vocabulary size from smaller model
        vocab_size = small_model.decoder.output_projection.weight.shape[0]

        # Create expanded model
        encoder = get_encoder({"encoder": expanded_config["model"]["encoder"]})
        decoder = get_decoder(
            {"decoder": expanded_config["model"]["decoder"]}, vocab_size
        )
        large_model = HMERModel(encoder, decoder, expanded_config)

        # Print comparison
        small_params = sum(p.numel() for p in small_model.parameters())
        large_params = sum(p.numel() for p in large_model.parameters())
        print(f"Small model parameters: {small_params:,}")
        print(f"Large model parameters: {large_params:,}")
        print(
            f"Parameter increase: {large_params - small_params:,} (+{(large_params / small_params - 1) * 100:.1f}%)"
        )
    else:
        # Expand and save the model
        expand_model(src_checkpoint, dst_checkpoint, config, device if device else None)
        print("Model expansion completed successfully!")


if __name__ == "__main__":
    typer.run(main)
