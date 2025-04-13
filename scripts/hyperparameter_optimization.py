"""
Hyperparameter optimization script using Weights & Biases Sweep.
"""

import argparse
import os
import sys

import yaml

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import wandb

from hmer.config import load_config


def run_sweep(config_path: str, experiment_name: str, num_runs: int = 10):
    """
    Run hyperparameter optimization sweep using Weights & Biases.

    Args:
        config_path: Path to base configuration file
        experiment_name: Name for the sweep experiment
        num_runs: Number of runs to perform
    """
    # Load base configuration
    base_config = load_config(config_path)

    # Define sweep configuration
    sweep_config = {
        "method": "bayes",  # Bayesian optimization
        "metric": {"name": "val_expression_recognition_rate", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"min": 0.00005, "max": 0.001},
            "batch_size": {"values": [32, 64, 96, 128]},
            "encoder.num_layers": {"values": [4, 6, 8]},
            "encoder.embedding_dim": {"values": [256, 384, 512]},
            "decoder.num_layers": {"values": [4, 6, 8]},
            "encoder.dropout": {"min": 0.1, "max": 0.3},
            "decoder.dropout": {"min": 0.1, "max": 0.3},
            "augmentation.scale_range": {
                "values": [[0.8, 1.2], [0.7, 1.3], [0.6, 1.4]]
            },
            "augmentation.rotation_range": {
                "values": [[-10, 10], [-15, 15], [-20, 20]]
            },
        },
    }

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=experiment_name)

    # Define training function
    def train_model():
        # Initialize wandb run
        run = wandb.init()

        # Get hyperparameters from wandb
        params = wandb.config

        # Update configuration with sweep parameters
        updated_config = base_config.copy()

        # Apply hyperparameters from sweep
        updated_config["training"]["learning_rate"] = params.learning_rate
        updated_config["training"]["batch_size"] = params.batch_size
        updated_config["model"]["encoder"]["num_layers"] = params.encoder.num_layers
        updated_config["model"]["encoder"]["embedding_dim"] = (
            params.encoder.embedding_dim
        )
        updated_config["model"]["decoder"]["num_layers"] = params.decoder.num_layers
        updated_config["model"]["encoder"]["dropout"] = params.encoder.dropout
        updated_config["model"]["decoder"]["dropout"] = params.decoder.dropout
        updated_config["data"]["augmentation"]["scale_range"] = (
            params.augmentation.scale_range
        )
        updated_config["data"]["augmentation"]["rotation_range"] = (
            params.augmentation.rotation_range
        )

        # Make sure model decoder embedding_dim matches encoder embedding_dim
        updated_config["model"]["decoder"]["embedding_dim"] = updated_config["model"][
            "encoder"
        ]["embedding_dim"]

        # Save updated config
        os.makedirs("outputs/sweep_configs", exist_ok=True)
        config_filename = f"outputs/sweep_configs/sweep_{run.id}.yaml"
        with open(config_filename, "w") as f:
            yaml.dump(updated_config, f)

        # Import training function here to avoid loading modules before setting up wandb
        from scripts.train import train

        # Run training
        output_dir = f"outputs/sweep_{run.id}"
        os.makedirs(output_dir, exist_ok=True)

        # Train model with the updated config
        train(config_filename, output_dir=output_dir)

    # Run the sweep
    wandb.agent(sweep_id, train_model, count=num_runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization using W&B Sweeps"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to base configuration file",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="hpo-hmer-ink",
        help="Name for the sweep experiment",
    )
    parser.add_argument(
        "--num-runs", type=int, default=10, help="Number of runs to perform"
    )

    args = parser.parse_args()

    run_sweep(args.config, args.experiment, args.num_runs)
