"""
Evaluation script for HMER-Ink models.
"""

import json
import logging
import os
import sys
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hmer.config import get_device, load_config
from hmer.data.dataset import HMERDataset
from hmer.data.transforms import get_eval_transforms
from hmer.models import HMERModel
from hmer.utils.metrics import compute_metrics
from hmer.utils.tokenizer import LaTeXTokenizer


def evaluate(
    model_path: str,
    config_path: str,
    output_path: Optional[str] = None,
    split: str = "test",
    beam_size: int = 4,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Dict[str, float]:
    """
    Evaluate a trained HMER model.

    Args:
        model_path: Path to model checkpoint
        config_path: Path to configuration file
        output_path: Path to save evaluation results
        split: Data split to evaluate on ('test', 'valid', etc.)
        beam_size: Beam size for generation
        batch_size: Batch size for evaluation
        num_workers: Number of workers for data loading

    Returns:
        Dictionary with evaluation metrics
    """
    # Load configuration
    config = load_config(config_path)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Get device
    device = get_device(config["training"])
    logging.info(f"Using device: {device}")

    # Load model
    logging.info(f"Loading model from {model_path}")
    model, checkpoint = HMERModel.load_checkpoint(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer_path = os.path.join(os.path.dirname(model_path), "vocab.json")
    if not os.path.exists(tokenizer_path):
        tokenizer_path = os.path.join(
            config["output"].get("checkpoint_dir", "outputs/checkpoints"), "vocab.json"
        )

    if not os.path.exists(tokenizer_path):
        raise ValueError(f"Tokenizer not found at {tokenizer_path}")

    tokenizer = LaTeXTokenizer()
    tokenizer.load_vocab(tokenizer_path)
    logging.info(f"Loaded tokenizer with vocabulary size {len(tokenizer)}")

    # Set up data
    data_dir = config["data"].get("data_dir", "data")

    # Get normalization parameters
    normalize = True
    data_config = config["data"]
    x_range = data_config.get("normalization", {}).get("x_range", (-1, 1))
    y_range = data_config.get("normalization", {}).get("y_range", (-1, 1))
    time_range = data_config.get("normalization", {}).get("time_range", (0, 1))

    # Create dataset
    dataset = HMERDataset(
        data_dir=data_dir,
        split_dirs=[split],
        tokenizer=tokenizer,
        max_seq_length=config["data"].get("max_seq_length", 512),
        max_token_length=config["data"].get("max_token_length", 128),
        transform=get_eval_transforms(config["data"]),
        normalize=normalize,
        use_relative_coords=True,
        x_range=x_range,
        y_range=y_range,
        time_range=time_range,
    )

    logging.info(f"Created dataset with {len(dataset)} samples")

    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=dataset.collate_fn,
    )

    # Run evaluation
    all_predictions = []
    all_targets = []
    all_file_ids = []

    logging.info("Starting evaluation")

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Get batch data
            input_seq = batch["input"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            labels = batch["labels"]
            file_ids = batch["file_ids"]

            # Limit batch size for efficient processing on MPS/Apple Silicon
            if input_seq.size(0) > 16 and device.type == "mps":
                # Use the first 16 samples of the batch
                gen_indices = list(range(16))
                gen_input_seq = input_seq[gen_indices]
                gen_input_lengths = input_lengths[gen_indices]
                gen_labels = [labels[i] for i in gen_indices]
                gen_file_ids = [file_ids[i] for i in gen_indices]
            else:
                gen_input_seq = input_seq
                gen_input_lengths = input_lengths
                gen_labels = labels
                gen_file_ids = file_ids

            # Generate predictions with beam search
            beam_results, beam_scores = model.generate(
                gen_input_seq,
                gen_input_lengths,
                max_length=128,
                beam_size=beam_size,
                fast_mode=False,  # Use full beam search for final evaluation
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
            all_file_ids.extend(gen_file_ids)

    # Calculate metrics
    metrics = compute_metrics(all_predictions, all_targets)

    # Log metrics
    logging.info("Evaluation results:")
    for key, value in metrics.items():
        logging.info(f"  {key}: {value:.4f}")

    # Save detailed results if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        detailed_results = []
        for file_id, pred, target in zip(all_file_ids, all_predictions, all_targets):
            # Calculate sample-level metrics
            edit_distance = compute_metrics(
                [pred], [target], metrics=["edit_distance"]
            )["edit_distance"]
            exact_match = 1 if pred == target else 0

            detailed_results.append(
                {
                    "file_id": file_id,
                    "prediction": pred,
                    "target": target,
                    "edit_distance": edit_distance,
                    "exact_match": exact_match,
                }
            )

        with open(output_path, "w") as f:
            json.dump({"metrics": metrics, "results": detailed_results}, f, indent=2)

        logging.info(f"Saved detailed results to {output_path}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate HMER-Ink model")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument("--output", type=str, help="Path to save evaluation results")
    parser.add_argument(
        "--split", type=str, default="test", help="Data split to evaluate on"
    )
    parser.add_argument(
        "--beam_size", type=int, default=4, help="Beam size for generation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation"
    )

    args = parser.parse_args()

    evaluate(
        args.model,
        args.config,
        args.output,
        args.split,
        args.beam_size,
        args.batch_size,
    )
