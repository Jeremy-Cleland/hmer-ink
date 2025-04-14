#!/bin/bash
# Script to train the expanded model after model expansion

# Path to the expanded model checkpoint
CHECKPOINT="/Users/jeremy/hmer-ink/outputs/checkpoints/expanded_model.pt"

# Path to the expanded model configuration
CONFIG="configs/fast_expanded.yaml"

# Set experiment name
EXPERIMENT="expanded_model_finetuning"

# Continue training with the expanded model
echo "Training expanded model..."
python cli.py train --config "$CONFIG" --checkpoint "$CHECKPOINT" --experiment "$EXPERIMENT"