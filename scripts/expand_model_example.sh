#!/bin/bash
# Example script to expand a model from a smaller to larger architecture

# Path to the source model checkpoint (smaller model)
SRC_CHECKPOINT="/Users/jeremy/hmer-ink/outputs/checkpoints/checkpoint_epoch_6.pt"

# Path for the expanded model checkpoint
DST_CHECKPOINT="/Users/jeremy/hmer-ink/outputs/checkpoints/expanded_model.pt"

# Path to the expanded model configuration
CONFIG="configs/fast_expanded.yaml"

# First do a dry-run to preview the model expansion
echo "Previewing model expansion..."
python scripts/expand_model.py "$SRC_CHECKPOINT" "$DST_CHECKPOINT" --config "$CONFIG" --dry-run

# Ask for confirmation
read -p "Do you want to proceed with the model expansion? (y/n) " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "Model expansion cancelled"
    exit 0
fi

# Expand the model
echo "Expanding model..."
python scripts/expand_model.py "$SRC_CHECKPOINT" "$DST_CHECKPOINT" --config "$CONFIG"

echo "Model expansion completed!"
echo "You can continue training with:"
echo "python cli.py train --config $CONFIG --checkpoint $DST_CHECKPOINT"
echo "or use the Makefile target:"
echo "make train-expanded"