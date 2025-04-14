# Resuming Training from a Checkpoint

This guide explains how to resume training from a specific checkpoint and how to manage model checkpoints during training.

## Resuming Training from a Specific Checkpoint

To resume training from a checkpoint (e.g., after epoch 6), you can use the following command:

```bash
python cli.py train --config configs/fast_enhanced.yaml --checkpoint /Users/jeremy/hmer-ink/outputs/checkpoints/checkpoint_epoch_6.pt --output-dir outputs/resumed_training
```

Or using the Makefile:

```bash
make train CONFIG=configs/fast_enhanced.yaml CHECKPOINT=/Users/jeremy/hmer-ink/outputs/checkpoints/checkpoint_epoch_6.pt EXPERIMENT=resumed_training
```

This will:
1. Load the model state from `checkpoint_epoch_6.pt`
2. Begin training from epoch 7 (the next epoch after 6)
3. Use the settings from `fast_enhanced.yaml`
4. Save outputs to a new directory `outputs/resumed_training`

## Understanding Where Checkpoints Are Saved

By default, model checkpoints are saved in the following locations:

1. **Default Location**: `outputs/checkpoints/`
   - When not specifying an output directory, checkpoints will go here
   - This can lead to confusion as new training runs will overwrite previous checkpoints

2. **Experiment-specific Location**: `outputs/{experiment_name}/checkpoints/`
   - When using `--output-dir` or the `EXPERIMENT` parameter with `make train`
   - Each experiment gets its own directory to keep checkpoints separate

## Best Practices for Managing Checkpoints

1. **Always specify an experiment name or output directory**:
   ```bash
   make train EXPERIMENT=my_experiment CONFIG=configs/fast_enhanced.yaml
   ```

2. **When resuming training, create a new output directory**:
   ```bash
   make train EXPERIMENT=resumed_from_epoch6 CONFIG=configs/fast_enhanced.yaml CHECKPOINT=/Users/jeremy/hmer-ink/outputs/checkpoints/checkpoint_epoch_6.pt
   ```

3. **Label your experiments descriptively**:
   - Use naming that indicates the configuration used
   - Include version numbers for experiments with similar configs

## Checkpoint Naming Convention

Checkpoints are named with the following pattern:

- **Epoch checkpoints**: `checkpoint_epoch_{epoch_number}.pt`
- **Best model**: `best_model.pt` - The model with the best performance on the validation set

## Checking Training Progress

To monitor training progress, you can use:

```bash
make watch-training
```

Or for a more interactive dashboard:

```bash
make dashboard
```

These tools will show metrics like loss, accuracy, and learning rate over time.