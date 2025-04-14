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

## Handling Vocabulary Size Mismatches

If you encounter a vocabulary size mismatch error (e.g., when the checkpoint model was trained with a different vocabulary size), use the `expand_model.py` script to adapt the checkpoint:

```bash
python scripts/expand_model.py /path/to/original_checkpoint.pt /path/to/adapted_model.pt --config configs/fast_enhanced.yaml
```

Then resume training with the adapted model:

```bash
python cli.py train --config configs/fast_enhanced.yaml --checkpoint /path/to/adapted_model.pt --output-dir outputs/resumed_training
```

### Preventing Vocabulary Mismatches

To prevent vocabulary mismatches in the future, you can:

1. **Use a fixed random seed**: The vocabulary generation uses a fixed random seed (default: 42) that can be set in the config:
   ```yaml
   data:
     vocab_random_seed: 42
   ```

2. **Enable shared vocabulary**: To use a consistent vocabulary across training runs, enable shared vocabulary in your config:
   ```yaml
   data:
     vocab_file: "vocab.json"
     use_shared_vocab: true
   ```

   This will save/load the vocabulary file from the data directory rather than the experiment-specific directory.

3. **Save vocabulary explicitly**: After successful training, copy the vocabulary file to a known location:
   ```bash
   cp outputs/your_experiment/vocab.json data/vocab.json
   ```

### Ensuring Consistent Loss Values Between Runs

To ensure consistent loss values between training runs, set a fixed seed for the DataLoader:

```yaml
data:
  dataloader_seed: 42
```

This ensures that batches are created in a consistent order between runs, which helps:
1. Stabilize initial loss values
2. Make training progress more comparable between experiments 
3. Reduce randomness in convergence behavior

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