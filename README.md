# HMER-Ink: Handwritten Mathematical Expression Recognition

HMER-Ink is a model for recognizing handwritten mathematical expressions from digital ink data.

## Dataset

This project uses the MathWriting dataset, which contains online handwritten mathematical expressions stored in InkML format. The dataset includes:

- `train`: 229,864 human-written samples
- `valid`: 15,674 validation samples
- `test`: 7,644 test samples
- `symbols`: 6,423 individual symbol samples
- `synthetic`: 396,014 synthetic samples (not used in default configuration)

## Setup

```bash
# Install dependencies
conda env create -f environment.yml
conda activate hmer-ink
```

## Training

The default configuration uses the `train` and `symbols` directories for training without synthetic data.

```bash
# One-click training using the provided scripts
./train.sh [experiment_name]  # Standard training
./train_curriculum.sh [experiment_name]  # Curriculum learning (easy to hard)

# Train the model with specific configurations
python cli.py train --config configs/default.yaml
python cli.py train --config configs/curriculum.yaml  # With curriculum learning

# Resume training from a checkpoint
python cli.py train --config configs/default.yaml --checkpoint path/to/checkpoint.pt
```

The default training configuration:
- Uses transformer-based encoder-decoder architecture
- Trains with AdamW optimizer and OneCycleLR scheduler
- Applies data augmentation (rotation, scaling, jitter)
- Takes approximately 25-30 minutes per epoch
- Supports Mac's MPS acceleration

### Curriculum Learning

The project supports curriculum learning, which trains the model progressively from easy to hard examples:

- **How it works**: The model starts with simpler examples and gradually introduces more complex ones as training progresses
- **Difficulty metrics**: 
  - `token_length`: Sorts examples by the number of tokens in the LaTeX expression (default)
  - `seq_length`: Sorts examples by the length of the input sequence
- **Configuration**: Set in `configs/curriculum.yaml` with parameters:
  - `enabled`: Whether to use curriculum learning
  - `metric`: Which difficulty metric to use
  - `epochs_to_full_difficulty`: Number of epochs until all examples are included

To use curriculum learning:
```bash
./train_curriculum.sh [experiment_name]
```

## Evaluation

```bash
# Evaluate a trained model
python cli.py evaluate --model path/to/model.pt --split test

# Make a prediction for a single file
python cli.py predict --model path/to/model.pt --input path/to/ink.inkml
```

## Configuration

Two main configuration files are available:

### Default Configuration (`configs/default.yaml`)

- **Synthetic Data**: Disabled by default to keep training time reasonable
- **Bounding Box Data**: Enabled in the model but only used when synthetic data is available
- **Training Settings**: Optimized batch size, learning rate, and gradient accumulation

### Curriculum Learning Configuration (`configs/curriculum.yaml`)

- **Progressive Training**: Starts with easy examples and gradually introduces harder ones
- **Difficulty Metrics**: Can use token length or sequence length to determine difficulty
- **Transition Period**: Takes 15 epochs to transition from easiest to full dataset
- **Otherwise Identical**: Uses the same model architecture and hyperparameters as the default config

## Visualization

```bash
# Visualize an ink sample
python cli.py visualize --input path/to/ink.inkml --output visualization.pdf

# Visualize data augmentation effects
python cli.py visualize-augmentations --input path/to/ink.inkml
```

## Monitoring

Training metrics are recorded and can be monitored using:

```bash
# Watch training progress
make watch-training METRICS_FILE=path/to/metrics.json

# View dashboard
make dashboard METRICS_DIR=path/to/metrics/dir
```