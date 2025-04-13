# HMER-Ink: Handwritten Mathematical Expression Recognition

A PyTorch implementation for recognizing handwritten mathematical expressions, using the MathWriting dataset.

## Overview

This project provides a deep learning solution for Handwritten Mathematical Expression Recognition (HMER), converting handwritten ink strokes into LaTeX code. It uses an encoder-decoder architecture to process online handwriting data in the InkML format.

## Dataset

The model is designed to work with the MathWriting dataset, which contains:
- 229,864 human-written mathematical expressions (train)
- 15,674 validation samples (valid)
- 7,644 test samples (test)
- 6,423 individual handwritten symbols (symbols)
- 396,014 synthetic samples (synthetic)

## Features

- **Modular Architecture**: Flexible encoder-decoder models including Transformer, LSTM, and CNN options
- **Apple Silicon Support**: Optimized for MPS on Apple Silicon Macs
- **Data Augmentation**: Comprehensive augmentation pipeline for handwritten data
- **Command Line Interface**: Simple CLI using Typer for training, evaluation, and visualization
- **Logging and Visualization**: Support for TensorBoard and Weights & Biases integration

## Requirements

- Python 3.8+
- PyTorch 2.0+
- See `environment.yml` for all dependencies

## Installation

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate hmer-ink
```

## Usage

### Training

```bash
# Train with default configuration
python cli.py train --config configs/default.yaml

# Resume training from checkpoint
python cli.py train --config configs/default.yaml --checkpoint outputs/checkpoints/checkpoint_epoch_10.pt
```

### Evaluation

```bash
# Evaluate on test set
python cli.py evaluate --model outputs/checkpoints/best_model.pt --config configs/default.yaml

# Save evaluation results
python cli.py evaluate --model outputs/checkpoints/best_model.pt --output results/evaluation.json
```

### Prediction

```bash
# Make prediction for a single InkML file
python cli.py predict --model outputs/checkpoints/best_model.pt --input data/test/sample.inkml

# Visualize prediction
python cli.py predict --model outputs/checkpoints/best_model.pt --input data/test/sample.inkml --visualize
```

### Visualization

```bash
# Visualize an InkML file
python cli.py visualize --input data/test/sample.inkml --output visualization.pdf
```

## Configuration

The model is controlled by a YAML configuration file. See `configs/default.yaml` for a complete example. Key configuration sections:

- **Data Settings**: Control data loading, normalization, and augmentation
- **Model Architecture**: Configure encoder and decoder settings
- **Training Parameters**: Set batch size, learning rate, etc.
- **Evaluation Settings**: Configure metrics and evaluation process
- **Output Settings**: Control checkpoints and logging

## Project Structure

```
hmer-ink/
├── hmer/                   # Main package
│   ├── data/               # Data processing modules
│   ├── models/             # Model architectures
│   ├── utils/              # Utility functions
│   └── config.py           # Configuration handling
├── scripts/                # CLI scripts
├── configs/                # Configuration files
├── cli.py                  # Typer CLI entry point
└── environment.yml         # Dependencies
```

## Model Architecture

The system uses an encoder-decoder architecture:

1. **Encoder**: Processes ink strokes (sequences of (x, y, t) coordinates)
   - Options: Transformer, BiLSTM, or CNN

2. **Decoder**: Generates LaTeX code from encoded representations
   - Options: Transformer or LSTM with attention

## Performance Optimization

- Uses Apple MPS for training on M-series Macs
- Automatic Mixed Precision (AMP) for faster training
- Beam search decoding for better prediction quality
- Batched processing and data caching

## License

See the LICENSE file for details.