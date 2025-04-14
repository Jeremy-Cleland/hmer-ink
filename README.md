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
- **Training Monitoring**: Multi-view training progress tracking with interactive dashboard
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

### Optional Dependencies for Monitoring

For the interactive dashboard and advanced visualizations:

```bash
# Install Streamlit for the interactive dashboard
pip install streamlit

# Install additional visualization dependencies
pip install altair>=5.0.0
```

## Usage

### Training

```bash
# Train with default configuration
make train
# or with specific config
make train CONFIG=configs/default.yaml

# Fast training with optimized settings for Apple Silicon
make train-fast

# Resume training from checkpoint
python cli.py train --config configs/default.yaml --checkpoint outputs/checkpoints/checkpoint_epoch_10.pt
```

### Stopping and Resuming Training

Training can be paused and resumed at any time:

1. **To stop training**: Press `Ctrl+C` in the terminal to stop training gracefully
2. **To resume training**: Use the checkpoint path from the last saved model

   ```bash
   python cli.py train --config configs/your_config.yaml --checkpoint outputs/experiment_name/checkpoints/checkpoint_epoch_N.pt
   ```

### Training Monitoring

HMER-Ink provides three ways to monitor training progress:

1. **Real-time static visualizations**:

   ```bash
   # Monitor training with auto-updating plots
   make watch-training
   ```

2. **Interactive dashboard** (requires streamlit):

   ```bash
   # Launch interactive web dashboard to monitor metrics
   make dashboard
   ```

3. **Extract metrics from Weights & Biases**:

   ```bash
   # If you're using wandb logging
   make visualize-training WANDB_DIR=wandb
   ```

The monitoring system automatically tracks:

- Training and validation loss
- Accuracy, expression recognition rate (exprate)
- Error rates (character and token level)
- Example predictions

**Where to find monitoring data**:

- Training summary: `outputs/training_metrics/latest_summary.md`
- Training plots: `outputs/training_metrics/plots/`
- Dashboard: <http://localhost:8501> (when using `make dashboard`)

### Training on Multiple Machines

To run training on multiple machines (without shared storage):

1. **On each machine**:
   - Clone the repository and set up the environment
   - Copy your dataset to each machine
   - Create a machine-specific config file based on the machine's specs:

     ```bash
     cp configs/fast.yaml configs/fast_machine2.yaml
     # Edit configs/fast_machine2.yaml to adjust batch_size, num_workers, etc.
     ```

2. **Start training independently on each machine**:

   ```bash
   # On machine 1
   make train-fast CONFIG=configs/fast.yaml
   
   # On machine 2 
   make train-fast CONFIG=configs/fast_machine2.yaml
   ```

3. **Monitor training on each machine separately**:

   ```bash
   make dashboard
   # or
   make watch-training
   ```

4. **Compare results** by reviewing trained models and metrics on each machine

### Evaluation

```bash
# Evaluate on test set
make evaluate MODEL=outputs/experiment_name/checkpoints/best_model.pt
# or
python cli.py evaluate --model outputs/checkpoints/best_model.pt --config configs/default.yaml

# Save evaluation results
python cli.py evaluate --model outputs/checkpoints/best_model.pt --output results/evaluation.json
```

### Prediction

```bash
# Make prediction for a single InkML file
make predict MODEL=outputs/checkpoints/best_model.pt INPUT=data/test/sample.inkml
# or
python cli.py predict --model outputs/checkpoints/best_model.pt --input data/test/sample.inkml

# Visualize prediction
make predict MODEL=outputs/checkpoints/best_model.pt INPUT=data/test/sample.inkml VISUALIZE=true
```

### Visualization

```bash
# Visualize an InkML file
make visualize INPUT=data/test/sample.inkml
# or with output
make visualize INPUT=data/test/sample.inkml OUTPUT=visualization.pdf
```

### Generating Reports

```bash
# Generate a comprehensive report for a trained model
make report MODEL=outputs/experiment_name/checkpoints/best_model.pt
```

## Configuration

The model is controlled by a YAML configuration file. See `configs/default.yaml` for a complete example. Key configuration sections:

- **Data Settings**: Control data loading, normalization, and augmentation
- **Model Architecture**: Configure encoder and decoder settings
- **Training Parameters**: Set batch size, learning rate, etc.
- **Evaluation Settings**: Configure metrics and evaluation process
- **Output Settings**: Control checkpoints and logging

### Key Configuration Parameters for Different Machines

When training on multiple machines, you may need to adjust these parameters:

```yaml
# Compute settings - adjust based on your machine's capabilities
training:
  batch_size: 64  # Reduce for machines with less RAM
  device: "mps"   # Use "cuda" for NVIDIA GPUs, "cpu" for no GPU
  num_workers: 4  # Adjust based on CPU cores available
  gradient_accumulation_steps: 8  # Increase for smaller batch sizes
  use_amp: true   # Mixed precision - keep enabled for faster training
  
# Model size - adjust for memory constraints
model:
  encoder:
    embedding_dim: 256  # Reduce for machines with less memory
    num_layers: 3       # Fewer layers = less memory usage
    
# MPS specific settings - for Apple Silicon
mps_configuration:
  enable_mps_fallback: true
  # other MPS settings...
```

The `configs/fast.yaml` provides a good starting point with reduced model size and efficient training settings for Apple Silicon.

## Project Structure

```
hmer-ink/
├── hmer/                   # Main package
│   ├── data/               # Data processing modules
│   ├── models/             # Model architectures
│   ├── utils/              # Utility functions
│   └── config.py           # Configuration handling
├── scripts/                # CLI scripts
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   ├── training_monitor.py # Training monitoring tools
│   ├── dashboard.py        # Interactive dashboard (auto-generated)
│   └── generate_report.py  # Model report generation
├── configs/                # Configuration files
│   ├── default.yaml        # Default configuration
│   └── fast.yaml           # Configuration optimized for fast training
├── cli.py                  # Typer CLI entry point
├── Makefile                # Convenience commands
├── outputs/                # Training outputs directory
│   ├── training_metrics/   # Training metrics and visualizations
│   └── checkpoints/        # Model checkpoints
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
