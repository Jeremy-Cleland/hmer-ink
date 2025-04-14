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
# Basic visualization of an InkML file
make visualize INPUT=data/test/sample.inkml
# or with output
make visualize INPUT=data/test/sample.inkml OUTPUT=visualization.pdf
# Show the visualization instead of saving to file
make visualize INPUT=data/test/sample.inkml SHOW=true

# Visualization of the normalization process
make visualize-normalization INPUT=data/test/sample.inkml OUTPUT=normalization.pdf
# Custom normalization ranges
make visualize-normalization INPUT=data/test/sample.inkml OUTPUT=normalization.pdf --x-min=-2.0 --x-max=2.0

# Visualization of data augmentations
make visualize-augmentations INPUT=data/test/sample.inkml OUTPUT=augmentations.pdf
# With a specific random seed for reproducible results
make visualize-augmentations INPUT=data/test/sample.inkml OUTPUT=augmentations.pdf SEED=42

# AUTOMATIC BATCH VISUALIZATION
# Generate visualizations for 5 random samples from the test set
make visualize-batch
# Customize the batch visualization
make visualize-batch DATA_DIR=data OUTPUT_DIR=outputs/custom_visualizations SPLIT=train NUM_SAMPLES=10 SEED=42
# Choose which types of visualizations to generate
make visualize-batch NORMALIZATION=false AUGMENTATION=false  # Only basic visualizations
```

These visualization tools help in understanding:
1. How the original ink strokes look
2. How normalization affects the data (original → normalized → relative coordinates)
3. How different augmentations transform the data, with adaptive behavior based on expression complexity

The `visualize-batch` command is particularly useful for exploring the dataset without needing to specify individual file paths. It automatically selects random samples and generates all visualization types.

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

# Data processing settings
data:
  normalization:
    x_range: [-1, 1]  # Target range for x coordinates
    y_range: [-1, 1]  # Target range for y coordinates
    # Note: Aspect ratio is preserved by default to prevent distortion
  
  augmentation:
    enabled: true
    scale_range: [0.9, 1.1]       # Gentle scaling to avoid distortion
    rotation_range: [-10, 10]     # Rotation angles in degrees
    rotation_probability: 0.7     # Probability of applying rotation
    translation_range: [-0.05, 0.05]  # Small translations
    stroke_dropout_prob: 0.03     # Low probability of stroke dropout
    max_dropout_ratio: 0.2        # Maximum ratio of strokes that can be dropped
    jitter_scale: 0.005           # Minimal jitter to retain legibility
    jitter_probability: 0.7       # Probability of applying jitter to each point
    # Note: All augmentations are adaptive based on expression complexity
  
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

## How It Works: From Ink Strokes to LaTeX

The model processes handwritten mathematical expressions through several stages:

1. **Data Preprocessing**:
   - **Parsing**: InkML files containing stroke data (x, y, t coordinates) are parsed
   - **Normalization**: Coordinates are normalized to a consistent range ([-1, 1]) with aspect ratio preservation
   - **Relative Encoding**: Absolute coordinates are converted to relative movements (dx, dy)
   - **Stroke Flattening**: All strokes are combined into a single sequence with pen-up indicators

2. **Encoder Processing**:
   - The preprocessed ink sequence is embedded into a higher-dimensional space
   - The encoder (Transformer/BiLSTM/CNN) processes this sequence
   - Produces a context-aware representation of the handwritten expression

3. **Decoder Generation**:
   - Starts with a special start token
   - At each step, attends to the encoded representation
   - Predicts the next LaTeX token based on previously generated tokens
   - Uses beam search to explore multiple candidate sequences

4. **Post-processing**:
   - Converts token IDs back to LaTeX symbols
   - Assembles the complete LaTeX expression

This approach effectively "translates" from the language of handwritten strokes to the language of LaTeX, similar to how machine translation works between natural languages.

## Performance Optimization

- Uses Apple MPS for training on M-series Macs
- Automatic Mixed Precision (AMP) for faster training
- Beam search decoding for better prediction quality
- Batched processing and data caching

## Future Development

### Adaptation to Image-Based Recognition

The current model is designed for handwritten ink strokes (online recognition), but the architecture can be adapted for image-based math recognition (offline recognition):

1. **Input Layer Adaptation**:
   - Replace the stroke encoder (currently uses x, y, t, pen_state) with a vision encoder (CNN or Vision Transformer)
   - Keep the transformer decoder architecture unchanged
   - Add a connection layer between image features and the transformer decoder

2. **Implementation Path**:
   - Modify `encoder.py` to include a new `ImageEncoder` class 
   - Update the model configuration to support image inputs
   - Create an image dataset class alongside the existing ink dataset

3. **Benefits of Current Design**:
   - The modular architecture separates encoder/decoder components
   - The decoder logic for generating LaTeX is independent of input type
   - Training on stroke data creates a strong foundation for LaTeX generation

This transition would enable the model to work with scanned mathematical expressions and digital images of handwritten math.

### Other Potential Extensions

- **Multi-modal input**: Support for both stroke and image inputs in a single model
- **Pre-training strategies**: Self-supervised pre-training on large unlabeled datasets
- **Language-specific adaptations**: Support for other mathematical notation systems
- **Transfer learning**: Fine-tuning on domain-specific mathematical expressions

## License

See the LICENSE file for details.
