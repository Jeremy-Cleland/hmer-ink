# HMER-Ink Lambda Labs Deployment Guide

This guide provides step-by-step instructions for running HMER-Ink training on Lambda Labs GPU instances.

## Step 1: Set Up Lambda Labs Account

1. Sign up for a Lambda Labs account at https://lambdalabs.com/ if you don't have one
2. Add payment information to your account
3. Verify your email address and complete account setup

## Step 2: Launch a GPU Instance on Lambda Labs

1. Log in to your Lambda Labs account
2. Navigate to "Instances" in the dashboard
3. Click "Launch Instance"
4. Select an appropriate instance type:
   - Recommended: L4 or A10 GPU instances for good performance/cost balance
   - A100 for fastest training (more expensive)
5. Choose Ubuntu 22.04 as the operating system
6. Set SSH key (upload your public key or create a new one)
7. Launch the instance
8. Note the IP address of your instance once it's running

## Step 3: Connect to Your Instance

```bash
# Connect via SSH (replace with your actual IP address)
ssh ubuntu@YOUR_INSTANCE_IP
```

## Step 4: Set Up the Environment

```bash
# Update package lists
sudo apt-get update

# Install git and other utilities
sudo apt-get install -y git tmux htop

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
echo 'export PATH=~/miniconda3/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Clone the repository
git clone https://github.com/YOUR_USERNAME/hmer-ink.git
cd hmer-ink

# Run the setup script (which will handle conda environment setup and dataset download)
chmod +x scripts/lambda_setup.sh
./scripts/lambda_setup.sh
```

The setup script will:
1. Install dependencies
2. Offer to download the MathWriting dataset (3GB)
3. Set up the conda environment using the Lambda-optimized environment file
4. Configure the Lambda-specific YAML file
5. Verify GPU availability

### Environment File for Lambda Labs

The repository includes `environment-lambda.yml`, which is specifically optimized for Lambda Labs GPU instances with:
- CUDA 11.8 support
- Optimized PyTorch installation
- DGL with CUDA support
- Proper environment variables for GPU computing

If this file is not present, you can create it yourself with:

```bash
# Create environment-lambda.yml with proper CUDA dependencies
cat > environment-lambda.yml << 'EOL'
name: hmer-ink
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - scikit-image
  - plotly
  - opencv
  - pillow
  - omegaconf
  - typer
  - pyyaml
  - jinja2
  - tqdm
  - optuna
  - pytest
  - black
  - isort
  - flake8
  - ruff
  - pydantic
  - mypy
  - shap
  - wandb
  - networkx
  - levenshtein
  - editdistance
  - albumentations
  - cudatoolkit=11.8
  - pytorch=2.0.1
  - torchvision
  - torchaudio
  - pip
  - pip:
      # Core ML/DL
      - torchmetrics
      - tensorboard
      
      # CLI and utilities
      - rich
      
      # Dashboard and reporting
      - streamlit>=1.26.0
      - altair>=5.0.0
      - tabulate
      - kaleido
      - umap-learn
      
      # Graph neural networks
      - dgl-cuda11.8
      - torch-geometric
      
      # LaTeX utilities
      - sympy
      - pylatexenc
      - latex2mathml
      
      # Additional packages for architectures
      - timm  # For vision transformer components
      - transformers  # For language model components
      - diffusers  # For DDPM augmentation

variables:
  CUDA_VISIBLE_DEVICES: "0"
  PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:128"
  OMP_NUM_THREADS: "16"
  MKL_NUM_THREADS: "16"
  NO_ALBUMENTATIONS_UPDATE: "1"
  HDF5_USE_FILE_LOCKING: "FALSE"
EOL
```

## Step 5: Prepare Your Dataset

### Option 1: Automatic Download (Recommended)

The setup script will offer to download the dataset automatically. If you choose this option, it will:
```bash
# Download the MathWriting dataset (3GB)
wget https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz

# Extract the dataset
tar -xzf mathwriting-2024.tgz -C data/
```

### Option 2: Manual Download on Lambda Instance

If you prefer to download manually or need a custom dataset:
```bash
# Make data directory
mkdir -p data

# Download the MathWriting dataset directly
wget https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz
tar -xzf mathwriting-2024.tgz -C data/
```

### Option 3: Transfer from Local Machine

Only use this if options 1 and 2 are not feasible:

On your local machine:
```bash
# Compress your dataset
tar -czf hmer-data.tar.gz data/

# Copy to Lambda instance (replace with your instance IP)
scp hmer-data.tar.gz ubuntu@YOUR_INSTANCE_IP:~/
```

On the Lambda instance:
```bash
# Extract dataset
cd ~/hmer-ink
tar -xzf ~/hmer-data.tar.gz
```

## Step 6: Run Training in a tmux Session

Using tmux ensures training continues if your connection drops:

```bash
# Create a new tmux session
tmux new -s hmer-training

# Activate environment
conda activate hmer-ink

# Run training with Lambda-optimized config
make train CONFIG=configs/lambda.yaml EXPERIMENT=lambda-run-v1

# Detach from tmux with Ctrl+B, then D
# You can reattach later with: tmux attach -t hmer-training
```

## Step 7: Monitor Training

```bash
# Reattach to tmux session to check progress
tmux attach -t hmer-training

# In a separate SSH session, monitor logs
ssh ubuntu@YOUR_INSTANCE_IP
tail -f ~/hmer-ink/outputs/lambda-run-v1/logs/train.log

# To monitor with the dashboard (if you installed streamlit):
make dashboard
```

## Step 8: Download Results

On your local machine:
```bash
# Create a directory for results
mkdir -p hmer-results

# Download results (replace with your instance IP)
scp -r ubuntu@YOUR_INSTANCE_IP:~/hmer-ink/outputs/* hmer-results/
```

## Step 9: Stopping Your Instance

When you're done, remember to stop your Lambda Labs instance to avoid unnecessary charges:

1. Log in to Lambda Labs dashboard
2. Go to "Instances"
3. Select your instance and click "Stop"
4. To resume later, click "Start"

## Performance Optimization Tips

1. **Adjust batch size**: If you encounter out-of-memory errors, reduce the batch size in `configs/lambda.yaml`
2. **Monitor GPU usage**: Use `nvidia-smi -l 1` to monitor GPU utilization and memory
3. **Increase workers**: Set `num_workers` to higher values (8-16) for faster data loading
4. **Use mixed precision**: Make sure `use_amp: true` is set in the config

## Troubleshooting

### Out of CUDA memory

If you encounter CUDA memory errors:
```bash
# Edit the Lambda config
nano configs/lambda.yaml

# Reduce batch size (e.g., from 128 to 64)
# Increase gradient_accumulation_steps (e.g., from 1 to 2)
```

### Slow data loading

If data loading is slow:
```bash
# Edit the Lambda config
nano configs/lambda.yaml

# Increase num_workers (e.g., from 8 to 16)
```

### Connection issues

If your SSH connection is unstable:
1. Always use tmux for training
2. Consider setting up automatic checkpointing more frequently:
   ```yaml
   # In configs/lambda.yaml
   training:
     save_every_n_epochs: 1  # Save checkpoints every epoch
   ``` 