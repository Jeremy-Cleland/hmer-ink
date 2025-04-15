#!/bin/bash
# Lambda Labs setup script for HMER-Ink
# This script will set up your Lambda Labs instance for HMER-Ink training

set -e  # Exit on any error

echo "Setting up Lambda Labs environment for HMER-Ink..."

# Update package lists
echo "Updating package lists..."
sudo apt-get update

# Install dependencies
echo "Installing dependencies..."
sudo apt-get install -y git tmux htop

# Create data directory if it doesn't exist
echo "Creating data directory..."
mkdir -p data

# Ask if user wants to download the dataset
echo "Do you want to download the MathWriting dataset (3GB)? (y/n)"
read download_dataset

if [[ $download_dataset == "y" || $download_dataset == "Y" ]]; then
    echo "Downloading MathWriting dataset..."
    wget https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz
    
    echo "Extracting dataset to data directory..."
    tar -xzf mathwriting-2024.tgz -C data/
    
    echo "Dataset downloaded and extracted successfully."
else
    echo "Skipping dataset download. You'll need to upload or download it manually later."
fi

# Check if conda is already installed
if command -v conda &> /dev/null; then
    echo "Conda is already installed, skipping installation."
else
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b
    echo 'export PATH=~/miniconda3/bin:$PATH' >> ~/.bashrc
    source ~/.bashrc
fi

# Create and activate conda environment
echo "Creating conda environment using Lambda-optimized environment file..."
if [ -f "environment-lambda.yml" ]; then
    conda env create -f environment-lambda.yml
else
    echo "Lambda environment file not found. Using standard environment file..."
    conda env create -f environment.yml
    echo "Note: For better CUDA compatibility, consider creating environment-lambda.yml"
fi

# Verify CUDA is available
echo "Checking CUDA availability..."
source ~/miniconda3/bin/activate hmer-ink
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'CUDA devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}');"

# Copy the Lambda config file if needed
if [ ! -f "configs/lambda.yaml" ]; then
    echo "Creating Lambda config file..."
    cp configs/default.yaml configs/lambda.yaml
    
    # Update device to CUDA
    sed -i 's/device: "mps"/device: "cuda"/g' configs/lambda.yaml
    
    # Update batch size
    sed -i 's/batch_size: 32/batch_size: 128/g' configs/lambda.yaml
    
    # Update gradient accumulation steps
    sed -i 's/gradient_accumulation_steps: 6/gradient_accumulation_steps: 1/g' configs/lambda.yaml
    
    # Update number of workers
    sed -i 's/num_workers: 4/num_workers: 8/g' configs/lambda.yaml
    
    # Update evaluation batch size
    sed -i 's/batch_size: 32/batch_size: 128/g' configs/lambda.yaml
    
    # Update project name
    sed -i 's/project_name: "hmer-ink-m4-max"/project_name: "hmer-ink-lambda"/g' configs/lambda.yaml
fi

# Create a simple script to check data directory structure
echo "Checking dataset structure..."
if [ -d "data/train" ] && [ -d "data/valid" ] && [ -d "data/test" ]; then
    echo "✅ Dataset structure looks correct with train, valid, and test directories."
else
    echo "⚠️ Warning: Expected dataset directories (train, valid, test) not found."
    echo "   You may need to organize the data files into the correct structure."
fi

echo ""
echo "Setup complete! You can now run:"
echo "  tmux new -s hmer-training"
echo "  conda activate hmer-ink"
echo "  make train CONFIG=configs/lambda.yaml EXPERIMENT=lambda-run-v1" 