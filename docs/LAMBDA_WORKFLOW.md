# Lambda Labs Training Workflow

This document provides a visual overview of the training workflow on Lambda Labs.

## Training Workflow Diagram

```
┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
│                    │     │                    │     │                    │
│  Local Computer    │     │   Lambda Labs      │     │    Monitoring      │
│                    │     │   GPU Instance     │     │    & Results       │
│                    │     │                    │     │                    │
└────────┬───────────┘     └─────────┬──────────┘     └──────────┬─────────┘
         │                           │                           │
         ▼                           ▼                           ▼
┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
│1. Prepare Code     │     │5. Run Setup Script │     │9. Monitor Training │
│  - Clone repo      │     │  - ./scripts/      │     │  - tmux attach     │
│  - Test locally    │     │    lambda_setup.sh │     │  - make dashboard  │
└────────┬───────────┘     └─────────┬──────────┘     └──────────┬─────────┘
         │                           │                           │
         ▼                           ▼                           ▼
┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
│2. Prepare Data     │     │6. Prepare Training │     │10. Evaluate Model  │
│  - Format dataset  │     │  - Verify data     │     │  - make evaluate   │
│  - Compress data   │     │  - Check config    │     │    MODEL=...       │
└────────┬───────────┘     └─────────┬──────────┘     └──────────┬─────────┘
         │                           │                           │
         ▼                           ▼                           ▼
┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
│3. Launch Lambda    │     │7. Start Training   │     │11. Download Results│
│  Instance          │     │  - tmux new        │     │  - scp outputs/    │
│  - Choose GPU      │     │  - make train      │     │    to local        │
└────────┬───────────┘     └─────────┬──────────┘     └──────────┬─────────┘
         │                           │                           │
         ▼                           ▼                           ▼
┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
│4. Upload Code/Data │     │8. Detach from tmux │     │12. Stop Instance   │
│  - scp local files │     │  - Ctrl+B, then D  │     │  - Prevent charges │
│  - SSH to instance │     │  - Process runs    │     │  - Save work first!│
└────────────────────┘     └────────────────────┘     └────────────────────┘
```

## Key Steps in Detail

### 1. Local Preparation

- Clone the repository: `git clone https://github.com/YOUR_USERNAME/hmer-ink.git`
- Test the model locally if possible with small dataset
- Check that configs/lambda.yaml exists or will be created by the setup script

### 2. Data Preparation

- Format your dataset according to the expected structure
- Organize into train/valid/test folders
- Compress your data: `tar -czf hmer-data.tar.gz data/`
- https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz

### 3. Launch Lambda Labs Instance

- Log in to Lambda Labs console
- Select appropriate GPU (L4, A10, or A100 for faster training)
- Set up SSH keys and launch instance

### 4. Upload to Instance

- SSH into your instance: `ssh ubuntu@YOUR_INSTANCE_IP`
- Upload code: `scp -r hmer-ink/ ubuntu@YOUR_INSTANCE_IP:~/`
- Upload data: `scp hmer-data.tar.gz ubuntu@YOUR_INSTANCE_IP:~/`

### 5. Run Setup Script

- Extract data: `tar -xzf hmer-data.tar.gz`
- Run setup script: `./scripts/lambda_setup.sh`
- This will install dependencies, create environment, and configure for GPU

### 6. Prepare for Training

- Verify data is available: `ls -la data/`
- Check CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Review configs/lambda.yaml for any adjustments

### 7. Start Training

- Create tmux session: `tmux new -s hmer-training`
- Activate environment: `conda activate hmer-ink`
- Start training: `make train CONFIG=configs/lambda.yaml EXPERIMENT=lambda-run-v1`

### 8. Detach from tmux

- Press `Ctrl+B`, then `D` to detach
- Training will continue in the background
- You can safely close your SSH session

### 9. Monitor Training

- Reattach to session: `tmux attach -t hmer-training`
- Check logs: `tail -f outputs/lambda-run-v1/logs/train.log`
- Start dashboard: `make dashboard`

### 10. Evaluate Model

- Once training is complete, evaluate:
- `make evaluate MODEL=outputs/lambda-run-v1/checkpoints/best_model.pt`

### 11. Download Results

- From your local machine:
- `scp -r ubuntu@YOUR_INSTANCE_IP:~/hmer-ink/outputs/* hmer-results/`

### 12. Stop the Instance

- Stop the instance via Lambda Labs dashboard
- Only start it when needed to save costs 