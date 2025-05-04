# Implement Transformer Mostly From Scratch

This repository provides a mostly-from-scratch implementation of the Transformer architecture (Vaswani et al., 2017), focusing on learning and clarity.

## Project Structure

- `models.py` — PyTorch implementation of the Transformer architecture.
- Other files (e.g. `train_wb.py`) — **Copied** from [hkproj/pytorch-transformer](https://github.com/hkproj/pytorch-transformer) for training and logging purposes.

## Running with Docker

A Docker environment is provided to simplify setup and ensure consistency. If you prefer to run locally, refer to `requirements.txt`.

### 1. Build Docker Image

```bash
docker build -t attention-py39 .
```

### 2. Run Docker Container

```bash
docker run -it -v $(pwd):/app attention-py39
```

This mounts the current directory into the container at `/app`.

### 3. Authenticate with Hugging Face

Inside the container:

```bash
huggingface-cli login
```

Log into your Hugging Face account.

### 4. Start Training

```bash
python train_wb.py
```

Training progress will be logged to [Weights & Biases](https://wandb.ai) if configured.

## Monitoring

This project integrates with **Weights & Biases (W&B)** for experiment tracking. Make sure you're logged in via `wandb login` or have the environment variable `WANDB_API_KEY` set.

## Credit

Thanks to [hkproj](https://github.com/hkproj/pytorch-transformer) for the excellent reference implementation.
