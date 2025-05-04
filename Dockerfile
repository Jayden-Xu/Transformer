# base image
FROM python:3.9-slim

ENV TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install \
        numpy==1.24.4 \
        torch==2.0.1 \
        torchvision==0.15.2 \
        torchaudio==2.0.2 \
        torchtext==0.15.2 \
        torchdata==0.6.1 \
        datasets==2.15.0 \
        tokenizers==0.13.3 \
        torchmetrics==1.0.3 \
        tensorboard==2.13.0 \
        altair==5.1.1 \
        wandb==0.15.9

# copy code
COPY . /app

CMD ["/bin/bash"]
