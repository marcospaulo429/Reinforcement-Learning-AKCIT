# Use CUDA 12.4 base image
#FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 # Especifica a imagem base
FROM python:3.10-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    python3.10 \
    python3.10-dev \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Create and activate virtual environment using Python 3.10 
RUN uv venv /opt/venv --python python3.10
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip e instale dependências básicas
RUN uv pip install --upgrade pip --link-mode=copy && \
    uv pip install wandb setuptools wheel --link-mode=copy

# Set working directory
WORKDIR /app

# Copia todo o código do projeto
COPY . .

# Baixa as dependências necessárias conforme nosso arquivo de requirements.txt
RUN uv pip install -r requirements.txt --link-mode=copy

# Add build arguments (ajuste conforme sua necessidade)
ARG GIT_COMMIT
ARG WANDB_KEY

# Define variáveis de ambiente para o treinamento
ENV GIT_COMMIT=$GIT_COMMIT
ENV WANDB_API_KEY=$WANDB_KEY

# Entry point: executa o arquivo main.py
ENTRYPOINT ["python", "main.py"]
