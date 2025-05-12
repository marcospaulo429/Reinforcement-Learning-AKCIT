# Use CUDA 12.4 base image
#FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
# Base multi-arch oficial do Python, compatível com ARM (Mac M1)
FROM python:3.10-slim

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-distutils \
    python3-pip \
    libgl1-mesa-glx \
    libosmesa6 \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev \
    patchelf \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Copia e instala dependências do Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install PyOpenGL==3.1.0 && \
    pip install -r requirements.txt

# Copia o código-fonte para o container
COPY . .

# Define o comando padrão
CMD ["python", "main.py"]
