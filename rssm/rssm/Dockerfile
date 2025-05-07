# Use CUDA 12.4 base image
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
# Base multi-arch oficial do Python, compatível com ARM (Mac M1)
#FROM python:3.10-slim

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libosmesa6 \
    libosmesa6-dev \
    libglfw3 \
    patchelf \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# Configura o ambiente Python
WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install PyOpenGL==3.1.0 && \
    pip install -r requirements.txt

# Copia o restante do código
COPY . .

# Comando de execução
CMD ["python3", "trainer.py"]