# Use CUDA 12.4 base image
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
# Base multi-arch oficial do Python, compatível com ARM (Mac M1)
#FROM python:3.10-slim

# Variáveis de ambiente básicas
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Configura MuJoCo e define o backend headless para renderização via OSMesa
ENV MJLIB_PATH=/tf/mujoco200_linux/bin/libmujoco200.so
ENV MJKEY_PATH=/tf/mujoco200_linux/bin/mjkey.txt
ENV MUJOCO_GL=osmesa
ENV LD_LIBRARY_PATH=/tf/mujoco200_linux/bin/
ENV PYOPENGL_PLATFORM=osmesa
# Força o Mesa a usar OpenGL 3.3 (ajuda na compatibilidade)
ENV MESA_GL_VERSION_OVERRIDE=3.3

# Atualiza os repositórios e instala as dependências necessárias para Mesa/OSMesa
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libgl1-mesa-dri \
    libglapi-mesa \
    mesa-utils \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Cria o ambiente de trabalho
WORKDIR /app

# Cria um ambiente virtual Python
RUN python -m venv /venv

# Atualiza o PATH para usar o ambiente virtual
ENV PATH="/venv/bin:$PATH"

# Copia o arquivo de dependências (requirements.txt)
COPY requirements.txt ./

# Atualiza o pip, força a instalação do PyOpenGL na versão 3.1.0 (mais estável para headless) 
# e instala as demais dependências
RUN pip install --upgrade pip && \
    pip install PyOpenGL==3.1.0 && \
    pip install -r requirements.txt

# Define o diretório de trabalho para a aplicação
WORKDIR /dreamer

# Copia o restante da aplicação
COPY . .

# Argumento de build para a chave do wandb (atenção: evite expor dados sensíveis)
ARG WANDB_KEY
ENV WANDB_API_KEY=wandbkey

# Comando para executar o main.py ao iniciar o container
CMD ["python", "main.py"]
