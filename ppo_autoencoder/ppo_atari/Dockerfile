FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && \
    apt-get -y install python3-pip xvfb ffmpeg git build-essential python-opengl
RUN ln -s /usr/bin/python3 /usr/bin/python

# install python dependencies
RUN mkdir cleanrl_utils && touch cleanrl_utils/__init__.py
RUN pip install poetry --upgrade
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
RUN poetry install

# install mujoco_py
RUN apt-get -y install wget unzip software-properties-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev patchelf
RUN poetry install -E "atari mujoco_py"
RUN poetry run python -c "import mujoco_py"

# copy local files
COPY . . 

RUN pip install --no-cache-dir -r requirements.txt

# Comando padrão
CMD ["python3", "main.py"]