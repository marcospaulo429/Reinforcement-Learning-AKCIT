FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
#FROM nvidia/cuda:11.4.2-runtime-ubuntu20.04

# install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && \
    apt-get -y install python3-pip xvfb ffmpeg git build-essential python3-opengl
RUN ln -s /usr/bin/python3 /usr/bin/python

# install python dependencies
RUN mkdir cleanrl_utils && touch cleanrl_utils/__init__.py
RUN pip install poetry --upgrade
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock README.md
RUN poetry install --no-root

# install mujoco_py
RUN apt-get -y install wget unzip software-properties-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev patchelf
#RUN poetry install -E "atari mujoco"
RUN poetry install -E "atari mujoco" --no-root
RUN poetry run python -c "import mujoco"

# copy local files
COPY . /dreamer

# Define o ponto de entrada para rodar o script dreamer.py
# Usamos 'poetry run' para garantir que o script seja executado no ambiente Poetry
ENTRYPOINT ["poetry", "run", "python3", "/dreamer/dreamer.py"]