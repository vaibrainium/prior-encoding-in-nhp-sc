FROM ghcr.io/walkerlab/docker-pytorch-jupyter-cuda:cuda-11.8.0-pytorch-1.13.0-torchvision-0.14.0-torchaudio-0.13.0-ubuntu-22.04

LABEL maintainer='vaibrainium (vaibhavt459@gmail.com)'

WORKDIR /src
COPY . /src/

RUN apt-get update -y

RUN pip3 install uv
RUN uv pip install -e . # for development
# RUN uv pip install --system . # for production, install in system python environment
