FROM ghcr.io/walkerlab/docker-pytorch-jupyter-cuda:cuda-11.8.0-pytorch-1.13.0-torchvision-0.14.0-torchaudio-0.13.0-ubuntu-22.04

LABEL maintainer='vaibrainium (vaibhavt459@gmail.com)'


COPY . /src/

RUN apt-get update -y
RUN apt-get upgrade -y
RUN pip3 install -e /src/.
RUN pip3 install -r /src/requirements.txt  


RUN pip3 install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
