FROM ghcr.io/walkerlab/docker-pytorch-jupyter-cuda:cuda-11.8.0-pytorch-1.13.0-torchvision-0.14.0-torchaudio-0.13.0-ubuntu-22.04

LABEL maintainer='vaibrainium (vaibhavt459@gmail.com)'

COPY . /src/

RUN apt-get update -y

# Install torch first since everything else needs to be compatible with it
RUN pip install torch==2.4.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Install numpy and cython once, cleanly
RUN pip install --no-cache-dir numpy cython

# Install ssm
RUN pip uninstall -y ssm || true
RUN pip install --no-cache-dir --force-reinstall git+https://github.com/vaibrainium/ssm.git

# Install remaining dependencies after numpy is stable
RUN pip install --no-cache-dir scipy matplotlib pandas
RUN pip install -r /src/requirements.txt
RUN pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN pip3 install -e /src/.
