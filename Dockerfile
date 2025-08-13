# Use Ubuntu 20.04 with manual CUDA install
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install basic deps
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    gcc-9 \
    g++-9 \
    && rm -rf /var/lib/apt/lists/*

# Set GCC 9 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 60

# Install CUDA toolkit 11.8
RUN wget -O /tmp/cuda.pin https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv /tmp/cuda.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && \
    apt-get install -y cuda-toolkit-11-8 && \
    rm -rf /var/lib/apt/lists/*

# Set CUDA environment
ENV PATH=/usr/local/cuda-11.8/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

WORKDIR /gaml
COPY . .

# Fix Makefile for container
RUN sed -i 's/g++-14/g++-9/g' Makefile

CMD ["bash"]