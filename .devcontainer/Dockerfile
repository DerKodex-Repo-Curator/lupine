FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV LIBRARY_PATH="${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    clang-format \
    cmake \
    git \
    libnghttp2-dev \
    libssl-dev \
    ninja-build \
    pip \
    python3 \
    python3-pip \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

# Codegen dependency: cxxheaderparser parses CUDA headers to emit gen_*.cpp/.h.
RUN pip install --no-cache-dir --break-system-packages cxxheaderparser

# Python adapter dev dependency
RUN pip install --no-cache-dir --break-system-packages pytest>=8.0

WORKDIR /workspaces/lupine