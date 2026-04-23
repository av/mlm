FROM ghcr.io/ggml-org/llama.cpp:server-rocm
RUN apt-get update && apt-get install -y cmake ninja-build git curl && rm -rf /var/lib/apt/lists/*
ENV PATH=/opt/rocm-7.0.0/bin:/opt/rocm-7.0.0/llvm/bin:$PATH
ENV ROCM_PATH=/opt/rocm-7.0.0
ENV HIP_PATH=/opt/rocm-7.0.0
