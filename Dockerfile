# ── damping-wings Dockerfile ─────────────────────────────────────────────────
# Multi-stage build using Miniconda to handle py21cmfast C dependencies.
# Includes NVIDIA GPU support for CUDA-accelerated 21cmFAST runs.
#
# Build:  docker build -t damping-wings:latest .
# Run:    docker run --rm --gpus all \
#             -v /your/output:/app/output \
#             -e DAMPING_WINGS_OUTPUT=/app/output \
#             damping-wings:latest

# ── Stage 1: Base image with conda ────────────────────────────────────────────
FROM continuumio/miniconda3:23.10.0-1 AS base

# Metadata
LABEL maintainer="Yash Mohan Sharma <yashmohansharma96@gmail.com>"
LABEL description="damping-wings: parametric simulation pipeline and Fisher Matrix inference"
LABEL version="0.1.0"

# Set working directory
WORKDIR /app

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies needed by py21cmfast C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgsl-dev \
    libfftw3-dev \
    libhdf5-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: Conda environment ────────────────────────────────────────────────
FROM base AS conda-env

# Copy environment file first — better Docker layer caching
COPY environment.yml .

# Create conda environment
# Use --no-default-packages to keep it clean
RUN conda env create -f environment.yml \
    && conda clean -afy

# Make conda environment the default shell
SHELL ["conda", "run", "-n", "damping-wings", "/bin/bash", "-c"]

# ── Stage 3: Install package ──────────────────────────────────────────────────
FROM conda-env AS package

# Copy source code
COPY src/ ./src/
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .

# Install the package in editable mode inside the conda environment
RUN conda run -n damping-wings pip install -e .

# ── Stage 4: Runtime ──────────────────────────────────────────────────────────
FROM package AS runtime

# Copy examples
COPY examples/ ./examples/

# Create default output directory
RUN mkdir -p /app/output/plots /app/output/txt_files /app/output/cache_files

# Environment variables
ENV DAMPING_WINGS_OUTPUT=/app/output
ENV CONDA_DEFAULT_ENV=damping-wings
ENV PATH=/opt/conda/envs/damping-wings/bin:$PATH

# Verify installation
RUN conda run -n damping-wings python -c "\
import damping_wings; \
print(f'damping-wings {damping_wings.__version__} installed successfully')"

# Default command — runs the example pipeline
# Override with: docker run ... python examples/your_script.py
CMD ["conda", "run", "--no-capture-output", "-n", "damping-wings", \
     "python", "examples/modelling.py"]