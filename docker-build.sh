#!/bin/bash
# GAML Docker Build Script for nvidia-container-toolkit environments

set -e

echo "🐋 Building GAML in Docker with CUDA support..."

# Build the Docker image
docker build -t gaml:latest .

echo "✅ Build complete!"
echo ""
echo "Usage examples:"
echo "  docker run --rm --gpus all gaml:latest --gpu-info"
echo "  docker run --rm --gpus all -v /path/to/models:/models gaml:latest /models/model.gguf"
echo "  docker run --rm --gpus all gaml:latest --benchmark"
echo ""
echo "For interactive shell:"
echo "  docker run --rm -it --gpus all gaml:latest bash"