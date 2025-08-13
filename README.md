# 🚀 GAML - GPU-Accelerated Model Loading

**Finally fix the 40-minute model loading bottleneck!**

GAML accelerates GGUF model loading using GPU parallel processing instead of slow CPU sequential operations. Load 70B models in **5-8 minutes** instead of 40+ minutes.

## 🎯 The Problem

Current GGUF loaders (Ollama, llama.cpp) are **pipeline-bound disasters**:
- Sequential processing of billions of quantized weights
- GPU idle 90% of the time while CPU struggles
- 40+ minutes to load large models (70B Q4_K)
- Terrible user experience for local AI

## ⚡ GAML Solution

**Overlapped GPU Pipeline**:
- Triple-buffered async processing
- Pinned memory + CUDA streams
- GPU busy while next chunk loads from disk
- 5-20x speedup vs CPU-only loading
- Bit-perfect accuracy guarantee

## 🏗️ Current Status: Complete Pipeline (v0.2)

**Phase 1 Complete**: Q4_K Dequantization Kernel ✅
- ✅ CUDA kernel for Q4_K quantized weights
- ✅ Parallel processing across GPU cores  
- ✅ Performance benchmarking vs CPU
- ✅ Accuracy verification

**Phase 2 Complete**: Full Loading Pipeline ✅
- ✅ GGUF file format parser
- ✅ Async triple-buffered loading system
- ✅ GPU memory management with streams
- ✅ Production CLI tool (`gaml`)

**Next Phases**:
- 🔄 Multi-format support (Q8_0, F16)
- 🔄 Advanced optimizations
- 🔄 Cross-platform GPU support
- 🔄 Integration with existing tools

## 🔧 Quick Start

### Option 1: Docker Build (Recommended)
Perfect for systems with nvidia-container-toolkit but no CUDA dev tools:

```bash
git clone https://github.com/Fimeg/GAML.git
cd GAML
./docker-build.sh

# Test GPU compatibility
docker run --rm --gpus all gaml:latest --gpu-info

# Process a model
docker run --rm --gpus all -v /path/to/models:/models gaml:latest /models/model.gguf

# Run benchmark
docker run --rm --gpus all gaml:latest --benchmark
```

### Option 2: Native Build
For systems with full CUDA toolkit installed:

```bash
# Prerequisites: CUDA Toolkit 11.0+
sudo dnf install nvidia-cuda-toolkit  # Fedora
# OR
sudo apt install nvidia-cuda-toolkit  # Ubuntu/Debian

git clone https://github.com/Fimeg/GAML.git
cd GAML
make check-cuda    # Verify CUDA installation
make              # Build complete GAML tool
make test-gpu     # Check GPU compatibility
./gaml --help     # See usage options
```

### Process a Model
```bash
# Basic usage
./gaml model.gguf

# Save processed tensors
./gaml model.gguf output/

# Custom chunk size
./gaml -c 512MB model.gguf

# Run benchmark
./gaml --benchmark
```

## 🎮 GPU Compatibility

**NVIDIA (Primary Target)**:
- GTX 1060, 1070, 1080 series ✅
- RTX 2060, 2070, 2080 series ✅  
- RTX 3060, 3070, 3080, 3090 series ✅
- RTX 4060, 4070, 4080, 4090 series ✅

**Requirements**:
- NVIDIA GPU with compute capability 6.1+
- 4GB+ VRAM minimum  
- nvidia-container-toolkit (Docker) OR CUDA Toolkit (native)

## 📊 Performance Expectations

| GPU | VRAM | Expected Speedup | 70B Load Time |
|-----|------|------------------|---------------|
| GTX 1070 Ti | 8GB | 5-10x | 5-8 min |
| RTX 3080 | 10GB | 10-15x | 3-4 min |
| RTX 4090 | 24GB | 15-20x | 2-3 min |

## 🔬 Technical Deep Dive

### The Real Bottleneck
```mermaid
graph LR
    A[Traditional: Sequential] --> B[Read→Process→Copy]
    B --> C[GPU Idle 90%]
    C --> D[40+ Minutes 😴]
    
    E[GAML: Overlapped] --> F[Read∥Process∥Copy]
    F --> G[GPU Busy 90%] 
    G --> H[5-8 Minutes ⚡]
```

### Triple-Buffer Pipeline
```
Buffer A: Loading chunk N+2 from disk
Buffer B: GPU processing chunk N+1  
Buffer C: Copying results from chunk N
ALL HAPPENING SIMULTANEOUSLY!
```

### Q4_K Format Optimization
Q4_K uses complex "super-block" quantization:
- 256 weights per block
- 4.5 bits per weight average
- Scales and minimums per sub-block
- **Perfect for GPU parallel processing!**

## 🚀 Roadmap

### v0.1 - Proof of Concept ✅
- Q4_K dequantization kernel
- Performance benchmarking
- Accuracy verification

### v0.2 - Complete Pipeline ✅
- GGUF file format parser
- Triple-buffered async loading
- Memory management with CUDA streams
- CLI tool (`gaml`)

### v0.3 - Multi-Format 🔄
- Q8_0 quantization support
- F16 half-precision support
- Auto-format detection
- Performance optimizations

### v1.0 - Production Ready 🎯
- Ollama/llama.cpp integration
- Cross-platform GPU support (AMD, Intel)
- Advanced memory management
- Error handling & fallbacks

## 🤝 Contributing

**We need help with**:
- AMD GPU support (ROCm/HIP)
- Intel GPU support (oneAPI) 
- macOS Metal implementation
- Integration testing
- Documentation

## 📜 License

MIT License - Build the future of local AI! 

---

**Time to fix what should have been built years ago.** 🚀

*No more 40-minute loading times. No more coffee breaks during model loads. Just fast, local AI that actually works.*

## 🐋 Docker Commands Quick Reference

```bash
# Build image
./docker-build.sh

# Check GPU
docker run --rm --gpus all gaml:latest --gpu-info

# Process model
docker run --rm --gpus all -v $(pwd):/workspace gaml:latest /workspace/model.gguf

# Benchmark
docker run --rm --gpus all gaml:latest --benchmark

# Interactive shell
docker run --rm -it --gpus all gaml:latest bash
```