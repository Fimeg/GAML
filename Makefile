# GPU-Accelerated GGUF Loading (GAML) - Makefile

# CUDA compiler and flags
NVCC = nvcc
CUDA_FLAGS = -std=c++17 -lcuda -O3 -arch=sm_61

# For GTX 1070 Ti, use compute capability 6.1
# Adjust for your GPU: RTX 30xx = sm_86, RTX 40xx = sm_89
TARGET_ARCH = sm_61

# Targets
TARGETS = gaml_poc

all: $(TARGETS)

gaml_poc: cuda_q4k_dequant.cu
	$(NVCC) $(CUDA_FLAGS) -arch=$(TARGET_ARCH) -o $@ $<

clean:
	rm -f $(TARGETS) *.o

# Test run (requires NVIDIA GPU)
test: gaml_poc
	./gaml_poc

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version || echo "NVCC not found - install CUDA toolkit"
	@nvidia-smi || echo "NVIDIA driver not found"

.PHONY: all clean test check-cuda