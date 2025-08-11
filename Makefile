# GPU-Accelerated GGUF Loading (GAML) - Makefile

# Compilers and flags
NVCC = nvcc
CXX = g++
CUDA_FLAGS = -std=c++17 -O3 -arch=sm_61
CXX_FLAGS = -std=c++17 -O3 -Wall -Wextra

# For GTX 1070 Ti, use compute capability 6.1
# Adjust for your GPU: RTX 30xx = sm_86, RTX 40xx = sm_89
TARGET_ARCH = sm_61

# Libraries
CUDA_LIBS = -lcuda -lcudart
SYSTEM_LIBS = -pthread

# Source files
CUDA_SOURCES = cuda_q4k_dequant.cu
CXX_SOURCES = gguf_reader.cpp gpu_loader.cpp gaml.cpp
HEADERS = gguf_reader.h gpu_loader.h

# Object files
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
CXX_OBJECTS = $(CXX_SOURCES:.cpp=.o)

# Targets
TARGETS = gaml gaml_poc

all: $(TARGETS)

# Main GAML tool (complete pipeline)
gaml: $(CXX_OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) $(CUDA_FLAGS) -arch=$(TARGET_ARCH) -o $@ $^ $(CUDA_LIBS) $(SYSTEM_LIBS)

# Proof of concept (original benchmark)
gaml_poc: cuda_q4k_dequant.cu
	$(NVCC) $(CUDA_FLAGS) -arch=$(TARGET_ARCH) -o $@ $< $(CUDA_LIBS)

# CUDA object files
%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -arch=$(TARGET_ARCH) -c $< -o $@

# C++ object files  
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Installation
install: gaml
	@echo "Installing GAML..."
	sudo cp gaml /usr/local/bin/
	@echo "GAML installed to /usr/local/bin/gaml"

# Clean build artifacts
clean:
	rm -f $(TARGETS) $(CXX_OBJECTS) $(CUDA_OBJECTS)

# Test runs (require NVIDIA GPU)
test-poc: gaml_poc
	./gaml_poc

test-gpu: gaml
	./gaml --gpu-info

test-benchmark: gaml
	./gaml --benchmark

# Development helpers
check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version || echo "NVCC not found - install CUDA toolkit"
	@nvidia-smi || echo "NVIDIA driver not found"

debug: CUDA_FLAGS += -g -G
debug: CXX_FLAGS += -g
debug: gaml

# Different GPU architectures
pascal: TARGET_ARCH = sm_61
pascal: gaml

turing: TARGET_ARCH = sm_75  
turing: gaml

ampere: TARGET_ARCH = sm_86
ampere: gaml

ada: TARGET_ARCH = sm_89
ada: gaml

# Help target
help:
	@echo "GAML Build System"
	@echo "=================="
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build everything (default)"
	@echo "  gaml         - Build main GAML tool"
	@echo "  gaml_poc     - Build proof-of-concept benchmark"
	@echo "  clean        - Remove build artifacts"
	@echo "  install      - Install GAML to /usr/local/bin"
	@echo ""
	@echo "Testing:"
	@echo "  test-poc     - Run proof-of-concept benchmark"
	@echo "  test-gpu     - Check GPU compatibility" 
	@echo "  test-benchmark - Run performance benchmark"
	@echo "  check-cuda   - Verify CUDA installation"
	@echo ""
	@echo "GPU Targets:"
	@echo "  pascal       - Build for GTX 10xx series (sm_61)"
	@echo "  turing       - Build for RTX 20xx series (sm_75)"
	@echo "  ampere       - Build for RTX 30xx series (sm_86)"
	@echo "  ada          - Build for RTX 40xx series (sm_89)"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make                    # Build for GTX 1070 Ti"
	@echo "  make ampere            # Build for RTX 3080"
	@echo "  make test-gpu          # Check if GPU works"
	@echo "  ./gaml model.gguf      # Process a model"

.PHONY: all clean install test-poc test-gpu test-benchmark check-cuda debug pascal turing ampere ada help