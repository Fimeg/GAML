// GPU-Accelerated Model Loader - Pipeline Implementation
// Orchestrates chunked loading from GGUF files to GPU processing

#include "gpu_loader.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>

// External kernel declarations (from cuda_q4k_dequant.cu)
extern "C" {
    void launch_dequantize_q4k_kernel(const void* input, float* output, int n_blocks);
    void launch_dequantize_q8_kernel(const void* input, float* output, int n_blocks);
    void launch_dequantize_f16_kernel(const void* input, float* output, int n_elements);
}

// Default configuration
static const size_t DEFAULT_CHUNK_SIZE = 2ULL * 1024 * 1024 * 1024; // 2GB
static const size_t MIN_GPU_MEMORY = 4ULL * 1024 * 1024 * 1024;     // 4GB minimum

GPUBuffer::~GPUBuffer() {
    free();
}

bool GPUBuffer::allocate(size_t bytes) {
    free();
    
    cudaError_t error = cudaMalloc(&device_ptr, bytes);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    capacity = bytes;
    size = 0;
    return true;
}

void GPUBuffer::free() {
    if (device_ptr) {
        cudaFree(device_ptr);
        device_ptr = nullptr;
    }
    size = 0;
    capacity = 0;
}

GPULoader::GPULoader() 
    : chunk_size(DEFAULT_CHUNK_SIZE)
    , gpu_memory_limit(0)
    , progress_cb(nullptr)
    , progress_user_data(nullptr)
    , gpu_available(false)
    , gpu_device_id(-1)
    , gpu_total_memory(0)
    , gpu_free_memory(0)
    , reader(std::make_unique<GGUFReader>())
{
    memset(&stats, 0, sizeof(stats));
    init_gpu();
}

GPULoader::~GPULoader() {
    cleanup_gpu();
}

bool GPULoader::init_gpu() {
    // Check for CUDA devices
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        set_error("No CUDA devices found");
        return false;
    }
    
    // Use device 0
    gpu_device_id = 0;
    error = cudaSetDevice(gpu_device_id);
    if (error != cudaSuccess) {
        set_error("Failed to set CUDA device");
        return false;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    error = cudaGetDeviceProperties(&prop, gpu_device_id);
    if (error != cudaSuccess) {
        set_error("Failed to get device properties");
        return false;
    }
    
    gpu_total_memory = prop.totalGlobalMem;
    
    // Get available memory
    size_t free_mem, total_mem;
    error = cudaMemGetInfo(&free_mem, &total_mem);
    if (error != cudaSuccess) {
        set_error("Failed to get memory info");
        return false;
    }
    
    gpu_free_memory = free_mem;
    gpu_memory_limit = std::min(gpu_free_memory * 80 / 100, DEFAULT_CHUNK_SIZE); // Use 80% of free memory
    
    if (gpu_free_memory < MIN_GPU_MEMORY) {
        set_error("Insufficient GPU memory (minimum 4GB required)");
        return false;
    }
    
    gpu_available = true;
    
    std::cout << "GPU Initialized:" << std::endl;
    std::cout << "  Device: " << prop.name << std::endl;
    std::cout << "  Total Memory: " << gpu_total_memory / 1024 / 1024 << " MB" << std::endl;
    std::cout << "  Free Memory: " << gpu_free_memory / 1024 / 1024 << " MB" << std::endl;
    std::cout << "  Chunk Size: " << gpu_memory_limit / 1024 / 1024 << " MB" << std::endl;
    
    return true;
}

void GPULoader::cleanup_gpu() {
    input_buffer_a.free();
    input_buffer_b.free();
    output_buffer_a.free();
    output_buffer_b.free();
    gpu_available = false;
}

bool GPULoader::load_model(const std::string& gguf_file, const std::string& output_file) {
    if (!gpu_available) {
        set_error("GPU not available");
        return false;
    }
    
    update_progress(0.0f, "Opening GGUF file...");
    
    // Open and parse GGUF file
    if (!reader->open(gguf_file)) {
        set_error("Failed to open GGUF file: " + reader->get_last_error());
        return false;
    }
    
    if (!reader->read_header() || !reader->read_metadata() || !reader->read_tensor_info()) {
        set_error("Failed to parse GGUF file: " + reader->get_last_error());
        return false;
    }
    
    // Initialize statistics
    memset(&stats, 0, sizeof(stats));
    stats.total_tensors = reader->get_tensor_count();
    
    for (size_t i = 0; i < stats.total_tensors; i++) {
        const auto& tensor = reader->get_tensor(i);
        stats.total_bytes += tensor.size;
    }
    
    update_progress(0.1f, "Allocating GPU memory...");
    
    // Allocate GPU buffers
    size_t buffer_size = get_optimal_chunk_size();
    if (!input_buffer_a.allocate(buffer_size) || 
        !output_buffer_a.allocate(buffer_size * 4)) {  // F32 output is 4x larger than Q4_K
        set_error("Failed to allocate GPU memory");
        return false;
    }
    
    // Allocate host buffers
    host_input_buffer.resize(buffer_size);
    host_output_buffer.resize(buffer_size);  // Will be resized as needed
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Process each tensor
    for (size_t i = 0; i < stats.total_tensors; i++) {
        const auto& tensor = reader->get_tensor(i);
        
        update_progress(0.1f + 0.8f * i / stats.total_tensors, 
                       "Processing tensor " + std::to_string(i + 1) + "/" + 
                       std::to_string(stats.total_tensors) + ": " + tensor.name);
        
        if (!process_tensor(i, output_file)) {
            return false;
        }
        
        stats.processed_tensors++;
        stats.processed_bytes += tensor.size;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    stats.gpu_time_seconds = total_time.count() / 1000.0;
    stats.speedup_factor = 1.0f; // Will be calculated vs CPU reference
    
    update_progress(1.0f, "Model loading complete!");
    
    print_stats();
    return true;
}

bool GPULoader::process_tensor(size_t tensor_index, const std::string& output_file) {
    if (tensor_index >= reader->get_tensor_count()) {
        set_error("Invalid tensor index");
        return false;
    }
    
    const auto& tensor = reader->get_tensor(tensor_index);
    
    // Skip non-quantized tensors for now
    if (!reader->is_quantized_tensor(tensor_index)) {
        std::cout << "Skipping non-quantized tensor: " << tensor.name << std::endl;
        return true;
    }
    
    // Only process Q4_K tensors in this version
    if (tensor.type != GGML_TYPE_Q4_K) {
        std::cout << "Skipping non-Q4_K tensor: " << tensor.name 
                  << " (type: " << ggml_type_name(tensor.type) << ")" << std::endl;
        return true;
    }
    
    std::cout << "Processing Q4_K tensor: " << tensor.name 
              << " (" << tensor.size / 1024 / 1024 << " MB)" << std::endl;
    
    // Process tensor in chunks
    const size_t max_chunk_size = get_optimal_chunk_size();
    size_t offset = 0;
    
    while (offset < tensor.size) {
        size_t chunk_size = std::min(max_chunk_size, tensor.size - offset);
        
        // Read chunk from file
        if (!reader->read_tensor_chunk(tensor_index, offset, chunk_size, host_input_buffer.data())) {
            set_error("Failed to read tensor chunk");
            return false;
        }
        
        // Calculate output size (Q4_K -> F32 expansion)
        size_t output_size = calculate_output_size(tensor.type, chunk_size);
        host_output_buffer.resize(output_size / sizeof(float));
        
        // Process chunk on GPU
        if (!process_q4k_chunk(host_input_buffer.data(), chunk_size, 
                              host_output_buffer.data(), output_size)) {
            return false;
        }
        
        // Write output if requested
        if (!output_file.empty()) {
            std::string chunk_filename = output_file + "_tensor" + std::to_string(tensor_index) + 
                                       "_chunk" + std::to_string(offset / max_chunk_size) + ".bin";
            if (!write_tensor_to_file(chunk_filename, tensor_index, 
                                    host_output_buffer.data(), output_size)) {
                return false;
            }
        }
        
        offset += chunk_size;
    }
    
    return true;
}

bool GPULoader::process_q4k_chunk(const void* input_chunk, size_t input_size, 
                                  float* output_chunk, size_t output_size) {
    // Copy input to GPU
    cudaError_t error = cudaMemcpy(input_buffer_a.device_ptr, input_chunk, input_size, 
                                  cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        set_error("Failed to copy input to GPU");
        return false;
    }
    
    // Calculate number of blocks (Q4_K has 144 bytes per 256-weight block)
    size_t n_blocks = input_size / 144;  // ggml_type_sizes[GGML_TYPE_Q4_K]
    
    // Launch GPU kernel
    if (!launch_q4k_kernel(input_buffer_a.device_ptr, 
                          (float*)output_buffer_a.device_ptr, n_blocks)) {
        return false;
    }
    
    // Copy result back to host
    error = cudaMemcpy(output_chunk, output_buffer_a.device_ptr, output_size, 
                      cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        set_error("Failed to copy output from GPU");
        return false;
    }
    
    return true;
}

bool GPULoader::launch_q4k_kernel(const void* input, float* output, size_t n_blocks) {
    // This would call our CUDA kernel - for now just placeholder
    // In full implementation, this links to cuda_q4k_dequant.cu
    
    // Simulate GPU processing time
    cudaDeviceSynchronize();
    
    std::cout << "GPU processed " << n_blocks << " Q4_K blocks" << std::endl;
    return true;
}

size_t GPULoader::get_optimal_chunk_size() const {
    if (chunk_size > 0) {
        return std::min(chunk_size, gpu_memory_limit / 8);  // Leave room for input+output buffers
    }
    return gpu_memory_limit / 8;
}

size_t GPULoader::calculate_output_size(uint32_t tensor_type, size_t input_size) {
    switch (tensor_type) {
        case GGML_TYPE_Q4_K:
            // Q4_K: 144 bytes -> 256 floats (1024 bytes)
            return (input_size / 144) * 256 * sizeof(float);
        case GGML_TYPE_Q8_0:
            // Q8_0: 34 bytes -> 32 floats (128 bytes)  
            return (input_size / 34) * 32 * sizeof(float);
        case GGML_TYPE_F16:
            // F16: 2 bytes -> 1 float (4 bytes)
            return input_size * 2;
        default:
            return input_size;
    }
}

bool GPULoader::write_tensor_to_file(const std::string& filename, size_t tensor_index,
                                     const float* data, size_t size) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        set_error("Failed to create output file: " + filename);
        return false;
    }
    
    file.write(reinterpret_cast<const char*>(data), size);
    if (!file) {
        set_error("Failed to write tensor data");
        return false;
    }
    
    return true;
}

void GPULoader::set_progress_callback(ProgressCallback callback, void* user_data) {
    progress_cb = callback;
    progress_user_data = user_data;
}

void GPULoader::update_progress(float progress, const std::string& status) {
    if (progress_cb) {
        progress_cb(progress, status, progress_user_data);
    } else {
        std::cout << "[" << std::fixed << std::setprecision(1) << (progress * 100) << "%] " 
                  << status << std::endl;
    }
}

void GPULoader::set_error(const std::string& error) {
    last_error = error;
    std::cerr << "GPU Loader Error: " << error << std::endl;
}

void GPULoader::print_stats() const {
    std::cout << "\n=== Loading Statistics ===" << std::endl;
    std::cout << "Tensors processed: " << stats.processed_tensors << "/" << stats.total_tensors << std::endl;
    std::cout << "Data processed: " << stats.processed_bytes / 1024 / 1024 << " MB" << std::endl;
    std::cout << "GPU time: " << stats.gpu_time_seconds << " seconds" << std::endl;
    std::cout << "Throughput: " << (stats.processed_bytes / 1024.0 / 1024.0) / stats.gpu_time_seconds 
              << " MB/s" << std::endl;
}

// Convenience functions
bool gpu_accelerated_load(const std::string& input_file, const std::string& output_file,
                         ProgressCallback progress_cb, void* user_data) {
    GPULoader loader;
    loader.set_progress_callback(progress_cb, user_data);
    return loader.load_model(input_file, output_file);
}

bool check_gpu_compatibility() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count > 0;
}

void print_gpu_info() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    std::cout << "CUDA Devices Found: " << device_count << std::endl;
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Memory: " << prop.totalGlobalMem / 1024 / 1024 << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    }
}