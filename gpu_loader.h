// GPU-Accelerated Model Loader - Pipeline Header
// Orchestrates chunked loading from GGUF files to GPU processing

#ifndef GPU_LOADER_H
#define GPU_LOADER_H

#include "gguf_reader.h"
#include <memory>
#include <vector>
#include <string>

// GPU memory management
struct GPUBuffer {
    void* device_ptr;
    size_t size;
    size_t capacity;
    
    GPUBuffer() : device_ptr(nullptr), size(0), capacity(0) {}
    ~GPUBuffer();
    
    bool allocate(size_t bytes);
    void free();
};

// Loading progress callback
typedef void (*ProgressCallback)(float progress, const std::string& status, void* user_data);

// Main GPU loader class
class GPULoader {
public:
    GPULoader();
    ~GPULoader();
    
    // Configuration
    void set_chunk_size(size_t bytes) { chunk_size = bytes; }
    void set_progress_callback(ProgressCallback callback, void* user_data);
    void set_gpu_memory_limit(size_t bytes) { gpu_memory_limit = bytes; }
    
    // Core loading functionality
    bool load_model(const std::string& gguf_file, const std::string& output_file = "");
    bool process_tensor(size_t tensor_index, const std::string& output_file = "");
    
    // GPU processing
    bool process_q4k_chunk(const void* input_chunk, size_t input_size, 
                          float* output_chunk, size_t output_size);
    bool process_q8_chunk(const void* input_chunk, size_t input_size,
                         float* output_chunk, size_t output_size);
    bool process_f16_chunk(const void* input_chunk, size_t input_size,
                          float* output_chunk, size_t output_size);
    
    // Status and info
    bool is_gpu_available() const { return gpu_available; }
    size_t get_gpu_memory() const { return gpu_total_memory; }
    size_t get_optimal_chunk_size() const;
    
    const std::string& get_last_error() const { return last_error; }
    
    // Statistics
    struct LoadingStats {
        size_t total_tensors;
        size_t processed_tensors;
        size_t total_bytes;
        size_t processed_bytes;
        double cpu_time_seconds;
        double gpu_time_seconds;
        double io_time_seconds;
        float speedup_factor;
    };
    
    const LoadingStats& get_stats() const { return stats; }
    void print_stats() const;

private:
    // GPU management
    bool init_gpu();
    void cleanup_gpu();
    
    // Memory buffers (double buffering for pipeline)
    GPUBuffer input_buffer_a, input_buffer_b;
    GPUBuffer output_buffer_a, output_buffer_b;
    std::vector<uint8_t> host_input_buffer;
    std::vector<float> host_output_buffer;
    
    // Configuration
    size_t chunk_size;              // Default: 2GB
    size_t gpu_memory_limit;        // Available GPU memory
    ProgressCallback progress_cb;
    void* progress_user_data;
    
    // GPU info
    bool gpu_available;
    int gpu_device_id;
    size_t gpu_total_memory;
    size_t gpu_free_memory;
    
    // File handling
    std::unique_ptr<GGUFReader> reader;
    
    // State
    std::string last_error;
    LoadingStats stats;
    
    // Helper functions
    void set_error(const std::string& error);
    void update_progress(float progress, const std::string& status);
    size_t calculate_output_size(uint32_t tensor_type, size_t input_size);
    bool write_tensor_to_file(const std::string& filename, size_t tensor_index, 
                             const float* data, size_t size);
    
    // GPU kernel wrappers
    bool launch_q4k_kernel(const void* input, float* output, size_t n_blocks);
    bool launch_q8_kernel(const void* input, float* output, size_t n_blocks);
    bool launch_f16_kernel(const void* input, float* output, size_t n_elements);
    
    // Disable copy
    GPULoader(const GPULoader&) = delete;
    GPULoader& operator=(const GPULoader&) = delete;
};

// Convenience functions
bool gpu_accelerated_load(const std::string& input_file, const std::string& output_file,
                         ProgressCallback progress_cb = nullptr, void* user_data = nullptr);
bool check_gpu_compatibility();
void print_gpu_info();

#endif // GPU_LOADER_H