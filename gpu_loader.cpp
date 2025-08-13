// GPU-Accelerated Model Loader - Pipeline Implementation (Optimized)
// Overlapped async I/O -> host pinned buffers -> async H->D -> GPU kernel -> async D->H
#include "gpu_loader.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <iomanip>

// External kernel declarations (from cuda_q4k_dequant.cu)
extern "C" {
    void launch_dequantize_q4k_kernel(const void* input_dev, float* output_dev, int n_blocks, cudaStream_t stream);
    void launch_dequantize_q8_kernel(const void* input, float* output, int n_blocks, cudaStream_t stream);
    void launch_dequantize_f16_kernel(const void* input, float* output, int n_elements, cudaStream_t stream);
}

// Default configuration
static const size_t DEFAULT_CHUNK_SIZE = 256ULL * 1024 * 1024; // 256MB (reduced for better pipelining)
static const size_t MIN_GPU_MEMORY = 4ULL * 1024 * 1024 * 1024;     // 4GB minimum
static const int NUM_BUFFERS = 3; // Triple buffering

// --- GPUBuffer implementation ---
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

// --- GPULoader implementation ---
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

    // Use up to 75% of free memory for buffers (more aggressive for better throughput)
    gpu_memory_limit = std::min(gpu_free_memory * 3 / 4, (size_t)2ULL * 1024 * 1024 * 1024); // Max 2GB

    if (gpu_free_memory < MIN_GPU_MEMORY) {
        set_error("Insufficient GPU memory (minimum 4GB required)");
        return false;
    }

    gpu_available = true;

    std::cout << "GPU Initialized:" << std::endl;
    std::cout << "  Device: " << prop.name << std::endl;
    std::cout << "  Total Memory: " << gpu_total_memory / 1024 / 1024 << " MB" << std::endl;
    std::cout << "  Free Memory: " << gpu_free_memory / 1024 / 1024 << " MB" << std::endl;
    std::cout << "  Memory Limit: " << gpu_memory_limit / 1024 / 1024 << " MB" << std::endl;

    return true;
}

void GPULoader::cleanup_gpu() {
    // Free all resources
    for (auto &ev : buffer_done_events) {
        if (ev) cudaEventDestroy(ev);
    }
    buffer_done_events.clear();

    for (auto &s : buffer_streams) {
        if (s) cudaStreamDestroy(s);
    }
    buffer_streams.clear();

    for (auto &d : device_input_buffers) {
        if (d) cudaFree(d);
    }
    device_input_buffers.clear();

    for (auto &d : device_output_buffers) {
        if (d) cudaFree(d);
    }
    device_output_buffers.clear();

    for (auto &h : host_pinned_buffers) {
        if (h) cudaFreeHost(h);
    }
    host_pinned_buffers.clear();

    // Free old buffers for compatibility
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

    update_progress(0.05f, "Allocating buffers and CUDA streams...");

    // Choose a practical chunk size
    size_t buffer_chunk_size = std::min(chunk_size, gpu_memory_limit / 4); // Divide by 4 for input+output+overhead
    buffer_chunk_size = std::max(buffer_chunk_size, (size_t)(64ULL * 1024 * 1024)); // At least 64MB

    // Allocate triple buffers (host pinned + device input + device output)
    host_pinned_buffers.resize(NUM_BUFFERS, nullptr);
    device_input_buffers.resize(NUM_BUFFERS, nullptr);
    device_output_buffers.resize(NUM_BUFFERS, nullptr);
    buffer_streams.resize(NUM_BUFFERS, nullptr);
    buffer_done_events.resize(NUM_BUFFERS, nullptr);

    // Calculate maximum output size needed (Q4_K -> ~4x expansion)
    size_t max_output_chunk = (buffer_chunk_size / 144) * 256 * sizeof(float);

    for (int i = 0; i < NUM_BUFFERS; ++i) {
        // Pinned host buffer (for disk I/O)
        cudaError_t cerr = cudaHostAlloc(&host_pinned_buffers[i], buffer_chunk_size, cudaHostAllocDefault);
        if (cerr != cudaSuccess) {
            set_error(std::string("cudaHostAlloc failed: ") + cudaGetErrorString(cerr));
            return false;
        }

        // Device input buffer (holds packed quantized data)
        cerr = cudaMalloc(&device_input_buffers[i], buffer_chunk_size);
        if (cerr != cudaSuccess) {
            set_error(std::string("cudaMalloc (input) failed: ") + cudaGetErrorString(cerr));
            return false;
        }

        // Device output buffer (holds dequantized float data)
        cerr = cudaMalloc(&device_output_buffers[i], max_output_chunk);
        if (cerr != cudaSuccess) {
            set_error(std::string("cudaMalloc (output) failed: ") + cudaGetErrorString(cerr));
            return false;
        }

        // Create stream and event for each buffer
        cudaStreamCreate(&buffer_streams[i]);
        cudaEventCreateWithFlags(&buffer_done_events[i], cudaEventDisableTiming);
    }

    // Host output buffer (for final results if needed)
    host_output_buffer.clear();

    update_progress(0.08f, "Beginning overlapped load pipeline...");

    auto start_time = std::chrono::high_resolution_clock::now();

    // Process each tensor
    for (size_t i = 0; i < stats.total_tensors; i++) {
        const auto& tensor = reader->get_tensor(i);
        update_progress(0.08f + 0.8f * i / stats.total_tensors,
                        "Processing tensor " + std::to_string(i + 1) + "/" +
                        std::to_string(stats.total_tensors) + ": " + tensor.name);

        if (!process_tensor(i, output_file, buffer_chunk_size, max_output_chunk)) {
            return false;
        }

        stats.processed_tensors++;
        stats.processed_bytes += tensor.size;
    }

    // Wait for all outstanding operations to finish
    for (int b = 0; b < NUM_BUFFERS; ++b) {
        if (buffer_streams[b]) cudaStreamSynchronize(buffer_streams[b]);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    stats.gpu_time_seconds = total_time.count() / 1000.0;

    update_progress(1.0f, "Model loading complete!");
    print_stats();

    return true;
}

bool GPULoader::process_tensor(size_t tensor_index, const std::string& output_file,
                               size_t buffer_chunk_size, size_t max_output_chunk) {
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
        std::cout << "Skipping non-Q4_K tensor: " << tensor.name << " (type ID: " << tensor.type << ")" << std::endl;
        return true;
    }

    std::cout << "Processing Q4_K tensor: " << tensor.name
    << " (" << tensor.size / 1024 / 1024 << " MB)" << std::endl;

    // Process tensor in overlapped chunks using pinned/dev buffers
    size_t offset = 0;
    int buf_idx = 0;

    // Pre-allocate host output buffer if we need to write to file
    if (!output_file.empty()) {
        host_output_buffer.resize(max_output_chunk / sizeof(float));
    }

    while (offset < tensor.size) {
        size_t chunk_size = std::min(buffer_chunk_size, tensor.size - offset);

        // Get the current buffer's resources
        void* host_pinned = host_pinned_buffers[buf_idx];
        void* device_input = device_input_buffers[buf_idx];
        void* device_output = device_output_buffers[buf_idx];
        cudaStream_t stream = buffer_streams[buf_idx];
        cudaEvent_t done_event = buffer_done_events[buf_idx];

        // Wait for this buffer to be free (if it has a previous outstanding op)
        cudaError_t qerr = cudaEventQuery(done_event);
        if (qerr == cudaErrorNotReady) {
            // Previous op is still running, wait for it to complete
            cudaStreamSynchronize(stream);
        }

        // Read chunk from file directly into pinned host buffer
        auto io_start = std::chrono::high_resolution_clock::now();
        if (!reader->read_tensor_chunk(tensor_index, offset, chunk_size, host_pinned)) {
            set_error("Failed to read tensor chunk");
            return false;
        }
        auto io_end = std::chrono::high_resolution_clock::now();
        auto io_time = std::chrono::duration_cast<std::chrono::milliseconds>(io_end - io_start);
        stats.io_time_seconds += io_time.count() / 1000.0;

        // Calculate output size (Q4_K -> F32 expansion)
        size_t output_size = calculate_output_size(tensor.type, chunk_size);

        // Async H->D copy on buffer's stream
        cudaError_t err = cudaMemcpyAsync(device_input, host_pinned, chunk_size,
                                          cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            set_error(std::string("cudaMemcpyAsync H->D failed: ") + cudaGetErrorString(err));
            return false;
        }

        // Number of Q4_K blocks in chunk
        size_t n_blocks = chunk_size / 144;
        if (n_blocks == 0) {
            set_error("Chunk too small or invalid Q4_K block alignment");
            return false;
        }

        // Launch dequant kernel on this buffer's stream
        auto gpu_start = std::chrono::high_resolution_clock::now();
        launch_dequantize_q4k_kernel(device_input, (float*)device_output, (int)n_blocks, stream);
        auto gpu_end = std::chrono::high_resolution_clock::now();
        auto gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
        stats.gpu_time_seconds += gpu_time.count() / 1000.0;

        // If user asked to persist output to host (output_file) we need to copy result back async
        if (!output_file.empty()) {
            err = cudaMemcpyAsync(host_output_buffer.data(), device_output, output_size,
                                  cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess) {
                set_error(std::string("cudaMemcpyAsync D->H failed: ") + cudaGetErrorString(err));
                return false;
            }

            // Record event to know when host buffer contains valid data
            cudaEventRecord(done_event, stream);

            // Wait for event to ensure host_output_buffer is ready before writing to disk
            cudaEventSynchronize(done_event);

            // Write to file
            std::string chunk_filename = output_file + "_tensor" + std::to_string(tensor_index) +
            "_chunk" + std::to_string(offset / buffer_chunk_size) + ".bin";
            if (!write_tensor_to_file(chunk_filename, tensor_index,
                host_output_buffer.data(), output_size)) {
                return false;
                }
        } else {
            // Not writing back to host — just record event to know when device buffer & kernel are done
            cudaEventRecord(done_event, stream);
        }

        // Advance to next buffer
        offset += chunk_size;
        buf_idx = (buf_idx + 1) % NUM_BUFFERS;
    }

    return true;
                               }

                               bool GPULoader::process_q4k_chunk(const void* input_chunk, size_t input_size,
                                                                 float* output_chunk, size_t output_size) {
                                   // This is a fallback synchronous path for compatibility
                                   // Copy input to GPU
                                   cudaError_t error = cudaMemcpy(input_buffer_a.device_ptr, input_chunk, input_size,
                                                                  cudaMemcpyHostToDevice);
                                   if (error != cudaSuccess) {
                                       set_error("Failed to copy input to GPU (fallback path)");
                                       return false;
                                   }

                                   // Calculate number of blocks (Q4_K has 144 bytes per 256-weight block)
                                   size_t n_blocks = input_size / 144;

                                   // Launch GPU kernel synchronously
                                   if (!launch_q4k_kernel(input_buffer_a.device_ptr,
                                       (float*)output_buffer_a.device_ptr, n_blocks)) {
                                       return false;
                                       }

                                       // Copy result back to host
                                       error = cudaMemcpy(output_chunk, output_buffer_a.device_ptr, output_size,
                                                          cudaMemcpyDeviceToHost);
                                       if (error != cudaSuccess) {
                                           set_error("Failed to copy output from GPU (fallback path)");
                                           return false;
                                       }

                                       return true;
                                                                 }

                                                                 bool GPULoader::launch_q4k_kernel(const void* input, float* output, size_t n_blocks) {
                                                                     // Synchronous kernel launch for fallback path
                                                                     cudaStream_t stream;
                                                                     cudaStreamCreate(&stream);

                                                                     launch_dequantize_q4k_kernel(input, output, (int)n_blocks, stream);

                                                                     cudaStreamSynchronize(stream);
                                                                     cudaStreamDestroy(stream);

                                                                     std::cout << "GPU processed (fallback sync) " << n_blocks << " Q4_K blocks" << std::endl;
                                                                     return true;
                                                                 }

                                                                 bool GPULoader::process_q8_chunk(const void* input_chunk, size_t input_size,
                                                                                                  float* output_chunk, size_t output_size) {
                                                                     // Similar implementation to process_q4k_chunk but for Q8 quantization
                                                                     // Copy input to GPU
                                                                     cudaError_t error = cudaMemcpy(input_buffer_a.device_ptr, input_chunk, input_size,
                                                                                                    cudaMemcpyHostToDevice);
                                                                     if (error != cudaSuccess) {
                                                                         set_error("Failed to copy input to GPU");
                                                                         return false;
                                                                     }

                                                                     // Calculate number of blocks (Q8_0 has 34 bytes per 32-weight block)
                                                                     size_t n_blocks = input_size / 34;

                                                                     // Launch GPU kernel
                                                                     cudaStream_t stream;
                                                                     cudaStreamCreate(&stream);
                                                                     launch_dequantize_q8_kernel(input_buffer_a.device_ptr,
                                                                                                 (float*)output_buffer_a.device_ptr, (int)n_blocks, stream);
                                                                     cudaStreamSynchronize(stream);
                                                                     cudaStreamDestroy(stream);

                                                                     // Copy result back to host
                                                                     error = cudaMemcpy(output_chunk, output_buffer_a.device_ptr, output_size,
                                                                                        cudaMemcpyDeviceToHost);
                                                                     if (error != cudaSuccess) {
                                                                         set_error("Failed to copy output from GPU");
                                                                         return false;
                                                                     }

                                                                     return true;
                                                                                                  }

                                                                                                  bool GPULoader::process_f16_chunk(const void* input_chunk, size_t input_size,
                                                                                                                                    float* output_chunk, size_t output_size) {
                                                                                                      // Similar implementation for F16 data
                                                                                                      // Copy input to GPU
                                                                                                      cudaError_t error = cudaMemcpy(input_buffer_a.device_ptr, input_chunk, input_size,
                                                                                                                                     cudaMemcpyHostToDevice);
                                                                                                      if (error != cudaSuccess) {
                                                                                                          set_error("Failed to copy input to GPU");
                                                                                                          return false;
                                                                                                      }

                                                                                                      // Calculate number of elements (F16 has 2 bytes per element)
                                                                                                      size_t n_elements = input_size / 2;

                                                                                                      // Launch GPU kernel
                                                                                                      cudaStream_t stream;
                                                                                                      cudaStreamCreate(&stream);
                                                                                                      launch_dequantize_f16_kernel(input_buffer_a.device_ptr,
                                                                                                                                   (float*)output_buffer_a.device_ptr, (int)n_elements, stream);
                                                                                                      cudaStreamSynchronize(stream);
                                                                                                      cudaStreamDestroy(stream);

                                                                                                      // Copy result back to host
                                                                                                      error = cudaMemcpy(output_chunk, output_buffer_a.device_ptr, output_size,
                                                                                                                         cudaMemcpyDeviceToHost);
                                                                                                      if (error != cudaSuccess) {
                                                                                                          set_error("Failed to copy output from GPU");
                                                                                                          return false;
                                                                                                      }

                                                                                                      return true;
                                                                                                                                    }

                                                                                                                                    size_t GPULoader::get_optimal_chunk_size() const {
                                                                                                                                        if (chunk_size > 0) {
                                                                                                                                            return std::min(chunk_size, gpu_memory_limit / 4);  // Leave room for input+output buffers
                                                                                                                                        }
                                                                                                                                        return gpu_memory_limit / 4;
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
                                                                                                                                                                             std::cout << "I/O time: " << stats.io_time_seconds << " seconds" << std::endl;
                                                                                                                                                                             std::cout << "GPU time: " << stats.gpu_time_seconds << " seconds" << std::endl;
                                                                                                                                                                             std::cout << "Total time: " << (stats.io_time_seconds + stats.gpu_time_seconds) << " seconds" << std::endl;

                                                                                                                                                                             if (stats.gpu_time_seconds > 0.0) {
                                                                                                                                                                                 std::cout << "Throughput: " << (stats.processed_bytes / 1024.0 / 1024.0) / stats.gpu_time_seconds
                                                                                                                                                                                 << " MB/s" << std::endl;
                                                                                                                                                                             }
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
                                                                                                                                                                                                       cudaError_t error = cudaGetDeviceCount(&device_count);
                                                                                                                                                                                                       if (error != cudaSuccess) {
                                                                                                                                                                                                           std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
                                                                                                                                                                                                           return false;
                                                                                                                                                                                                       }
                                                                                                                                                                                                       return device_count > 0;
                                                                                                                                                                                                   }

                                                                                                                                                                                                   void print_gpu_info() {
                                                                                                                                                                                                       int device_count = 0;
                                                                                                                                                                                                       cudaError_t error = cudaGetDeviceCount(&device_count);
                                                                                                                                                                                                       if (error != cudaSuccess) {
                                                                                                                                                                                                           std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
                                                                                                                                                                                                           return;
                                                                                                                                                                                                       }

                                                                                                                                                                                                       std::cout << "CUDA Devices Found: " << device_count << std::endl;
                                                                                                                                                                                                       for (int i = 0; i < device_count; i++) {
                                                                                                                                                                                                           cudaDeviceProp prop;
                                                                                                                                                                                                           error = cudaGetDeviceProperties(&prop, i);
                                                                                                                                                                                                           if (error != cudaSuccess) {
                                                                                                                                                                                                               std::cerr << "Error getting properties for device " << i << ": "
                                                                                                                                                                                                               << cudaGetErrorString(error) << std::endl;
                                                                                                                                                                                                               continue;
                                                                                                                                                                                                           }

                                                                                                                                                                                                           std::cout << "Device " << i << ": " << prop.name << std::endl;
                                                                                                                                                                                                           std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
                                                                                                                                                                                                           std::cout << "  Total Memory: " << prop.totalGlobalMem / 1024 / 1024 << " MB" << std::endl;
                                                                                                                                                                                                           std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
                                                                                                                                                                                                       }
                                                                                                                                                                                                   }
