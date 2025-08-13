// GGUF File Format Reader - Header
// Parses GGUF files and feeds chunks to GPU kernels

#ifndef GGUF_READER_H
#define GGUF_READER_H

#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <unordered_map>

// GGUF format constants
#define GGUF_MAGIC 0x46554747  // "GGUF" in little-endian
#define GGUF_VERSION 3

// GGUF data types
enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT,
};

// GGML tensor types (quantization formats)
enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,  // Our main target!
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_COUNT,
};

// GGUF header structure
struct gguf_header {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
};

// Tensor information
struct gguf_tensor_info {
    std::string name;
    uint32_t n_dimensions;
    std::vector<uint64_t> dimensions;
    uint32_t type;  // ggml_type
    uint64_t offset;
    uint64_t size;
};

// Metadata key-value pair
struct gguf_kv {
    std::string key;
    uint32_t type;  // gguf_type
    std::vector<uint8_t> value;
};

// Main GGUF reader class
class GGUFReader {
public:
    GGUFReader();
    ~GGUFReader();
    
    // Core functionality
    bool open(const std::string& filename);
    void close();
    
    // Header and metadata
    bool read_header();
    bool read_metadata();
    bool read_tensor_info();
    
    // Tensor data access
    size_t get_tensor_count() const { return tensors.size(); }
    const gguf_tensor_info& get_tensor(size_t index) const { return tensors[index]; }
    
    // Chunked reading for GPU processing
    bool read_tensor_chunk(size_t tensor_index, uint64_t offset, size_t chunk_size, void* buffer);
    bool is_quantized_tensor(size_t tensor_index) const;
    
    // Utility functions
    std::string get_metadata_string(const std::string& key) const;
    int64_t get_metadata_int(const std::string& key) const;
    float get_metadata_float(const std::string& key) const;
    
    // File info
    uint64_t get_file_size() const { return file_size; }
    size_t get_tensor_data_offset() const { return tensor_data_offset; }
    
    // Error handling
    const std::string& get_last_error() const { return last_error; }

private:
    FILE* file;
    int fd;  // Add file descriptor for direct I/O
    uint64_t file_size;
    size_t tensor_data_offset;
    
    gguf_header header;
    std::vector<gguf_kv> metadata;
    std::vector<gguf_tensor_info> tensors;
    std::unordered_map<std::string, size_t> metadata_map;
    
    std::string last_error;
    
    // Helper functions
    bool read_string(std::string& str);
    bool read_value(uint32_t type, std::vector<uint8_t>& value);
    size_t get_type_size(uint32_t type) const;
    void set_error(const std::string& error);
    
    // Disable copy
    GGUFReader(const GGUFReader&) = delete;
    GGUFReader& operator=(const GGUFReader&) = delete;
};

// Utility functions
const char* ggml_type_name(uint32_t type);
size_t ggml_type_size(uint32_t type);
bool ggml_type_is_quantized(uint32_t type);

#endif // GGUF_READER_H