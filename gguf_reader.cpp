// GGUF File Format Reader - Implementation
// Parses GGUF files and feeds chunks to GPU kernels

#include "gguf_reader.h"
#include <cstring>
#include <iostream>
#include <sys/stat.h>

// Type size lookup table
static const size_t ggml_type_sizes[] = {
    [GGML_TYPE_F32]  = sizeof(float),
    [GGML_TYPE_F16]  = sizeof(uint16_t),
    [GGML_TYPE_Q4_0] = 18,  // 16 weights + 2 bytes metadata per block
    [GGML_TYPE_Q4_1] = 20,  // 16 weights + 4 bytes metadata per block
    [GGML_TYPE_Q5_0] = 22,  // Complex structure
    [GGML_TYPE_Q5_1] = 24,  // Complex structure
    [GGML_TYPE_Q8_0] = 34,  // 32 weights + 2 bytes metadata per block
    [GGML_TYPE_Q8_1] = 36,  // 32 weights + 4 bytes metadata per block
    [GGML_TYPE_Q2_K] = 84,  // K-quant super-block
    [GGML_TYPE_Q3_K] = 110, // K-quant super-block
    [GGML_TYPE_Q4_K] = 144, // K-quant super-block (our target!)
    [GGML_TYPE_Q5_K] = 176, // K-quant super-block
    [GGML_TYPE_Q6_K] = 210, // K-quant super-block
    [GGML_TYPE_Q8_K] = 292, // K-quant super-block
};

static const char* ggml_type_names[] = {
    [GGML_TYPE_F32]  = "F32",
    [GGML_TYPE_F16]  = "F16", 
    [GGML_TYPE_Q4_0] = "Q4_0",
    [GGML_TYPE_Q4_1] = "Q4_1",
    [GGML_TYPE_Q5_0] = "Q5_0",
    [GGML_TYPE_Q5_1] = "Q5_1",
    [GGML_TYPE_Q8_0] = "Q8_0",
    [GGML_TYPE_Q8_1] = "Q8_1",
    [GGML_TYPE_Q2_K] = "Q2_K",
    [GGML_TYPE_Q3_K] = "Q3_K",
    [GGML_TYPE_Q4_K] = "Q4_K",
    [GGML_TYPE_Q5_K] = "Q5_K",
    [GGML_TYPE_Q6_K] = "Q6_K",
    [GGML_TYPE_Q8_K] = "Q8_K",
};

GGUFReader::GGUFReader() : file(nullptr), file_size(0), tensor_data_offset(0) {
    memset(&header, 0, sizeof(header));
}

GGUFReader::~GGUFReader() {
    close();
}

bool GGUFReader::open(const std::string& filename) {
    close();
    
    file = fopen(filename.c_str(), "rb");
    if (!file) {
        set_error("Failed to open file: " + filename);
        return false;
    }
    
    // Get file size
    struct stat st;
    if (stat(filename.c_str(), &st) != 0) {
        set_error("Failed to get file size");
        close();
        return false;
    }
    file_size = st.st_size;
    
    std::cout << "Opened GGUF file: " << filename << " (" << file_size / 1024 / 1024 << " MB)" << std::endl;
    return true;
}

void GGUFReader::close() {
    if (file) {
        fclose(file);
        file = nullptr;
    }
    file_size = 0;
    tensor_data_offset = 0;
    metadata.clear();
    tensors.clear();
    metadata_map.clear();
    last_error.clear();
}

bool GGUFReader::read_header() {
    if (!file) {
        set_error("File not open");
        return false;
    }
    
    fseek(file, 0, SEEK_SET);
    
    if (fread(&header, sizeof(header), 1, file) != 1) {
        set_error("Failed to read header");
        return false;
    }
    
    // Verify magic number
    if (header.magic != GGUF_MAGIC) {
        set_error("Invalid GGUF magic number");
        return false;
    }
    
    // Check version
    if (header.version != GGUF_VERSION) {
        set_error("Unsupported GGUF version: " + std::to_string(header.version));
        return false;
    }
    
    std::cout << "GGUF Header:" << std::endl;
    std::cout << "  Version: " << header.version << std::endl;
    std::cout << "  Tensors: " << header.tensor_count << std::endl;
    std::cout << "  Metadata KV pairs: " << header.metadata_kv_count << std::endl;
    
    return true;
}

bool GGUFReader::read_metadata() {
    metadata.clear();
    metadata_map.clear();
    
    for (uint64_t i = 0; i < header.metadata_kv_count; i++) {
        gguf_kv kv;
        
        // Read key string
        if (!read_string(kv.key)) {
            set_error("Failed to read metadata key");
            return false;
        }
        
        // Read value type
        if (fread(&kv.type, sizeof(uint32_t), 1, file) != 1) {
            set_error("Failed to read metadata value type");
            return false;
        }
        
        // Read value data
        if (!read_value(kv.type, kv.value)) {
            set_error("Failed to read metadata value");
            return false;
        }
        
        metadata_map[kv.key] = metadata.size();
        metadata.push_back(std::move(kv));
    }
    
    std::cout << "Read " << metadata.size() << " metadata entries" << std::endl;
    return true;
}

bool GGUFReader::read_tensor_info() {
    tensors.clear();
    
    for (uint64_t i = 0; i < header.tensor_count; i++) {
        gguf_tensor_info tensor;
        
        // Read tensor name
        if (!read_string(tensor.name)) {
            set_error("Failed to read tensor name");
            return false;
        }
        
        // Read number of dimensions
        if (fread(&tensor.n_dimensions, sizeof(uint32_t), 1, file) != 1) {
            set_error("Failed to read tensor dimensions count");
            return false;
        }
        
        // Read dimensions
        tensor.dimensions.resize(tensor.n_dimensions);
        for (uint32_t j = 0; j < tensor.n_dimensions; j++) {
            if (fread(&tensor.dimensions[j], sizeof(uint64_t), 1, file) != 1) {
                set_error("Failed to read tensor dimension");
                return false;
            }
        }
        
        // Read tensor type
        if (fread(&tensor.type, sizeof(uint32_t), 1, file) != 1) {
            set_error("Failed to read tensor type");
            return false;
        }
        
        // Read tensor offset
        if (fread(&tensor.offset, sizeof(uint64_t), 1, file) != 1) {
            set_error("Failed to read tensor offset");
            return false;
        }
        
        // Calculate tensor size
        uint64_t element_count = 1;
        for (uint64_t dim : tensor.dimensions) {
            element_count *= dim;
        }
        
        if (tensor.type < GGML_TYPE_COUNT) {
            tensor.size = element_count * ggml_type_sizes[tensor.type] / 256;  // Blocks of 256
            if (tensor.type == GGML_TYPE_F32 || tensor.type == GGML_TYPE_F16) {
                tensor.size = element_count * ggml_type_sizes[tensor.type];
            }
        } else {
            set_error("Unknown tensor type: " + std::to_string(tensor.type));
            return false;
        }
        
        tensors.push_back(std::move(tensor));
    }
    
    // Calculate tensor data offset (aligned to 32 bytes)
    tensor_data_offset = ftell(file);
    tensor_data_offset = (tensor_data_offset + 31) & ~31;
    
    std::cout << "Read " << tensors.size() << " tensor definitions" << std::endl;
    std::cout << "Tensor data starts at offset: " << tensor_data_offset << std::endl;
    
    // Print tensor summary
    size_t q4k_tensors = 0, total_q4k_size = 0;
    for (const auto& tensor : tensors) {
        if (tensor.type == GGML_TYPE_Q4_K) {
            q4k_tensors++;
            total_q4k_size += tensor.size;
        }
    }
    
    std::cout << "Found " << q4k_tensors << " Q4_K tensors (" 
              << total_q4k_size / 1024 / 1024 << " MB)" << std::endl;
    
    return true;
}

bool GGUFReader::read_tensor_chunk(size_t tensor_index, uint64_t offset, size_t chunk_size, void* buffer) {
    if (!file || tensor_index >= tensors.size()) {
        set_error("Invalid tensor index or file not open");
        return false;
    }
    
    const auto& tensor = tensors[tensor_index];
    uint64_t file_offset = tensor_data_offset + tensor.offset + offset;
    
    // Bounds checking
    if (offset + chunk_size > tensor.size) {
        set_error("Chunk extends beyond tensor bounds");
        return false;
    }
    
    if (fseek(file, file_offset, SEEK_SET) != 0) {
        set_error("Failed to seek to tensor data");
        return false;
    }
    
    size_t bytes_read = fread(buffer, 1, chunk_size, file);
    if (bytes_read != chunk_size) {
        set_error("Failed to read tensor chunk");
        return false;
    }
    
    return true;
}

bool GGUFReader::is_quantized_tensor(size_t tensor_index) const {
    if (tensor_index >= tensors.size()) return false;
    return ggml_type_is_quantized(tensors[tensor_index].type);
}

bool GGUFReader::read_string(std::string& str) {
    uint64_t len;
    if (fread(&len, sizeof(uint64_t), 1, file) != 1) {
        return false;
    }
    
    if (len > 1024 * 1024) {  // Sanity check: 1MB max string
        return false;
    }
    
    str.resize(len);
    if (len > 0 && fread(&str[0], 1, len, file) != len) {
        return false;
    }
    
    return true;
}

bool GGUFReader::read_value(uint32_t type, std::vector<uint8_t>& value) {
    size_t size = get_type_size(type);
    if (size == 0) return false;
    
    value.resize(size);
    return fread(value.data(), 1, size, file) == size;
}

size_t GGUFReader::get_type_size(uint32_t type) const {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            return 8;
        case GGUF_TYPE_STRING:
        case GGUF_TYPE_ARRAY:
            // Variable length - need special handling
            return 0;
        default:
            return 0;
    }
}

void GGUFReader::set_error(const std::string& error) {
    last_error = error;
    std::cerr << "GGUF Error: " << error << std::endl;
}

// Utility functions
const char* ggml_type_name(uint32_t type) {
    if (type < GGML_TYPE_COUNT) {
        return ggml_type_names[type];
    }
    return "UNKNOWN";
}

size_t ggml_type_size(uint32_t type) {
    if (type < GGML_TYPE_COUNT) {
        return ggml_type_sizes[type];
    }
    return 0;
}

bool ggml_type_is_quantized(uint32_t type) {
    return type >= GGML_TYPE_Q4_0 && type <= GGML_TYPE_Q8_K;
}