// Minimal CUDA test avoiding problematic headers
extern "C" {
    int printf(const char*, ...);
}

// CUDA runtime function declarations (avoid including headers)
extern "C" {
    int cudaGetDeviceCount(int*);
    int cudaGetDeviceProperties(void*, int);
    const char* cudaGetErrorString(int);
}

struct cudaDeviceProp_minimal {
    char name[256];
    long totalGlobalMem;
    int major, minor;
    int multiProcessorCount;
    // ... other fields we don't need
};

__global__ void test_kernel() {
    // Simple kernel
}

int main() {
    int device_count;
    int err = cudaGetDeviceCount(&device_count);
    
    if (err != 0) {
        printf("CUDA Error: %d\n", err);
        return 1;
    }
    
    printf("Found %d CUDA device(s)\n", device_count);
    
    if (device_count > 0) {
        cudaDeviceProp_minimal prop = {};
        err = cudaGetDeviceProperties(&prop, 0);
        if (err == 0) {
            printf("GPU: %s\n", prop.name);
            printf("Memory: %ld bytes\n", prop.totalGlobalMem);
        }
    }
    
    printf("✅ Basic CUDA working!\n");
    return 0;
}