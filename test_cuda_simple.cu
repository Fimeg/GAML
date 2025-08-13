#include <stdio.h>
#include <cuda_runtime.h>
int main() {
  int count;
  cudaGetDeviceCount(&count);
  printf("CUDA devices: %d\n", count);
  return 0;
}
