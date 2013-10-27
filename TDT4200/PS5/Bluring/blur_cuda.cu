#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

//#include "bmp.h"
extern "C" void write_bmp(unsigned char* data, int width, int height);
extern "C" unsigned char* read_bmp(char* filename);

//#include "host_blur.h"
extern "C" void host_blur(unsigned char* inputImage, unsigned char* outputImage, int size);

void print_properties() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    printf("Device count: %d\n", deviceCount);

    cudaDeviceProp p;
    cudaSetDevice(0);
    cudaGetDeviceProperties (&p, 0);

    printf("Compute capability: %d.%d\n", p.major, p.minor);
    printf("Name: %s\n" , p.name);
    printf("\n\n");
}
__global__ void device_blur(/*Put arguments here*/) {
    //Put you device code here
}
int main(int argc,char **argv) {

    // Prints some device properties, also to make sure the GPU works etc.
    print_properties();

    unsigned char* A = read_bmp("peppers.bmp");
    unsigned char* B = (unsigned char*)malloc(sizeof(unsigned char) * 512 * 512);
    
    //Currently we do the bluring on the CPU
    host_blur(A, B, 512);

    // You need to:
    // 1. Allocate buffers for the input image and the output image

    // 2. Transfer the input image from the host to the device

    // 3. Launch the kernel which does the bluring
    //device_blur<<< >>>();

    // 4. Transfer the result back to the host.
    write_bmp(B, 512, 512);

    free(A);
    free(B);

    return 0;
}
