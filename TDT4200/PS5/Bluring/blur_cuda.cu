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

__global__ void device_blur(unsigned char *input_img, unsigned char *output_img) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + y * 512; 

    output_img[index] = 0;
    for(int k = -1; k < 2; k++) {
        for(int l = -1; l < 2; l++) {
            // TOOD: Add 1px border to input_img and fix index
            output_img[index] += (input_img[index + k + l] / 9.0);
        }
    }
}

int main(int argc,char **argv) {
    // Prints some device properties, also to make sure the GPU works etc.
    print_properties();

    //Currently we do the bluring on the CPU
    unsigned char *A = read_bmp("peppers.bmp");
    unsigned char *B = (unsigned char *) malloc(sizeof(unsigned char) * 512 * 512);

    dim3 numBlocks, threadsPerBlock;

    numBlocks.x = 64; numBlocks.y = 64; // 4096 blocks
    threadsPerBlock.x = 8; threadsPerBlock.y = 8; // 64 threads per block

    // 1. Allocate buffers for the input image and the output image
    unsigned char *input_img;
    cudaMalloc((void**) &input_img, sizeof(unsigned char) * 512 * 512);
    printf("1: %s \n", cudaGetErrorString(cudaGetLastError()));

    unsigned char *output_img;
    cudaMalloc((void**) &output_img, sizeof(unsigned char) * 512 * 512);
    printf("2: %s \n", cudaGetErrorString(cudaGetLastError()));

    // 2. Transfer the input image from the host to the device
    cudaMemcpy(input_img, A, sizeof(unsigned char) * 512 * 512, cudaMemcpyHostToDevice);
    printf("3: %s \n", cudaGetErrorString(cudaGetLastError()));

    // 3. Launch the kernel which does the bluring
    device_blur<<<numBlocks, threadsPerBlock>>>(input_img, output_img);
    printf("4: %s \n", cudaGetErrorString(cudaGetLastError()));

    // 4. Transfer the result back to the host.
    cudaMemcpy(B, output_img, sizeof(unsigned char) * 512 * 512, cudaMemcpyDeviceToHost);
    printf("5: %s \n", cudaGetErrorString(cudaGetLastError()));

    write_bmp(B, 512, 512);

    free(A);
    free(B);

    return 0;
}
