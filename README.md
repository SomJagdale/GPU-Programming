Concepts 

GPU architecture 

![image](https://github.com/SomJagdale/GPU-Programming/assets/97079268/30788d4f-d9c6-42c4-88d8-ae683f603ee8)

Vector Processors

Multi processors 

Cluster 

Shader program

Compute Shader 

Vertext Shader

Pixel Shader 

Gemetry Shader 

SIMT vs SIMD

wraps consist typically consist 32 threads

SFU

ISA

Register Files

ALU

SIMD Processor

SM - Streaming Multiprocesossors

SP - Streaming Processors 

CUDO cores 

![image](https://github.com/SomJagdale/GPU-Programming/assets/97079268/f0b0770b-7bde-47bc-8430-a438fe64c3d2)


Memory Hiearchy 

![image](https://github.com/SomJagdale/GPU-Programming/assets/97079268/a13bbe51-dba7-4a08-9d6a-fa89c3da486b)


![image](https://github.com/SomJagdale/GPU-Programming/assets/97079268/62995fd4-08b5-411e-892b-9c08c610188a)


![image](https://github.com/SomJagdale/GPU-Programming/assets/97079268/d3177390-4552-4959-915a-9c29e68158a9)


PTX

VDA vs GPU
VDA - basically integrated into CPU and share system RAM for graphics processing 
integrated graphics processing units (GPUs), Zeor copy is employed

![image](https://github.com/SomJagdale/GPU-Programming/assets/97079268/72a03eeb-7783-4551-97b6-dacc5925e1b5)

![image](https://github.com/SomJagdale/GPU-Programming/assets/97079268/508dd235-2ff7-4df0-b065-3ae08ae93e98)

DDR

![image](https://github.com/SomJagdale/GPU-Programming/assets/97079268/93418cdb-2a21-4230-847c-0f341b80d448)


ARM 

PCI

GDDR

HDMI

GPU has seprate RAM

MMU

DisplayPort


GPU Programming Concepts
Parallal processing used primariliy - Graphics Rendering, Scientific Simulations, Machine Learning, Chriptocurrancy Mining, Gaming, Virtual Reality
					Augmented Reality, Image processing, Video trascending 
CUDA for Nvidia
OpenCL for other vendors
Global Memory
Shared Memory 
Constat Memory
Language to write the code for GPU
CUDO - Common Unified Device Architecture

Requirements for CUDA Developement 
CUDA Toolkits
 1. CUDA Compilers - nvcc
 2. CUDA Libraries 
 3. Developement Tools

GPU Configuration
NVidia Maxwell Architecture Overview
128 NVIDIA CUDA cores

CUDA Toolkit
NVIDIA GPU Drivers

vector_add.cu
nvcc -o vector_add vector_add.cu
nvcc -ptx vector_add.cu -o vector_add.ptx
vector_add.ptx
nvcc -c vector_add.cu -o vector_add.o
vector_add.o
nvcc -cubin vector_add.cu -o vector_add.cubin

vector_add.cubin

#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel to add two vectors on the GPU
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    // Vector size
    int n = 1000;

    // Allocate memory on the host
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];

    // Initialize input vectors
    for (int i = 0; i < n; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate memory on the device
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_b, n * sizeof(float));
    cudaMalloc((void**)&d_c, n * sizeof(float));

    // Copy input vectors from host to device
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define block size and grid size
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch the kernel on the GPU
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // Copy result vector from device to host
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result (optional)
    for (int i = 0; i < 10; ++i) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}



