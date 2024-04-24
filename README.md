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

CUDA cores vs CPU cores

vector_add.cu
nvcc -o vector_add vector_add.cu
nvcc -ptx vector_add.cu -o vector_add.ptx
vector_add.ptx
nvcc -c vector_add.cu -o vector_add.o
vector_add.o
nvcc -cubin vector_add.cu -o vector_add.cubin

vector_add.cubin

nvcc automatically recognizes the CUDA kernel functions in the source file (vector_add.cu) and compiles them to run on the GPU. Any other code outside of kernel functions will run on the CPU by default.

To determine which parts of the code are executed on the GPU and which parts are executed on the CPU, you can look for CUDA kernel function definitions (functions declared with __global__ specifier) in the source file. These functions will be compiled to run on the GPU. Any other code in the file will run on the CPU.

how I can see that there is process runing on GPU like ps we use for normal process running on the cpu
nvidia-smi

When scheduling tasks to the GPU, several system and shared libraries are involved in the process. 
CUDA Runtime (libcudart):  for managing GPU memory, launching CUDA kernels, and synchronizing between the CPU and GPU.
CUDA Driver (libcuda): It allows applications to interact directly with the GPU, bypassing the CUDA runtime API. This library is used by CUDA-enabled applications to initialize the GPU, allocate memory, and launch kernels.
CUDA Compiler (nvcc): It translates CUDA kernel functions written in CUDA C/C++ into machine code that can be executed on the GPU.
GPU Driver - The GPU driver acts as a bridge between the operating system and the GPU hardware.
GPU Runtime Libraries (e.g., cuBLAS, cuDNN): Examples include cuBLAS for linear algebra operations and cuDNN for deep learning tasks


///////////////////////////////////////////////////////printing time all details

#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel to calculate factorial of a number
__global__ void factorialKernel(int *result, int n, float *executionTimes) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Start timing
    clock_t start_time = clock();
    
    // Calculate factorial
    int fact = 1;
    for (int i = 2; i <= tid + 1; ++i) {
        fact *= i;
    }
    
    // End timing
    clock_t end_time = clock();
    
    // Store execution time
    executionTimes[tid] = (float)(end_time - start_time) / CLOCKS_PER_SEC;
    
    // Print thread and block information along with factorial value and execution time
    printf("Thread ID: %d, Block ID: %d, Block Dim: %d, Factorial of %d: %d, Execution Time: %f seconds\n",
           threadIdx.x, blockIdx.x, blockDim.x, tid + 1, fact, executionTimes[tid]);
    
    if (tid < n) {
        result[tid] = fact;
    }
}

int main() {
    const int N = 100;
    int *h_result = new int[N];
    int *d_result;
    float *d_executionTimes;
    float *h_executionTimes = new float[N];

    // Start timing for main function
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Allocate memory on device for result and execution times
    cudaMalloc((void**)&d_result, N * sizeof(int));
    cudaMalloc((void**)&d_executionTimes, N * sizeof(float));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    factorialKernel<<<gridSize, blockSize>>>(d_result, N, d_executionTimes);

    // Copy result and execution times from device to host
    cudaMemcpy(h_result, d_result, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_executionTimes, d_executionTimes, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print factorial results
    for (int i = 0; i < N; ++i) {
        std::cout << "Factorial of " << i+1 << ": " << h_result[i] << ", Execution Time: " << h_executionTimes[i] << " seconds\n";
    }

    // Free memory
    cudaFree(d_result);
    cudaFree(d_executionTimes);
    delete[] h_result;
    delete[] h_executionTimes;

    // End timing for main function
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Total time taken for all operations in main function: " << milliseconds << " milliseconds\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
////////////////////////////////////////// Sequential on the CPU////////////
#include <iostream>
#include <chrono>

// Function to calculate factorial of a number
int factorial(int n) {
    int fact = 1;
    for (int i = 2; i <= n; ++i) {
        fact *= i;
    }
    return fact;
}

int main() {
    const int N = 100;
    int *h_result = new int[N];

    // Start timing for main function
    auto start = std::chrono::high_resolution_clock::now();

    // Calculate factorial of numbers from 1 to N on CPU
    for (int i = 1; i <= N; ++i) {
        h_result[i-1] = factorial(i);
    }

    // End timing for main function
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Print factorial results and execution times
    for (int i = 0; i < N; ++i) {
        std::cout << "Factorial of " << i+1 << ": " << h_result[i] << std::endl;
    }

    // Free memory
    delete[] h_result;

    // Print total time taken for all operations in main function
    std::cout << "Total time taken for all operations in main function: " << duration.count() * 1000 << " milliseconds" << std::endl;

    return 0;
}
////////////////////////////


////////////////////parallel on the cpu
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>

// Function to calculate factorial of a number
int factorial(int n) {
    int fact = 1;
    for (int i = 2; i <= n; ++i) {
        fact *= i;
    }
    return fact;
}

int main() {
    const int N = 100;
    std::vector<int> h_result(N);

    // Start timing for main function
    auto start = std::chrono::high_resolution_clock::now();

    // Define the number of threads to be used
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 4; // Set a default number of threads if hardware_concurrency returns 0
    }

    // Define a vector to store threads
    std::vector<std::thread> threads;

    // Define a lambda function to calculate factorial in parallel
    auto calculate_factorial = [&](int start_idx, int end_idx) {
        for (int i = start_idx; i < end_idx; ++i) {
            h_result[i] = factorial(i + 1);
        }
    };

    // Launch threads
    for (int i = 0; i < num_threads; ++i) {
        int start_idx = (i * N) / num_threads;
        int end_idx = ((i + 1) * N) / num_threads;
        threads.emplace_back(calculate_factorial, start_idx, end_idx);
    }

    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }

    // End timing for main function
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Print factorial results and execution times
    for (int i = 0; i < N; ++i) {
        std::cout << "Factorial of " << i + 1 << ": " << h_result[i] << std::endl;
    }

    // Print total time taken for all operations in main function
    std::cout << "Total time taken for all operations in main function: " << duration.count() * 1000 << " milliseconds" << std::endl;

    return 0;
}
//////////////////////////////////////






`
