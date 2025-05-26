#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <random>

const int N = 1000000;
const float k = 2.5f;

void scaleVectorCPU(const float* A, float* B, int N, float k) {
    #pragma omp simd
    for(int i = 0; i < N; i++) {
        B[i] = A[i] * k;
    }
}

__global__ void scaleVectorCUDA(const float* __restrict__ A, 
                               float* __restrict__ B, 
                               int N, float k) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        B[idx] = A[idx] * k;
    }
}

int main() {
    // Выделяем память
    float *A, *B_cpu, *B_gpu;
    cudaMallocHost(&A, N * sizeof(float));
    cudaMallocHost(&B_cpu, N * sizeof(float));
    cudaMallocHost(&B_gpu, N * sizeof(float));

    // Инициализация данных
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);
    for(int i = 0; i < N; i++) {
        A[i] = dis(gen);
    }

    // CPU вычисления
    auto cpu_start = std::chrono::high_resolution_clock::now();
    scaleVectorCPU(A, B_cpu, N, k);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(cpu_end - cpu_start).count();

    // GPU вычисления
    float *d_A, *d_B;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));

    // Настройка параметров запуска
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    cudaStream_t stream;
    cudaEvent_t start, stop;
    cudaStreamCreate(&stream);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Полный цикл GPU с копированием
    cudaEventRecord(start);
    cudaMemcpyAsync(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice, stream);
    scaleVectorCUDA<<<gridSize, blockSize, 0, stream>>>(d_A, d_B, N, k);
    cudaMemcpyAsync(B_gpu, d_B, N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    // Вывод результатов
    std::cout << "=== Performance Comparison ===" << std::endl;
    std::cout << "CPU execution time: " << cpu_time * 1000 << " ms" << std::endl;
    std::cout << "GPU execution time: " << gpu_time_ms << " ms" << std::endl;
    std::cout << "GPU speedup: " << (cpu_time * 1000) / gpu_time_ms << "x" << std::endl;

    // Освобождение ресурсов
    cudaFreeHost(A);
    cudaFreeHost(B_cpu);
    cudaFreeHost(B_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
