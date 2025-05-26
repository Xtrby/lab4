#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <random>

const int WIDTH = 1024;
const int HEIGHT = 1024;
const unsigned char THRESHOLD = 128;

void thresholdFilterCPU(unsigned char* input, unsigned char* output, int width, int height, unsigned char threshold) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            output[idx] = (input[idx] > threshold) ? 255 : 0;
        }
    }
}

// GPU ядро
__global__ void thresholdFilterCUDA(unsigned char* input, unsigned char* output, int width, int height, unsigned char threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = (input[idx] > threshold) ? 255 : 0;
    }
}

int main() {
    const int imageSize = WIDTH * HEIGHT * sizeof(unsigned char);

    // Выделение памяти и инициализация
    unsigned char* h_input = new unsigned char[WIDTH * HEIGHT];
    unsigned char* h_output_cpu = new unsigned char[WIDTH * HEIGHT];
    unsigned char* h_output_gpu = new unsigned char[WIDTH * HEIGHT];

    // Генерация случайных данных
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_input[i] = static_cast<unsigned char>(dis(gen));
    }

    // CPU обработка
    auto cpu_start = std::chrono::high_resolution_clock::now();
    thresholdFilterCPU(h_input, h_output_cpu, WIDTH, HEIGHT, THRESHOLD);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;

    // GPU обработка
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);

    // Копирование данных на GPU
    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);

    // Настройка параметров запуска
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, 
                 (HEIGHT + blockSize.y - 1) / blockSize.y);

    // Замер времени GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    thresholdFilterCUDA<<<gridSize, blockSize>>>(d_input, d_output, WIDTH, HEIGHT, THRESHOLD);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    // Копирование результатов обратно
    cudaMemcpy(h_output_gpu, d_output, imageSize, cudaMemcpyDeviceToHost);

    // Вывод результатов
    std::cout << "=== Performance Results ===" << std::endl;
    std::cout << "CPU execution time: " << cpu_duration.count() * 1000 << " ms" << std::endl;
    std::cout << "GPU execution time: " << gpu_ms << " ms" << std::endl;
    std::cout << "GPU speedup: " << (cpu_duration.count() * 1000) / gpu_ms << "x" << std::endl;

    // Освобождение памяти
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
