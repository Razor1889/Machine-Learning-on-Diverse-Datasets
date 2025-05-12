import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from time import perf_counter

if __name__ == '__main__':
    dim = []
    speeds = []
    mem_used = []
    n = 0
    MAX_VALUE = 10000
    while n < MAX_VALUE:
        
        dim.append(n)
        X_DIMENSION = n
        size = (int(X_DIMENSION)), int(X_DIMENSION)
        input_cpu = torch.randn(size).cpu()
        input_gpu = input_cpu.cuda()  # Move tensor to GPU

        def cpu_computation(input):
            start_time = perf_counter()
            result = torch.matmul(input, input)
            end_time = perf_counter()
            return result, end_time - start_time

        def gpu_computation(input):
            # torch.cuda.memory.empty_cache()
            # print(torch.cuda.memory.memory_reserved() / (1024 ** 2))
            start_time = perf_counter()
            result = torch.matmul(input, input)
            torch.cuda.synchronize()
            end_time = perf_counter()
            return result, end_time - start_time, torch.cuda.memory_allocated() / (1024 ** 2)

        cpu_result, cpu_time = cpu_computation(input_cpu)
        gpu_result, gpu_time, mem_used_MB = gpu_computation(input_gpu)

        speedup = cpu_time / gpu_time
        mem_used.append(mem_used_MB)
        speeds.append(speedup)
        print(f"Speedup (CPU / GPU): {speedup}")
        ratio = n / MAX_VALUE
        if ratio < 0.1:
            n += 10
        elif ratio < 0.2:
            n += 25
        elif ratio < 0.3:
            n += 50
        elif ratio < 0.4:
            n += 75
        elif ratio < 0.5:
            n += 150
        elif ratio < 0.6:
            n += 250
        elif ratio < 0.7:
            n += 350
        else:
            n += 500

    poly_degree = 15
    coefficients = np.polyfit(dim, speeds, poly_degree)
    trend_line = np.poly1d(coefficients)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(dim, speeds, marker='o', linestyle='', label='GPU Speedup factor')
    plt.plot(dim, trend_line(dim), linestyle='-', label='Trend Line')
    plt.title('GPU Speedup vs. Matrix Dimensions')
    plt.xlabel('Matrix Dimension (n)')
    plt.ylabel('Speedup factor (CPU / GPU)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(dim, mem_used, marker='o', linestyle='', color='r', label='Memory Used (MB)')
    plt.title('Memory Usage vs. Matrix Dimensions')
    plt.xlabel('Matrix Dimension (n)')
    plt.ylabel('Memory Used (MB)')
    poly_degree = 2
    coefficients = np.polyfit(dim, mem_used, poly_degree)
    trend_line = np.poly1d(coefficients)
    plt.plot(dim, trend_line(dim), linestyle='-', label=None)
    
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
