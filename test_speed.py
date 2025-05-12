import sys, torch
from time import perf_counter

def main():
    X_DIMENSION : float = None
    if len(sys.argv) > 1:
        try:
            X_DIMENSION = int(sys.argv[1])
        except ValueError:
            print("invalid command line argument (must be int)")
            sys.exit(-1)

    else:
        X_DIMENSION : float = 1e3
    Y_DIMENSION : float = X_DIMENSION # matrixes need to be square to multiply
    size : tuple[int]= (int(X_DIMENSION)), int(Y_DIMENSION)
    print(f"\nTesting matrix multiplication for dimensions: {size[0]:,} by {size[1]:,}")
    # print(f"Number of operations {X_DIMENSION ** 3:.1e}")
    # Create random input tensors for CPU and GPU
    input_cpu = torch.randn(size).cpu()
    input_gpu = input_cpu.cuda()  # Move tensor to GPU

    # Define a function to perform the computation using CPU
    def cpu_computation(input):
        start_time = perf_counter()
        result = torch.matmul(input, input)
        end_time = perf_counter()
        return result, end_time - start_time

    # Define a function to perform the computation using GPU
    def gpu_computation(input):
        start_time = perf_counter()
        # torch.cuda.memory._record_memory_history()
        result = torch.matmul(input, input)
        # Ensure synchronization to measure complete execution time
        torch.cuda.synchronize()
        # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
        end_time = perf_counter()
        # print(f"GPU Processes: {torch.cuda.memory.list_gpu_processes()}")
        # print(f"GPU Processes: {torch.cuda.memory.memory_summary()}")
        
        return result, end_time - start_time
    # Perform computation using CPU
    cpu_result, cpu_time = cpu_computation(input_cpu)
    # Perform computation using GPU
    gpu_result, gpu_time = gpu_computation(input_gpu)
    # Print results
    print(f"d type: {cpu_result.dtype}. Size of {cpu_result.dtype}: {cpu_result.dtype.itemsize}")
    print(f"CPU Execution Time: {cpu_time:.4f}")
    print(f"GPU Execution Time: {gpu_time:.4f}")
    speedup = cpu_time / gpu_time
    print(f"Speedup (CPU / GPU): {speedup}")
    
    
    print(f"Max memory allocated on GPU (MB): {torch.cuda.memory.max_memory_allocated() / (1024 ** 2)}")
    # Verify correctness by comparing results
    # print("Results Match:", torch.allclose(cpu_result, gpu_result.cpu()))  # Move GPU result back to CPU for comparison

if __name__ == '__main__':
    main()