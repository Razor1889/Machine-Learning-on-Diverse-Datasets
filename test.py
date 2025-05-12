import torch
import master
import torch.cuda as cuda
import psutil

# Check if CUDA is available
print(f"torch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"Number of CUDA devices available: {torch.cuda.device_count()}")
    # Print the name of the current CUDA device
    
    print("\nGPU DEVICE INFO:\n")
    gpu_index = cuda.current_device()
    print(f"Current CUDA device: {torch.cuda.get_device_name(gpu_index)}")
    print(f"GPU properties: {torch.cuda.get_device_properties(gpu_index)}")
    
    print(f"memory usage (percent): {cuda.memory_usage(gpu_index)}")
    print(f"utilization (percent): {cuda.utilization(gpu_index)}")
    print(f"temperature (Degrees Centigrade): {cuda.temperature(gpu_index)}")
    print(f"average power draw (milliWatts): {cuda.power_draw(gpu_index)}")
    print(f"clock rate (Hz): {cuda.clock_rate(gpu_index)}")


    # info = psutil.Process().memory_info()
    # for item in info:
    #     print(item)
    #     memory_usage =  item / 1024 / 1024  # in MB
    
    # print(f"modules from torch:")
    # print(*(f"{x}" if i % 5 != 0 else f"{x}\n" for i, x in enumerate(dir(torch))))
    
else:
    print("CUDA is not available. Go install CUDA.")