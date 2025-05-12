import matplotlib.pyplot as plt
import master
from subprocess import run, CalledProcessError
from time import perf_counter

times_list_categorical_cpu, times_list_categorical_cuda, times_list_normal_cpu, times_list_normal_cuda = [], [], [], []

CAT_EPOCHS = 3000
EPOCHS = 1500

NUM_LOOPS = 10

for i in range(100, CAT_EPOCHS, CAT_EPOCHS // NUM_LOOPS):
    start = perf_counter()
    run(['python', 'cat_no_split.py', 'cpu',  f'{str(i)}'], check=True)
    times_list_categorical_cpu.append(perf_counter() - start)
for i in range(100, CAT_EPOCHS, CAT_EPOCHS // NUM_LOOPS):
    start = perf_counter()
    run(['python', 'cat_no_split.py', 'cuda', f'{str(i)}'], check=True)
    times_list_categorical_cuda.append(perf_counter() - start)
for i in range(100, EPOCHS, EPOCHS // NUM_LOOPS):
    master.EPOCHS = i
    start = perf_counter()
    run(['python', 'normal_no_split.py', 'cpu', f'{str(i)}'], check=True)
    times_list_normal_cpu.append(perf_counter() - start)
    
for i in range(100, EPOCHS, EPOCHS // NUM_LOOPS):
    start = perf_counter()
    run(['python', 'normal_no_split.py', 'cuda', f'{str(i)}'], check=True)
    times_list_normal_cuda.append(perf_counter() - start)

# Plotting
# Calculate speedup for categorical data
speedup_list_categorical = [cpu_time / cuda_time for cpu_time, cuda_time in zip(times_list_categorical_cpu, times_list_categorical_cuda)]

# Calculate speedup for normal data
speedup_list_normal = [cpu_time / cuda_time for cpu_time, cuda_time in zip(times_list_normal_cpu, times_list_normal_cuda)]

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Plot categorical speedup
axes[0].plot(range(100, CAT_EPOCHS, CAT_EPOCHS // NUM_LOOPS), speedup_list_categorical, label='Categorical Speedup')
axes[0].set_title('Categorical Speedup')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Speedup')
axes[0].legend()
axes[0].grid(True)

# Plot normal speedup
axes[1].plot(range(100, EPOCHS, EPOCHS // NUM_LOOPS), speedup_list_normal, label='Normal Speedup')
axes[1].set_title('Normal Speedup')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Speedup')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
