import subprocess
import matplotlib.pyplot as plt
import random

batch_sizes = [10, 100, 500, 1000]

optimized_times = []
unoptimized_times = []

seeds = []

# # seed = int(random.random()*10000)
seed = 3331
random.seed(seed)
# print(f"{seed=}")

# # populate the seed list
# for index in range(0,2):
#     seed = int(random.random()*10000)
#     seeds.append(seed)



print(f'Running for seed: {seed}')

for batch in batch_sizes:
    print(f'Running for batch size: {batch}')

    unopt = subprocess.run(["python", "unoptimized.py", "--seed", str(seed), "--batch_size", str(batch)], capture_output=True, text=True)
    print(f'Finished processing unoptimal. Time taken = {unopt}')

    opt = subprocess.run(["python", "optimized.py", "--seed", str(seed), "--batch_size", str(batch)], capture_output=True, text=True)
    print(f'Finished processing optimal. Time taken = {opt}')

    optimized_time = float(opt.stdout.strip().splitlines()[-1])
    unoptimized_time = float(unopt.stdout.strip().splitlines()[-1])

    optimized_times.append(optimized_time)
    unoptimized_times.append(unoptimized_time)

# Plotting
plt.plot(batch_sizes, optimized_times, label="Optimized", marker='o')
plt.plot(batch_sizes, unoptimized_times, label="Unoptimized", marker='o')
plt.xlabel("Batch Size")
plt.ylabel("Execution Time (s)")
plt.title("Performance Comparison")
plt.legend()
plt.grid(True)
plt.show()
