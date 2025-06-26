import subprocess
import matplotlib.pyplot as plt
import random

batch_sizes = [10, 100, 500, 1000]
# batch_sizes = [1]
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

    opt = subprocess.run(["python", "optimized3.py", "--seed", str(seed), "--batch_size", str(batch)], capture_output=True, text=True)
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


"""
import subprocess
import matplotlib.pyplot as plt
import random

# batch_sizes = [10, 100, 500, 1000]
batch_sizes = [1, 2]
optimized_times = []
unoptimized_times = []

optimized_free_times = []
unoptimized_free_times = []

optimized_connects_times = []
unoptimized_connects_times = []

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
    print(f'Finished processing unoptimal. Results = {unopt}')

    # opt = subprocess.run(["python", "optimized.py", "--seed", str(seed), "--batch_size", str(batch)], capture_output=True, text=True)
    # print(f'Finished processing optimal. Results = {opt}')

    # optimized_time = float(opt.stdout.strip().splitlines()[-1])
    # unoptimized_time = float(unopt.stdout.strip().splitlines()[-1])

    # optimized_times.append(optimized_time)
    # unoptimized_times.append(unoptimized_time)

    # optimized_free_time = float(opt.stdout.strip().splitlines()[-2])
    unoptimized_free_time = float(unopt.stdout.strip().splitlines()[-1])

    # optimized_connects_time = float(opt.stdout.strip().splitlines()[-1])
    unoptimized_connects_time = float(unopt.stdout.strip().splitlines()[-1])

    # optimized_free_times.append(optimized_free_time)
    unoptimized_free_times.append(unoptimized_free_time)

    # optimized_connects_times.append(optimized_connects_time)
    unoptimized_connects_times.append(unoptimized_connects_time)

# Plotting
# plt.plot(batch_sizes, optimized_times, label="Optimized", marker='o')
plt.plot(batch_sizes, unoptimized_free_times, label="Unoptimized", marker='o')
plt.xlabel("Batch Size")
plt.ylabel("Execution Time (s)")
plt.title("Performance Comparison")
plt.legend()
plt.grid(True)
plt.show()

"""