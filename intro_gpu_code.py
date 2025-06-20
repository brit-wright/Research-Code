import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('GPU is available and will be used')
else:
    device = torch.device('cpu')
    print('GPU not available. Falling back on CPU')

# Create random matrices on the GPU
a = torch.rand((1000,1000), device = device)
b = torch.rand((1000,1000), device = device)

# Multiply the matrices
result = a @ b

# Print part of the result and the device used
print("Result shape:", result.shape)
print("Result device:", result.device)
print("First 5 values of result:\n", result.flatten()[:5])
