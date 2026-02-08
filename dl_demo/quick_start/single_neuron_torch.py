import torch

# Single neuron example (PyTorch)

# Input (1 x 3)
x = torch.tensor([[1.0, 2.0, 3.0]])

# Weight (3 x 1)
w = torch.tensor([[-0.5],
                  [ 0.2],
                  [ 0.1]])

print("X:\n", x)
print("Shape:", x.shape)

print("\nW:\n", w)
print("Shape:", w.shape)

# Bias (scalar)
b1 = torch.tensor(0.5)

# Linear computation: z = xW + b
z1 = torch.matmul(x, w) + b1

print("\nZ:\n", z1)
print("Shape:", z1.shape)
