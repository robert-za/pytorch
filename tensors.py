import torch
import numpy as np

# creating tensor from a list
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# creating from numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# from another tensor - keeps dimensions of x_data
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  #override datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# tensor shape
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

# tensor attributes
tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")  # Shape of tensor: torch.Size([3, 4])
print(f"Datatype of tensor: {tensor.dtype}")  # Datatype of tensor: torch.float32
print(f"Device tensor is stored on: {tensor.device}")  # Device tensor is stored on: cpu

# if GPU is available
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f"Device tensor is stored on {tensor.device}")
print("CUDA not available.")

# standard numpy-like indexing and slicing
tensor = torch.ones(4, 4)
# tensor[1:2,2:4] = 0
tensor[:,1] = 0
print(tensor)
print("\n")

# joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)  # TODO: verify dim argument
print(t1)
print("\n")

# multiplying tensors
# this computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# alt syntax
print(f"tensor * tensor \n {tensor * tensor}")
print("\n")

# matrix multiplication
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
print("\n")

# in place ops
print(tensor, "\n")
tensor.add_(5)
print(tensor)

# bridge with numpy
# https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#bridge-with-numpy