import torch
import numpy as np


#Directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

print(x_data)

#From numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(x_np)

#From another tensor
x_ones = torch.ones_like(x_data)

print(x_ones)
x_rand = torch.rand_like(x_data, dtype=torch.float)

print(x_rand)

#With random or constant values
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


#Tensor attributes: Describe the shape, data types and the device on which it is stored
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")