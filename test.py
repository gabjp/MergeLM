import torch

a = torch.Tensor([[1,2,3],[4,5,6]])
print(a)
b = torch.Tensor([[2],[6]])
print(b)
print(a >= b)