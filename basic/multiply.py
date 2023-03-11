import torch

a=torch.rand(5,2,8,2,3)
b=torch.rand(11,21,15,5,2,8,3,10)

c=torch.matmul(a,b)
print(a.shape)
print(b.shape)
print(c.shape)