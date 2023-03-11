import torch

a=torch.rand(2,8,2,3)
b=torch.rand(2,8,3,10)

c=torch.matmul(a,b)
print(c)