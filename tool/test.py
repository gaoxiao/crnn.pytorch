import torch

a = torch.randn(4, 4)
print(a)
print(torch.topk(a, 2, dim=0))
print(torch.topk(a, 2, dim=1))
