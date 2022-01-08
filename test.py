import torch

BATCH_SIZE = 1
DATA_DIM = 4

generator = torch.nn.Linear(DATA_DIM, DATA_DIM)

seed = torch.rand(10, 4)
data = generator(seed)

A = torch.rand(3, 3, requires_grad=True)
B = torch.rand(3, 3, requires_grad=True)

P = A*B
P[0][0] = 0



a = torch.tensor([1.], requires_grad=True)
b = torch.tensor([1.], requires_grad=True)
c = torch.tensor([1.], requires_grad=True)

aa = a.detach()

print(a)
print(aa)
aa[0] = 99
print(a)
print(aa)
