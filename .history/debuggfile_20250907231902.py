import torch


x = torch.rand(10) * 540
y =torch.distributions.Dirichlet(torch.ones(10),).sample()

print(torch.dot(x, y))