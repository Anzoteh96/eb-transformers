import torch

print(torch.distributions.Dirichlet(torch.ones(m=100)).sample())