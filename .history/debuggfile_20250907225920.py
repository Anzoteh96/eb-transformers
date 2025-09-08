import torch

print(torch.distributions.Dirichlet(torch.ones(100),).sample())