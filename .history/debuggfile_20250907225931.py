import torch

print(torch.distributions.Dirichlet(torch.ones(10),).sample())