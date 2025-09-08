import torch


x = torch.rand(10) * 540
y =torch.distributions.Dirichlet(torch.ones(10),).sample()

scalar = float(torch.dot(x, y))
true_parameters = torch.full((100), scalar, dtype=torch.float32)

print(torch.tensor(true_parameters).reshape(1, 100, 1))