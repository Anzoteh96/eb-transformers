import torch


x = torch.rand(10) * 540
y =torch.distributions.Dirichlet(torch.ones(10),).sample()

#print(x)
#print(y)
scalar = float(torch.dot(x, y))
true_parameters = torch.full((100,), scalar, dtype=torch.float32)
print(true_parameters)
#print(torch.tensor(true_parameters).reshape(1, 100, 1))