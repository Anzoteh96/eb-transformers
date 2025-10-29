import torch


x = torch.rand(1000) * 540 # support
y =torch.distributions.Dirichlet(torch.ones(1000),).sample() # weights

#print(x)
#print(y)
scalar = float(torch.dot(x, y))
true_parameters = torch.full((100,), scalar, dtype=torch.float32)
print(true_parameters)
#print(torch.tensor(true_parameters).reshape(1, 100, 1))


indices = torch.multinomial(y, 100, replacement=True) #apparently multinomial is more efficient
thetas = x[indices]
print(thetas.reshape(1, 5, 1))