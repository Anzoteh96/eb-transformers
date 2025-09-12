import torch


x = torch.rand(100) * 100 # support
y =torch.distributions.Dirichlet(torch.ones(100),).sample() # weights

#print(x)
#print(y)
#print(torch.tensor(true_parameters).reshape(1, 100, 1))



indices = torch.multinomial(y, 100, replacement=True) #apparently multinomial is more efficient
thetas = x[indices]
print(thetas.reshape(1, 100, 1))