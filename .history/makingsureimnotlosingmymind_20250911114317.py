import torch


x = torch.rand(80) * 50 # support
y =torch.distributions.Dirichlet(torch.ones(80),).sample() # weights

print(x)
print(y)
#print(torch.tensor(true_parameters).reshape(1, 100, 1))



indices = torch.multinomial(y, 512, replacement=True) #apparently multinomial is more efficient
thetas = x[indices]
print(thetas.reshape(1, 512, 1))