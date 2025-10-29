import torch
DIM = 512
A = 50
m = [5,10,20,40,80] # possible values of m 

x = torch.rand(80) * A # support
y =torch.distributions.Dirichlet(torch.ones(80),).sample() # weights

print(x)
print(y)
#print(torch.tensor(true_parameters).reshape(1, 100, 1))



indices = torch.multinomial(y, DIM, replacement=True) #apparently multinomial is more efficient
thetas = x[indices]
print(thetas.reshape(1, DIM, 1))