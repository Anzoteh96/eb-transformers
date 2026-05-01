# Here we consider some of the unconvential channels. 

import numpy as np 
import torch 

def slcp_channel(x: torch.Tensor): 
    """ Simple likelihood and complex posterior channel as in Wilson et al. 2016.
    Reference: 
    A. G. Wilson, Z. Hu, R. Salakhutdinov, and E. P. Xing. Deep kernel learning. In AISTATS, 2016.
    """
    assert x.shape[-1] == 5, "Input dimension must be 5 for SLCP channel."
    mu = x[...,:2]
    s1 = x[...,2]**2
    s2 = x[...,3]**2
    rho = torch.tanh(x[...,4])  # ensure correlation is between -1 and 1
    cov = torch.zeros((*x.shape[:-1], 2, 2), device=x.device)
    cov[...,0,0] = s1**2
    cov[...,1,1] = s2**2
    cov[...,0,1] = rho * s1 * s2
    cov[...,1,0] = cov[...,0,1]
    # from IPython import embed; embed()
    dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov)
    y1 = dist.rsample()
    y2 = dist.rsample()
    y3 = dist.rsample()
    y4 = dist.rsample()
    y = torch.cat([y1, y2, y3, y4], dim=-1)
    return y 

def slcp_mle(y: torch.Tensor, clamp=None):
    """ MLE for SLCP channel.
    """
    assert y.shape[-1] == 8, "Input dimension must be 4 for SLCP MLE."
    y1 = y[..., 0:2]
    y2 = y[..., 2:4]
    y3 = y[..., 4:6]
    y4 = y[..., 6:8]
    y_expand = torch.stack([y1, y2, y3, y4], dim=-2)  # shape [..., 4, 2]
    mu_hat = torch.mean(y_expand, dim=-2)  # shape [..., 2]
    # For sigma_hat, things will get more interesting. 
    s1 = y_expand[..., :, 0] - mu_hat[..., 0].unsqueeze(-1)
    s2 = y_expand[..., :, 1] - mu_hat[..., 1].unsqueeze(-1)
    s1_sum = torch.mean(s1**2, dim=-1)
    s2_sum = torch.mean(s2 **2, dim=-1)
    s1s2 = torch.mean(s1 * s2, dim=-1)
    quo = s1s2 / torch.sqrt(s1_sum * s2_sum + 1e-8)
    if clamp is not None:
        quo = torch.clamp(quo, np.tanh(-clamp), np.tanh(clamp))
    s3 = torch.atanh(quo)
    x2 = torch.stack([torch.sqrt(torch.sqrt(s1_sum)), torch.sqrt(torch.sqrt(s2_sum)), s3], dim=-1)
    x = torch.cat([mu_hat, x2], dim=-1)
    return x

def two_moons_channel(x: torch.Tensor):
    """ Two moons dataset channel.
    """
    assert x.shape[-1] == 2, "Input dimension must be 2 for two moons channel."
    x1 = x[..., 0]
    x2 = x[..., 1]
    a = torch.randn((*x.shape[:-1],), device=x.device) # 0, 1. 
    theta = (2 * a - 1) * np.pi 
    r = torch.distributions.Normal(0.1, 0.01 ** 22).rsample(x.shape[:-1]).to(x.device)
    p1 = r * torch.cos(theta) + 0.25
    p2 = r * torch.sin(theta)
    p = torch.stack([p1, p2], dim=-1)
    y1 = -torch.abs(x1+x2) / np.sqrt(2)
    y2 = (-x1+x2) / np.sqrt(2)
    y = p + torch.stack([y1, y2], dim=-1)
    return y

def inv_kinematrics_channel(x: torch.Tensor):
    """ Inverse kinematics channel.
    Reference:
    C. M. Bishop. Pattern recognition and machine learning. springer, 2006.
    """
    assert x.shape[-1] == 4, "Input dimension must be 4 for inverse kinematics channel."

    l1 = 0.5 # + torch.normal(0, 0.0001**2).rsample(x.shape[:-1])
    l2 = 0.5 
    l3 = 1.0
    x1 = x[..., 0]
    x2 = x[..., 1]
    x3 = x[..., 2]
    x4 = x[..., 3]

    y1 = x1 + l1* torch.sin(x2 + torch.normal(0, 0.0001**2).rsample(x.shape[:-1]))\
    + l2 * torch.sin(x2 + x3 + torch.normal(0, 0.0001**2).rsample(x.shape[:-1]))\
    + l3 * torch.sin(x2 + x3 + x4 + torch.normal(0, 0.0001**2).rsample(x.shape[:-1]))
    y2 = l1* torch.sin(x2 + torch.normal(0, 0.0001**2).rsample(x.shape[:-1]))\
    + l2 * torch.sin(x2 + x3 + torch.normal(0, 0.0001**2).rsample(x.shape[:-1]))\
    + l3 * torch.sin(x2 + x3 + x4 + torch.normal(0, 0.0001**2).rsample(x.shape[:-1]))
    y = torch.stack([y1, y2], dim=-1)
    return y