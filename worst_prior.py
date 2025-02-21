# Test file, lol. 

from gen_priors import WorstPrior
from easydict import EasyDict as edict
import scipy as sp
import torch
import sys
import numpy as np
import os 

theta_max = 50
dinput = 1
seed = int(sys.argv[1])
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
num_grids = 2500
tot_iter = 20000
args = edict({"theta_max": theta_max, "dinput": dinput, "seed": seed, "num_grids": num_grids, "tot_iter": tot_iter})

save_file = "worst_priors/theta{}_seed{}.npz".format(theta_max, seed)

wp = WorstPrior(args)
#print(wp.support, wp.probs)

# Now we wanna check the loss func. 

all_thetas = wp.support
all_probs = wp.probs

np.savez(save_file, atoms=all_thetas.numpy(), probs = all_probs.numpy())
# Can we also save the thetas and probability into an npz file? 


loss_theta = []
# all_thetas = torch.arange(50)

xs = torch.arange(10 * args.theta_max)
est = wp.gen_bayes_est(xs)
for theta, prob in zip(all_thetas, all_probs):
    # Case theta == 0 needs to be taken care of! 
    if theta == 0:
        dens = torch.tensor([xs != 0])
    else:
        log_dens = torch.stack([-theta - sp.special.gammaln(x + 1) + x * torch.log(theta) for x in xs])
        dens = torch.exp(log_dens)
    diff = est - theta
    weighted_diff = torch.sum(dens * (diff ** 2))
    loss_theta.append(weighted_diff)
total_loss = torch.sum(all_probs * loss_theta)
print(total_loss.item())

all_thetas = torch.linspace(0, 50, 1000)
for theta in all_thetas:
    log_dens = torch.stack([-theta - sp.special.gammaln(x + 1) + x * torch.log(theta) for x in xs])
    dens = torch.exp(log_dens)
    try:
        diff = est - theta
    except:
        from IPython import embed; embed()
    weighted_diff = torch.sum(dens * (diff ** 2))
    # print(theta.item(),  weighted_diff.item())

# Overall loss
loss_theta = torch.tensor(loss_theta)
# print(wp.gen_bayes_est(xs))
