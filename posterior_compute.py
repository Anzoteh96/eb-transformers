# Temporary file such that given x, return the posterior. 

import numpy as np
import torch
import argparse
import os
import sys
import pickle

from algo_helpers import poison_eb_npmle_prior, eval_regfunc

def get_posterior_from_x(x_all):
    """
        Args:
            x: torch Tensor of length B x N
        Returns:
            f_{\pi}(x+post_shift)
    """
    marg_ = []
    margnext_ = []  
    for x in x_all:
        pi, grid = poison_eb_npmle_prior(x.flatten())
        # Now need to get the posterior density. 

        grid_poisson = torch.distributions.poisson.Poisson(rate = grid)
        phi_mat = torch.exp(grid_poisson.log_prob(x.reshape(-1,1)))
        phi_mat_nxt = torch.exp(grid_poisson.log_prob((x + 1).reshape(-1,1)))
        marginals = phi_mat @ pi
        marg_nxt = phi_mat_nxt @ pi
        marg_.append(marginals.numpy())
        margnext_.append(marg_nxt.numpy())
    
    marg_ = np.stack(marg_)
    margnext_ = np.stack(margnext_)
    return marg_, margnext_


if __name__ == "__main__":
    
    print(sys.argv)
    parser = argparse.ArgumentParser(description='none')
    parser.add_argument('--filename')
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--outdir')
    args = parser.parse_args()

    lookup = pickle.load(open(args.filename, "rb"))
    x_all = lookup["x"][args.start:args.end]

    marg, margnext = get_posterior_from_x(torch.from_numpy(x_all))

    outfile_name = os.path.join(args.outdir, "{}_{}.pkl".format(args.start, args.end))
    with open(outfile_name, "wb") as f:
        pickle.dump({"x": x_all, "start": args.start, "end": args.end, "posterior": marg, "posterior_next": margnext}, f)

