import pickle
import scipy as sp
import numpy as np
import torch
import sys
import argparse
from tqdm import tqdm

def get_posterior(lookup):
    inputs_all = torch.from_numpy(lookup["inputs"])
    outputs_all = torch.from_numpy(lookup["outputs"])
    labels_all = torch.from_numpy(lookup["labels"])
    B, N, D = inputs_all[0].shape
    assert (D == 1) # Others are not supported. 

    lookup_rets = {}
    for key in lookup:
        if key in ["pi", "grid"]:
            continue
        lookup_rets[key] = lookup[key]

    pi_all = lookup["pi"]
    grids_all = lookup["grid"]

    marginals_all = []
    marginals_next_all = []

    for inputs, outputs, labels, pi, grids in zip(inputs_all, outputs_all, labels_all, pi_all, grids_all):
        marg_ = []
        margnext_ = []
        for inp, outp, lab, p, g in tqdm(zip(inputs, outputs, labels, pi, grids)):
            grid_poisson = torch.distributions.poisson.Poisson(rate = torch.from_numpy(g))
            phi_mat = torch.exp(grid_poisson.log_prob(inp.reshape(-1,1)))
            phi_mat_nxt = torch.exp(grid_poisson.log_prob((inp + 1).reshape(-1,1)))
            marginals = phi_mat @ p
            marg_nxt = phi_mat_nxt @ p
            marg_.append(marginals.numpy())
            margnext_.append(marg_nxt.numpy())

        marg_ = np.stack(marg_)
        marginals_all.append(marg_)
        margnext_ = np.stack(margnext_)
        marginals_next_all.append(margnext_)

    lookup_rets["posterior"] = np.stack(marginals_all)
    lookup_rets["posterior_next"] = np.stack(marginals_next_all)
    
    outs = (1 + (inputs_all)[...,0]) * lookup_rets["posterior_next"] / lookup_rets["posterior"]
    diff = outs - outputs_all[...,0]
    print(torch.abs(diff).sum().item())
    return lookup_rets

if __name__ == "__main__":
    
    print(sys.argv)
    parser = argparse.ArgumentParser(description='none')
    parser.add_argument('--filename')
    args = parser.parse_args()

    lookup = pickle.load(open(args.filename, "rb"))
    lookup_rets = get_posterior(lookup)
    outfile_name = "{}-posterior".format(args.filename)
    print("I am outputting here", outfile_name)
    with open(outfile_name, "wb") as f:
        pickle.dump(lookup_rets, f)
