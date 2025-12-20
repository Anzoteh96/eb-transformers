import pickle
import torch
import io
import glob
import argparse
import eb_train
import os
#from tabulate import tabulate
import math
from eb_transformer import EBTransformer
import numpy as np
import scipy.stats as st
import sys
from collections import defaultdict
import sgd_placket_luce
from tabulate import tabulate
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

from reducer_helper import extract_evals, mse_to_regret, get_tstats

if __name__ == "__main__":
    center_name = "mle"
    indir = sys.argv[1]
    outfile = sys.argv[2]

    # TODO: try to dig the model parameter(s) from the file and store it in a meta file? 

    # Here we determine the keys we could use to produce the PL rankings we're looking for. 
    met_keys = ['mses', 'norm_mses'] # metric keys

    mse_files = glob.glob(f"{indir}/*")
    
    eval_dict = {key: defaultdict(lambda : defaultdict(lambda : [])) for key in met_keys}

    mses_df, args_col = extract_evals(indir, "", [], 'mses', True)
    # print(mses_df)
    # normed_df, _ = extract_evals(indir, 'norm_mses')

   

    mses_vals_df = mses_df.drop(args_col, axis = 1).dropna(axis = 1)
    
    mses_vals = mses_vals_df.values # A numpy array. 
    mses_avg = np.mean(mses_vals, axis = 1)
    mses_std = np.std(mses_vals, axis = 1)
    # TODO: plant the confidence interval. 
    # ci = st.t.interval(0.95, len(mdl_mses)-1, loc=np.mean(mdl_mses), scale=st.sem(mdl_mses))

    # Maybe we can still keep placket luce, idk. 
    ranks = mses_vals.argsort(axis = 0)
    coefs = sgd_placket_luce.sgd_placket_luce(torch.from_numpy(ranks.T), max_iter = 20000)
    coefs = coefs.cpu().detach().numpy()
    # Now we can put things together. 
    results_df = pd.DataFrame({"Model": mses_df.index, "mean": mses_avg, "std": mses_std, "coefs": coefs})
    # If MLE present, subtract by MLE. 
    if 'mle' in mses_df.index:
        coefs_mle = results_df.loc[results_df["Model"] == 'mle', "coefs"].values.item()
        results_df['coefs'] -= coefs_mle
    
    if 'bayes' in mses_df.index:
        regret_df = mse_to_regret(mses_df, args_col)
        regret_vals = regret_df.drop(args_col, axis = 1).values
        regret_mean = np.mean(regret_vals, axis = 1)
        results_df.loc[(results_df["Model"] != 'bayes'), 'regret'] = regret_mean
        #TODO: check how to do this now that bayes row is gone. 
        # regret_val = results_df

    # Now that we have the results, we can just dump things. 
    results_df = results_df.sort_values("coefs", ascending = False).set_index("Model")
    results_df.to_csv(os.path.join(indir, "results.csv"), float_format='%.4f')
    tstats_mses, pval_mses = get_tstats(mses_df, args_col)
    tstats_mses.to_csv(os.path.join(indir, "tstats.csv"), float_format='%.3f')
    # TODO: add plots for everything. 
