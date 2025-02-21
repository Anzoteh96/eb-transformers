import argparse
import glob
import io

# from tabulate import tabulate
import math
import os
import pickle
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import torch
from matplotlib.colors import LogNorm, SymLogNorm

import eb_train
import sgd_placket_luce
from eb_train import EBTransformer
from plot_utils import get_model_plot_name
from reducer_helper import extract_evals, get_tstats, mse_to_regret

if __name__ == "__main__":
    center_name = "mle"
    indir = sys.argv[1]

    # TODO: try to dig the model parameter(s) from the file and store it in a meta file?

    # Here we determine the keys we could use to produce the PL rankings we're looking for.
    met_keys = ["mses", "norm_mses"]  # metric keys

    mse_files = glob.glob(f"{indir}/*")

    eval_dict = {key: defaultdict(lambda: defaultdict(lambda: [])) for key in met_keys}

    mses_df, args_col = extract_evals(indir, "mses")
    # print(mses_df)
    # normed_df, _ = extract_evals(indir, 'norm_mses')

    tstats_mses, pval_mses = get_tstats(mses_df, args_col)

    mses_vals = (mses_df.drop(args_col, axis=1)).values  # A numpy array.
    mses_avg = np.mean(mses_vals, axis=1)
    mses_std = np.std(mses_vals, axis=1)
    # ci = st.t.interval(0.95, len(mdl_mses)-1, loc=np.mean(mdl_mses), scale=st.sem(mdl_mses))

    ranks = mses_vals.argsort(axis=0)
    coefs = sgd_placket_luce.sgd_placket_luce(torch.from_numpy(ranks.T), max_iter=1000)
    coefs = coefs.cpu().detach().numpy()
    # Now we can put things together.
    results_df = pd.DataFrame(
        {"Model": mses_df.index, "mean": mses_avg, "std": mses_std, "coefs": coefs}
    )
    # If MLE present, subtract by MLE.
    if "mle" in mses_df.index:
        coefs_mle = results_df.loc[results_df["Model"] == "mle", "coefs"].values.item()
        results_df["coefs"] -= coefs_mle

    if "bayes" in mses_df.index:
        regret_df = mse_to_regret(mses_df, args_col)
        regret_vals = regret_df.drop(args_col, axis=1).values
        regret_mean = np.mean(regret_vals, axis=1)
        results_df.loc[(results_df["Model"] != "bayes"), "regret"] = regret_mean
        # TODO: check how to do this now that bayes row is gone.
        # regret_val = results_df

    # Now that we have the results, we can just dump things.
    results_df = (
        results_df.sort_values("coefs", ascending=False)
        .drop_duplicates(subset=["Model"])
        .set_index("Model")
    )
    results_df.to_csv(os.path.join(indir, "results.csv"), float_format="%.3f")
    tstats_mses.to_csv(os.path.join(indir, "tstats.csv"), float_format="%.3f")
