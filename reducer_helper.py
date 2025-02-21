# Here we basically just get all the evaluations within a folder, and then try to aggregate them.
# We are primarily interested in MSEs Regret, and the t-stat comparison.

import os
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from tqdm import tqdm

from eb_arena_mapper import load_model_dict
from sgd_placket_luce import sgd_placket_luce


def extract_evals(eval_dir, met="mses", target_args=None, model_dir=None):
    """
    Args:
        eval_dir: the directory with all pickle files
        met: the metric that we wanted to store.
        target_args: the list of arguments we want to get (if None, we ignore for now but should really move to storing every dict).
        model_dir: if supplied, the directory where we get the model arguments (usually it should be in the pickle files in eval_dir)
    """
    # Here each pickle file should contain start, end, mdl_name
    args_lookup = {}
    mses_df_lst = []
    # norm_mses_lookup = {}
    for file in tqdm(os.listdir(eval_dir)):
        full_path = os.path.join(eval_dir, file)
        try:
            pkl_dict = pickle.load(open(full_path, "rb"))
        except:
            # Not pickle file, continue.
            continue

        # Next let's load the seeds.
        seed_start = pkl_dict["start"]
        seed_end = pkl_dict["end"]
        nme = pkl_dict["mdl_name"]
        # We want to also get the arguments.
        if "args" in pkl_dict:
            model_args = pkl_dict["args"]
        else:
            try:
                model_name = os.path.join(
                    model_dir, pkl_dict["mdl_name"]
                )  # If model_dir is None, will throw an error.
                model_args = load_model_dict(model_name, "cpu")["args"]
            except:
                model_args = None

        if target_args is not None:
            for arg in target_args:
                if not (nme in args_lookup):
                    args_lookup[nme] = {}
                if model_args is not None:
                    args_lookup[nme][arg] = eval("model_args.{}".format(arg))
                else:
                    args_lookup[nme][arg] = np.nan
        elif model_args is not None:
            args_lookup[nme] = vars(model_args)  # Convert namespace to dict.
        else:
            args_lookup[nme] = None

        start_seed = pkl_dict["start"]
        end_seed = pkl_dict["end"]
        temp_df = pd.DataFrame(
            {"seed": range(start_seed, end_seed), met: pkl_dict[met]}
        )
        temp_df["mdl_name"] = nme
        temp_df.set_index(["mdl_name", "seed"], inplace=True)
        mses_df_lst.append(temp_df)

    # print(mses_df_lst)
    # print(pd.concat(mses_df_lst))

    mses_all = pd.concat(mses_df_lst).drop_duplicates().unstack()
    # Now, get all the args.
    all_keys = target_args

    key_df = pd.DataFrame(args_lookup).T
    all_args = key_df.columns
    # print(key_df)
    df_return = pd.concat([key_df, mses_all], axis=1)
    return df_return, all_args  # Returning this just in case we passed in [].


# Next, given a dataframe of loss (index = model; column = item), we want to do the plackett luce thing.
def get_plackettluce_coefs(loss_df, lr=1e-2, zero_coeff=None):
    loss_torch = torch.from_numpy(loss_df.values.argsort(axis=0))
    coeffs = sgd_placket_luce(loss_torch.T, max_iter=10000, lr=1e-2)
    coeffs_df = pd.Series(coeffs.detach().numpy(), index=loss_df.index)
    if zero_coeff is not None and zero_coeff in loss_df.index:
        coeffs_df = coeffs_df - coeffs_df.loc[zero_coeff]
    return coeffs_df


# Given a DataFrame, if there are arguments we want to replace the index with arguments, and let the index stay as it is otherwise.
def replace_ind_with_args(df, arg_col_names):
    arg_cols = df[arg_col_names].columns
    new_ind = np.where(
        ~df[arg_cols[0]].isnull(),
        np.array([str(tuple(row)) for row in np.array(df[arg_col_names])]),
        df.index,
    )
    new_vals = df.drop(arg_col_names, axis=1)
    answer_df = pd.DataFrame(new_vals.values, index=new_ind, columns=new_vals.columns)
    return answer_df


# Given an MSE dataframe, get the regret dataframe.
def mse_to_regret(df, args_col, bayes_key="bayes"):
    df_vals = df.drop(args_col, axis=1)
    df_cols = df[args_col]
    mse_bayes = df_vals.loc[bayes_key]
    regret_raw = df_vals - mse_bayes
    regret_df = pd.concat([df_cols, regret_raw], axis=1).drop(bayes_key)
    return regret_df


# Another helper function: get the T-stats.
def get_tstats(df, args_col):
    comparisons = []
    df_vals = df.drop(args_col, axis=1).values
    df_cols = df[args_col]
    for i, val1 in enumerate(df_vals):
        key_i = tuple(df_cols.iloc[i])
        if key_i is None or pd.isna(key_i[0]):
            key_i = df_cols.index[i]
        for j, val2 in enumerate(df_vals):
            if i == j:
                continue
            key_j = tuple(df_cols.iloc[j])
            if key_j is None or pd.isna(key_j[0]):
                key_j = df_cols.index[j]
            try:
                tstats = ttest_rel(val1, val2)
            except:
                print(val1)
                print(val2)
            comparisons.append(
                {
                    "Model 1": key_i,
                    "Model 2": key_j,
                    "T-stat": tstats.statistic,
                    "P-val": tstats.pvalue,
                }
            )
    df_comp = pd.DataFrame(comparisons)
    tstat_df = pd.DataFrame(comparisons).pivot(
        index="Model 1", columns="Model 2", values="T-stat"
    )
    pval_df = pd.DataFrame(comparisons).pivot(
        index="Model 1", columns="Model 2", values="P-val"
    )
    return tstat_df, pval_df
