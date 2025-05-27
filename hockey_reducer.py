import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import argparse
from tqdm import tqdm
from scipy.stats import ttest_rel
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

def process_hockey_data(target_dir, model_lst, num_space):
    results = []
    for filename in tqdm(os.listdir(target_dir)):
        if filename[-3:] != 'pkl':
            continue
        fullpath = os.path.join(target_dir, filename)
        try:
            lookup = pickle.load(open(fullpath, "rb"))
        except:
            continue
        model = filename[:-(num_space + 15)]
        if not (model in model_lst):
            continue
        if not "args" in lookup:
            model_args = None
        else:
            model_args = lookup['args']
        if model_args is not None:
            if model[-4:] == ".pkl": # remove .pkl
                 model = model[:-4]
            model_name = model
            #print(israndom, model_name)
        elif model == "fixed_grid_npmle":
            model_args = None
            model_name = "NPMLE"
        else:
            model_args = None
            model_name = model

        year = int(filename[-8:-4])
        sub_results = pd.DataFrame({"Input": lookup['input'].flatten(), "Output": lookup['output'].flatten(), "Labels": lookup['label'].flatten(), "Model": model_name, "Year": year})
        sub_results['Squared Diff'] = (sub_results["Output"] - sub_results["Labels"]) ** 2
        sub_results['Abs Diff'] = np.abs(sub_results["Output"] - sub_results["Labels"])
        results.append(sub_results)
        #if model == "erm":
            #print(filename)
            #print(sub_results)

    results = pd.concat(results)
    print(results["Model"].unique())
    return results

def hockey_compare_stats(df, model1, model2, metric):
    df_map = {mod: df for mod, df in df.groupby('Model')}
    df1 = df_map[model1]
    df2 = df_map[model2]
    df_both = pd.merge(df1, df2, on = ["Year"])
    met_x = metric + '_x'
    met_y = metric + '_y'

    return df_both[met_x] / df_both[met_y]

def hockey_tstats(df, model_lst_1, model_lst_2, metric, alternative='two-sided'):
    df_map = {mod: df for mod, df in df.groupby('Model')}
    comparisons = []
    for mod1 in model_lst_1:
        df1 = df_map[mod1]
        for mod2 in model_lst_2:
            df2 = df_map[mod2]
            #print(df2[metric].reset_index() / df1[metric].reset_index())
            df_both = pd.merge(df1, df2, on = ["Year"])
            met_x = metric + '_x'
            met_y = metric + '_y'
            tstats = ttest_rel(df_both[met_x], df_both[met_y], alternative=alternative)
            comparisons.append({"Model1": mod1, "Model2": mod2, "T-stat": tstats.statistic, "P-val": tstats.pvalue})
    return pd.DataFrame(comparisons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hockey_reducer")
    parser.add_argument(
        "--results_dir", type=str, help="results directory"
    )
    parser.add_argument(
        "--pos", type=str, help="positions"
    )
    args = parser.parse_args()
    transformers_lst = ['T24r', 'L24r']
    model_lst = ['mle', 'erm', 'fixed_grid_npmle', 'robbins'] + transformers_lst
    hockey_pos = ["all", "defender", "center", "winger"]
    num_space_map = {"all": -1, "defender": 8, "center": 6, "winger": 6}
    assert args.pos in hockey_pos, "Position not in {}".format(hockey_pos)
    
    result_dir = args.results_dir
    df = process_hockey_data(result_dir, model_lst, num_space=num_space_map[args.pos])
    hockey_results0 = df
    # Normalize by MLE's losses. 
    hockey_rmse0 = {}
    hockey_rmse_norm0 = {}
    hockey_mae0 = {}
    hockey_mae_norm0 = {}

    df = hockey_results0.groupby(['Model', 'Year'])['Squared Diff'].mean().reset_index()
    df['RMSE'] = np.sqrt(df['Squared Diff'])
    hockey_rmse0 = df
    df_pivot = df.pivot(index='Model', columns='Year', values='RMSE')
    df_norm = df_pivot / df_pivot.loc['mle']
    hockey_rmse_norm0 = df_norm.melt(ignore_index=False, value_name='RMSE/MLE').reset_index()
    
    df_mae = hockey_results0.groupby(['Model', 'Year'])['Abs Diff'].mean().reset_index()
    hockey_mae0 = df_mae
    df_mae_pivot = df_mae.pivot(index='Model', columns='Year', values='Abs Diff')
    df_mae_norm = df_mae_pivot / df_mae_pivot.loc['mle']
    hockey_mae_norm0 = df_mae_norm.melt(ignore_index=False, value_name='MAE/MLE').reset_index()

    # Get the t-stat/p-value.
    rmse_tstat = hockey_tstats(hockey_rmse0, ['erm', 'NPMLE', "robbins", "mle"], transformers_lst,'RMSE', 'greater')
    rmse_pval = rmse_tstat.pivot(index='Model2', columns='Model1', values='P-val').loc[transformers_lst][['mle', 'robbins', 'erm', 'NPMLE']]
    mae_tstat = hockey_tstats(hockey_mae0, ['erm', 'NPMLE', "robbins", "mle"], transformers_lst,'Abs Diff', 'greater')
    mae_pval = mae_tstat.pivot(index='Model2', columns='Model1', values='P-val').loc[transformers_lst][['mle', 'robbins', 'erm', 'NPMLE']]
    print("RMSE", rmse_pval)
    print("MAE", mae_pval)
    
    # How we get the plots. 
    rmse_pivot = hockey_rmse0.pivot(index='Model', columns='Year', values='RMSE')
    ratio_all = (rmse_pivot / rmse_pivot.loc['mle']).loc[['erm', 'NPMLE', 'T24r', 'L24r']]
    ratio_melt = ratio_all.melt(value_name = 'RMSE ratio', ignore_index=False).reset_index()
    sns.violinplot(ratio_melt, x = 'Model', y = 'RMSE ratio')
    plt.axhline(y = 1.00, color = 'r', linestyle='-')
    
    mae_pivot = hockey_mae0.pivot(index='Model', columns='Year', values='Abs Diff').loc[models]
    ratio_mae_all = (mae_pivot / mae_pivot.loc['mle']).loc[['erm', 'NPMLE', 'T24r', 'L24r']]
    ratio_melt = ratio_mae_all.melt(value_name = 'MAE ratio', ignore_index=False).reset_index()
    sns.violinplot(ratio_melt, x = 'Model', y = 'MAE ratio')
    plt.axhline(y = 1.00, color = 'r', linestyle='-')
