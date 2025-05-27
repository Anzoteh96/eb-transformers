import os
import pickle
import argparse
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Get the t-stats of baseball comparison. 
def baseball_tstats(df, model_lst_1, model_lst_2, metric, alternative = 'two-sided'):
    df_map = {mod: df for mod, df in df.groupby('Model')}
    comparisons = []
    for mod1 in model_lst_1:
        df1 = df_map[mod1]
        for mod2 in model_lst_2:
            df2 = df_map[mod2]
            #print(df2[metric].reset_index() / df1[metric].reset_index())
            df_both = pd.merge(df1, df2, on = ["Year", "Position"])
            met_x = metric + '_x'
            met_y = metric + '_y'
            tstats = ttest_rel(df_both[met_x], df_both[met_y], alternative=alternative)
            comparisons.append({"Model1": mod1, "Model2": mod2, "T-stat": tstats.statistic, "P-val": tstats.pvalue})
    return pd.DataFrame(comparisons)

# Comparison with MLE. 
def baseball_compare_stats(df, pos_name, mod1, mod2, metric):
    df_map = {mod: df for mod, df in df.groupby('Model')}
    df1 = df_map[mod1]
    df2 = df_map[mod2]
    df_both = pd.merge(df1, df2, on = ["Year", "Position"])
    met_x = metric + '_x'
    met_y = metric + '_y'
    return df_both[met_x]/df_both[met_y]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="baseball_reducer")
    parser.add_argument(
        "--results_dir", type=str, help="results directory"
    )

    results = []
    time_df = []
    args = parser.parse_args()
    results_dir_name = args.results_dir

    for filename in os.listdir(results_dir_name):
        fullpath = os.path.join(results_dir_name, filename)
        tokens = filename.split("_")
        pos = tokens[-2]
        model = "_".join(tokens[:-2])
        lookup = pickle.load(open(fullpath, "rb"))
    
        if lookup['args'] is not None:
            model_args = lookup['args']
            if model[-4:] == ".pkl": # remove .pkl
                model = model[:-4]
        else:
            model_args = None
        if model == "fixed_grid_npmle":
            model_name = "NPMLE"
        else:
            model_name = model
        #print(tokens)
        year = int(filename[-8:-4])
    
        sub_results = pd.DataFrame({"Input": lookup['input'].flatten(), "Output": lookup['output'].flatten(), "Labels": lookup['label'].flatten(), "Model": model_name, "Year": year, "Position": pos})
    
        # sub_results[sub_results["Output"] < 0] = 0
    
        sub_results['Squared Diff'] = (sub_results["Output"] - sub_results["Labels"]) ** 2
        sub_results['Abs Diff'] = np.abs(sub_results["Output"] - sub_results["Labels"])
        time_df.append((model_name, pos, year, lookup['time']))
        results.append(sub_results)
    results_df = pd.concat(results)
    time_df = pd.DataFrame(time_df)
    
    time_df.columns = ['Model', 'Position', 'Year', 'Time']
    
    rmse_df = np.sqrt(results_df.groupby(['Model', 'Year', 'Position'])['Squared Diff'].mean()).reset_index()
    rmse_df.columns = ['Model', 'Year', 'Position', 'RMSE']
    rmse_df_bat = rmse_df[rmse_df['Position'] == 'bat']
    rmse_df_pitch = rmse_df[rmse_df['Position'] == 'pitch']
    
    mae_df = results_df.groupby(['Model', 'Year', 'Position'])['Abs Diff'].mean().reset_index()
    mae_df.columns = ['Model', 'Year', 'Position', 'MAE']
    mae_df_bat = mae_df[mae_df['Position'] == 'bat']
    mae_df_pitch = mae_df[mae_df['Position'] == 'pitch']
    
    transformers_models = ['T24r', 'L24r']
    
    pitch_compare_rmse = baseball_tstats(rmse_df_pitch, ['erm', 'NPMLE', "robbins", "mle"], transformers_models,'RMSE','greater')
    bat_compare_rmse = baseball_tstats(rmse_df_bat, ['erm', 'NPMLE', "robbins", "mle"], transformers_models,'RMSE','greater')
    pitch_compare_mae = baseball_tstats(mae_df_pitch, ['erm', 'NPMLE', "robbins", "mle"], transformers_models,'MAE','greater')
    bat_compare_mae = baseball_tstats(mae_df_bat, ['erm', 'NPMLE', "robbins", "mle"], transformers_models,'MAE','greater')
    
    print("RMSE Pitch", pitch_compare_rmse)
    print("RMSE Bat", bat_compare_rmse)
    print("MAE Pitch", pitch_compare_mae)
    print("MAE Bat", bat_compare_mae)
    # Plotting RMSE
    ratio_all = []
    for model in ['erm', 'NPMLE', 'T24r', 'L24r']:
        ratio = baseball_compare_stats(rmse_df_bat, "batting", model, 'mle', 'RMSE')
        ratio.name = model
        ratio_all.append(ratio)
    ratio_melt = pd.concat(ratio_all, axis = 1).melt(var_name='Model', value_name = 'RMSE ratio')
    sns.violinplot(ratio_melt, x = 'Model', y = 'RMSE ratio')
    plt.axhline(y = 1.00, color = 'r', linestyle='-')
    
    ratio_all = []
    for model in ['erm', 'NPMLE', 'T24r', 'L24r']:
        ratio = baseball_compare_stats(rmse_df_pitch, "pitch", model, 'mle', 'RMSE')
        ratio.name = model
        ratio_all.append(ratio)
    ratio_melt = pd.concat(ratio_all, axis = 1).melt(var_name='Model', value_name = 'RMSE ratio')
    sns.violinplot(ratio_melt, x = 'Model', y = 'RMSE ratio')
    plt.axhline(y = 1.00, color = 'r', linestyle='-')
    
    ratio_all = []
    for model in ['erm', 'NPMLE', 'T24r', 'L24r']:
        ratio = baseball_compare_stats(mae_df_bat, "batting", model, 'mle', 'RMSE')
        ratio.name = model
        ratio_all.append(ratio)
    ratio_melt = pd.concat(ratio_all, axis = 1).melt(var_name='Model', value_name = 'RMSE ratio')
    sns.violinplot(ratio_melt, x = 'Model', y = 'RMSE ratio')
    plt.axhline(y = 1.00, color = 'r', linestyle='-')
    
    ratio_all = []
    for model in ['erm', 'NPMLE', 'T24r', 'L24r']:
        ratio = baseball_compare_stats(mae_df_pitch, "pitch", model, 'mle', 'RMSE')
        ratio.name = model
        ratio_all.append(ratio)
    ratio_melt = pd.concat(ratio_all, axis = 1).melt(var_name='Model', value_name = 'RMSE ratio')
    sns.violinplot(ratio_melt, x = 'Model', y = 'RMSE ratio')
    plt.axhline(y = 1.00, color = 'r', linestyle='-')
