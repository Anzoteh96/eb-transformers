import pandas as pd 
import numpy as np
import torch
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import argparse

if __name__ == "__main__":
    print(sys.argv)
    parser = argparse.ArgumentParser(description="bookcorpus_reduce")
    #parser.add_argument("--results_dir", type=str, help='where are we getting our results?')
    #parser.add_argument("--store_dir", type=str, help='where are we storing our results summary?')
    parser.add_argument("--tokenizer", type=str, help="tokenizer")
    args = parser.parse_args()
    #main_folder = args.results_dir
    tokenizer = args.tokenizer # Just hardcode this lol

    main_folder = "bookcorpus_results_{}".format(tokenizer)
    all_folders = sorted(os.listdir(main_folder))
    rmse_all = {}
    mae_all = {}
    time_all = {}
    model_sets = ["mle", "robbins", "fixed_grid_npmle", "erm", "eb_2024_12_23-11_42_qex.pkl", 
                  "eb_2025_04_30-20_55_sBb.pkl",  "eb_2025_05_05-08_42_lSX.pkl", 'eb_2025_09_29-13_10_L90.pkl', 'eb_2025_09_25-12_21_zZ3.pkl']
    for folder in tqdm(all_folders):
        full_folder = os.path.join(main_folder, folder)
        rmse_all[folder] = {}
        mae_all[folder] = {}
        time_all[folder] = {}
        for pkl_file in os.listdir(full_folder):
            model_fname = pkl_file[:-4]
            if not(model_fname in model_sets):
                continue
            full_path = os.path.join(full_folder, pkl_file)
            # print(full_path)
            results = pickle.load(open(full_path, "rb"))
            label = results["label"]
            # Maybe do x->max(0, x)?
            output = np.maximum(results["output"], 0)
            if results["args"] is not None:
                #prefix = "L" if "att_activ" in results["args"] and results["args"].att_activ == "linear" else "T"
                #suffix = "r" if results["args"].theta_max_israndom else "f"
                #model_name = "{}{}{}".format(prefix, results["args"].layers, suffix)
                # print(model_fname, model_name)
                model_name = model_fname
            elif model_fname == "fixed_grid_npmle":
                model_name = "NPMLE"
            else:
                model_name = model_fname
            # Just in case some of the outputs are negative. 
            output = np.maximum(output, 0.0)
            rmse = np.sqrt(np.mean(np.square(label - output)))
            mae = np.mean(np.abs(label - output))
            rmse_all[folder][model_name] = rmse
            mae_all[folder][model_name] = mae
            time_all[folder][model_name] = results["time"]
    
    with open("bookcorpus_rmse_allv2_{}.pkl".format(tokenizer), "wb") as f:
        pickle.dump(rmse_all, f)
    with open("bookcorpus_mae_allv2_{}.pkl".format(tokenizer), "wb") as f:
        pickle.dump(mae_all, f)
