import numpy as np
import pandas as pd
import torch
import argparse
import os
import sys
from tqdm import tqdm
import pickle
import math
from eb_train import EBTransformer
import time
from eb_arena_mapper import load_model_dict
from algo_helpers import robbins, erm, npmle, fixed_grid_npmle

# This is real data, so we don't have to set seed manually. 

# TODO: fill in the blanks for getting hockey data. 

# Helper function in reading hockey data.
# Creds: original Poisson repo that exists since NPMLE days.
def read_hock_position(filename,position = None):
    df = pd.read_csv(filename)
    dfx = df[['Unnamed: 1', 'Scoring.1', 'Unnamed: 4']] #Goals
    dfx = dfx[1:]; # Delete first spurious row
    dfx.columns = ['name', 'goals', 'position']
    dfx.set_index('name')
    if position=="winger":
        dfx = dfx.loc[(dfx["position"] == "LW") + (dfx["position"] == "RW")+(dfx["position"] == "W")]
    if position == "center":
        dfx = dfx.loc[(dfx["position"] == "C")]
    if position == "defender":
        dfx = dfx.loc[(dfx["position"] == "D")]
    del dfx["position"];
    return dfx

# Allows for even more general processing of the hockey dataset (the dataset we scrap from the internet is probably too messy). 
def process_hockey_csv(filename, position = None):
    df = pd.read_csv(filename).reset_index()
    target_colname = df.iloc[0]
    df = df[1:]
    df.columns = target_colname
    if 0 in df.columns:
        df = df.drop(columns = [0])
    df = df.rename(columns={"-9999": "ID"})
    if "ID" in df.columns:
        df["Player"] = df.apply(lambda x: x["Player"] + "\\" + x["ID"], axis = 1)
    team_name = "Team" if "Team" in df.columns else "Tm"
    # Next, consider the duplicates.
    dup_keywords = ['TOT'] + [str(i)+'TM' for i in range(2, 10)]
    dups = df["Rk"].duplicated(False)
    df = df[(~dups) | ((dups) & df[team_name].isin(dup_keywords))]
    dfx = df[["Player", "G", "Pos"]]
    dfx.columns = ['name', 'goals', 'position']
    dfx = dfx.set_index('name')
    if position=="winger":
        dfx = dfx.loc[(dfx["position"] == "LW") + (dfx["position"] == "RW")+(dfx["position"] == "W")]
    if position == "center":
        dfx = dfx.loc[(dfx["position"] == "C")]
    if position == "defender":
        dfx = dfx.loc[(dfx["position"] == "D")]
    del dfx["position"];

    # Let us also support multiple positions. 
    return dfx

def hockey_data(file1, file2, position = None):
    df1x = process_hockey_csv(file1, position)
    df2x = process_hockey_csv(file2, position)
    df = pd.merge(df1x,df2x, on='name', suffixes=('_past','_future'))
    G = np.asarray(df[['goals_past', 'goals_future']].astype('int32'))
    gpast = G[:,0]
    gfut = G[:,1]
    return gpast, gfut

def main():
    print(sys.argv)
    parser = argparse.ArgumentParser(description='eb_arena')
    parser.add_argument('--model', help='name of model to use')
    parser.add_argument('--llmap_out', help='output dir passed by LLMapReduce')
    parser.add_argument('--dbg_file', help='path to debug stuff')
    parser.add_argument('--pos', default=None)
    parser.add_argument('--data_dir', help='path we want to extract our data from')
    parser.add_argument('--prev_year', help='year we have the samples')
    parser.add_argument('--next_year', help='year we want to predict')
    args = parser.parse_args()
    args.mdl_name = args.model
    args.prev_season_file = os.path.join('datasets/hockey', 'season_{}.csv'.format(args.prev_year))
    args.next_season_file = os.path.join('datasets/hockey', 'season_{}.csv'.format(args.next_year))
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    # Next, jump straight to benchmarks. 

    benchmarks = {}

    benchmarks['mle'] = lambda x : x
    benchmarks['robbins'] = robbins
    benchmarks["erm"] = erm
    benchmarks["npmle"] = npmle
    benchmarks["fixed_grid_npmle"] = fixed_grid_npmle
    benchmarks["worst_prior"] = "worst_prior" # Okay maybe this is not too too well-defined in the other file. Will be a TODO. 

    model = args.model
    mdl_name = ""
    if model in benchmarks:
        mdl_name =  model
        model = benchmarks[model]
        model_args = None
    else:
        mdl_name =  model.split("/")[-1]
        model = load_model_dict(model, args.device)['model']
        model.args.device = args.device # Makes it possible to evaluate on CPU. 
        model_args = model.args

    # Now try to get the hockey data. 
    inputs, labels = hockey_data(args.prev_season_file, args.next_season_file, args.pos)
    inputs = torch.from_numpy(inputs[np.newaxis, :, np.newaxis]).to(args.device).float()
    labels = torch.from_numpy(labels[np.newaxis, :, np.newaxis]).to(args.device).float()
    # Then predict?
    with torch.inference_mode():
        start_time = time.time()
        outputs = model(inputs).float()
        end_time = time.time()
        inference_time = end_time - start_time

    mse = torch.mean((outputs - labels) ** 2)
    # Finally dump the output? 
    output = {"args": model_args, "input": inputs.detach().cpu().numpy(), "label": labels.detach().cpu().numpy(), 
              "output": outputs.detach().cpu().numpy(), "time": inference_time, "pos": args.pos}
    if args.pos is not None:
        out_name = f"hockey_results/{args.pos}/{mdl_name}_{args.pos}_{args.prev_year}_{args.next_year}.pkl"
    else:
        out_name = f"hockey_results/{mdl_name}_{args.prev_year}_{args.next_year}.pkl"
    print(mse)
    print(out_name)
    os.makedirs("hockey_results" , exist_ok = True)
    with open(out_name, "wb") as f:
        pickle.dump(output, f)

if __name__ == "__main__":
    main()
