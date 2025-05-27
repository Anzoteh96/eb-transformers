import numpy as np
import pandas as pd
import torch
import argparse
import os
import sys
from tqdm import tqdm
import pickle
import math
import time
from eb_transformer import EBTransformer
from eb_arena_mapper import load_model_dict
from algo_helpers import robbins, erm, npmle, fixed_grid_npmle

def get_baseball_dataset(filename, train_ratio):
    """
        Args:
            filename: name of df that contains the games
            (Assumption: this df should contain columns game_id, date, stat, player_id; stat = hit number)
            threshold_ratio
    """
    bat_yr = pd.read_csv(filename)
    assert(0 < train_ratio < 1), "train ratio must be strictly between 0 and 1"
    L = bat_yr['game_id'].unique().shape[0]
    num_train_games = int(L * train_ratio)
    dates = np.sort(bat_yr['date'].unique())
    i = 0
    while (bat_yr[bat_yr['date'] < dates[i]]['game_id'].unique().shape[0] < num_train_games):
        i += 1
    threshold_date = dates[i]
    hit_X = bat_yr[(bat_yr['date'] < threshold_date)]
    hit_Y = bat_yr[(bat_yr['date'] >= threshold_date)]
    hit_X_count = hit_X.groupby(['player_id']).count()['stat']
    hit_Y_count = hit_Y.groupby(['player_id']).count()['stat']
    # Some players have zero-count. Do we want to keep both of them?
    hit_XY = pd.merge(pd.DataFrame(hit_X_count), pd.DataFrame(hit_Y_count), on = 'player_id', how = 'outer').fillna(0)
    # from IPython import embed; embed()
    return hit_XY['stat_x'].values, hit_XY['stat_y'].values

def baseball_data(file1, file2):
    """
        Returns: 
            gpast: numpy array
            gfut: numpy array
    """
    df1 = get_baseball_dataset(file1)
    df2 = get_baseball_dataset(file2)

    # Now we don't really have to preprocess it. 
    df = pd.merge(df1, df2, on = 'playerID')
    gpast = df['stat_x']
    gfut = df['stat_y']

    return gpast.values, gfut.values

# Can we also concatenate a few past "fit" values together? 
def prev_fit_data(target_year, num_fit, position, train_ratio):
    dataset_dir = 'datasets/baseball'
    inputs_all = []
    labels_all = []
    file_prefix = "batting" if position == "bat" else "pitching"
    for i in range(1, num_fit + 1):
        prev_year = target_year - i
        season_file = os.path.join(dataset_dir, '{}_hitcount_{}.csv'.format(file_prefix, prev_year))
        print(season_file)
        inputs, labels = get_baseball_dataset(season_file, train_ratio)
        inputs_all.append(inputs)
        labels_all.append(labels)
    inputs_all = np.concatenate(inputs_all)
    labels_all = np.concatenate(labels_all)

    return inputs_all, labels_all

def main():
    print(sys.argv)
    parser = argparse.ArgumentParser(description='eb_arena')
    parser.add_argument('--model', help='name of model to use')
    parser.add_argument('--pos', default=None) # default: either bat or pitch. 
    parser.add_argument('--data_dir', help='path we want to extract our data from')
    parser.add_argument('--year', help='year of interest')
    parser.add_argument('--out_dir', help='directory where we output our predictions')

    args = parser.parse_args()
    args.mdl_name = args.model

    assert(args.pos in ["bat", "pitch"]), "only batting and pitching positions are supported"

    file_prefix = "batting" if args.pos == "bat" else "pitching"
    args.season_file = os.path.join('datasets/baseball', '{}_hitcount_{}.csv'.format(file_prefix, args.year))

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
        model_args = None
        mdl_name =  model.split("/")[-1]
        model = load_model_dict(model, args.device)['model']
        if args.device == 'cuda':
            model = model.cuda()
        if isinstance(model, EBTransformer): 
            model.args.device = args.device
            model_args = model.args

    # Now try to get the baseball data. 
    if args.pos == "bat":
        train_ratio = 0.50 # Old: 1/5
    elif args.pos == "pitch":
        train_ratio = 0.50 # Old: 1/6
    inputs, labels = get_baseball_dataset(args.season_file, train_ratio)
    inputs = torch.from_numpy(inputs[np.newaxis, :, np.newaxis]).to(args.device).float()
    labels = torch.from_numpy(labels[np.newaxis, :, np.newaxis]).to(args.device).float()
    # Then predict?
    # from IPython import embed; embed()
    with torch.inference_mode():
        start_time = time.time()
        outputs = model(inputs).float() * ((1 - train_ratio) / train_ratio)
        end_time = time.time()
        inference_time = end_time - start_time

    mse = torch.mean((outputs - labels) ** 2)
    # Finally dump the output? 
    output = {"args": model_args, "input": inputs.detach().cpu().numpy(), "label": labels.detach().cpu().numpy(), 
              "output": outputs.detach().cpu().numpy(), "time": inference_time}
    out_name = f"{args.out_dir}/{mdl_name}_{args.pos}_{args.year}.pkl"
    os.makedirs(args.out_dir, exist_ok=True)
    print(mse.item())
    print(out_name)
    with open(out_name, "wb") as f:
        pickle.dump(output, f)

if __name__ == "__main__":
    main()

