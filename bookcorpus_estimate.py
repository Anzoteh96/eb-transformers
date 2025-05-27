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
from eb_train import EBTransformer
from eb_arena_mapper import load_model_dict
from algo_helpers import robbins, erm, npmle, fixed_grid_npmle

def main():
    print(sys.argv)
    parser = argparse.ArgumentParser(description='bookcorpus')
    parser.add_argument('--model', help='name of model to use')
    parser.add_argument('--dataset_dir', help='directory of dataset')
    parser.add_argument('--filename', help = 'name of file to do')
    parser.add_argument('--tokenizer', help = 'tokenizer')
    parser.add_argument('--out_dir', help='where to store those outputs')
    
    args = parser.parse_args()
    assert args.tokenizer in ['countvec', 'tiktoken']

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
        model.args.device = args.device
        model_args = model.args

    file_root = args.filename.split(".")[0]
    full_filename = os.path.join(args.dataset_dir, args.filename + ".csv")
    df = pd.read_csv(full_filename)
    inputs = df['Frequency_x'].values
    labels = df['Frequency_y'].values
    len_X = df['len_X'].mean()
    len_Y = df['len_Y'].mean()
    ratio = len_Y / len_X
    inputs = torch.from_numpy(inputs[np.newaxis, :, np.newaxis]).to(args.device).float()
    labels = torch.from_numpy(labels[np.newaxis, :, np.newaxis]).to(args.device).float()

    with torch.inference_mode():
        start_time = time.time()
        outputs = model(inputs).float() * len_Y / len_X
        end_time = time.time()
        inference_time = end_time - start_time

    mse = torch.mean((outputs - labels) ** 2)

    output = {"args": model_args, "input": inputs.detach().cpu().numpy(), "label": labels.detach().cpu().numpy(), 
              "output": outputs.detach().cpu().numpy(), "time": inference_time}
    # Let's do dir / model, lol. 
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = os.path.join(out_dir, f"{mdl_name}.pkl")
    print(mse.item())
    print(out_name)
    with open(out_name, "wb") as f:
        pickle.dump(output, f)

if __name__ == "__main__":
    main()
