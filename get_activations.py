import argparse
import os
import pickle
import random
from collections import defaultdict

import torch

from eb_arena_mapper import erm, gen_batch_from_seed, load_model_dict, set_seed
from eb_train import EBTransformer

# This Dumps The activations of the model at each layer for a given seed.


def get_layer_outputs(seeds, layer, model, args):
    activations = defaultdict(list)
    for seed in seeds:
        args.seed = seed
        activations["seed"].append(seed)
        inputs, labels = gen_batch_from_seed(args)
        with torch.inference_mode():
            y_hat = model(inputs)
            y_hat = model.forward(inputs)
            act = model.get_layer_activation(inputs, layer)
        activations["x"].append(inputs)
        activations["y"].append(labels)
        activations["y_hat"].append(y_hat)
        for k in act:
            activations[k].append(act[k])
    return activations


def save_layer_outpus(seeds, layer, model, args, out_file):
    sample = get_layer_outputs(seeds, layer, model, args)
    with open(out_file, "wb") as f:
        pickle.dump(sample, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eb_arena")
    # 1. model_path
    parser.add_argument("--model", help="name of model to use")
    # 2. prior
    parser.add_argument(
        "--prior", type=str, default="neural", help="prior we use for training"
    )
    parser.add_argument(
        "--prior_file", type=str, default=None, help="file we load prior from"
    )
    parser.add_argument(
        "--alpha", type=float, default=None, help="alpha param for dirichlet"
    )
    parser.add_argument(
        "--dirich_prob", type=float, default=None, help="Dirichlet mixture probability"
    )
    parser.add_argument(
        "--uniform_percentage", type=float, default=0.0, help="percentage that"
    )
    parser.add_argument(
        "--dinput", type=int, default=1, help="dimensionality of inputs and labels"
    )
    parser.add_argument(
        "--theta_max", type=float, default=50, help="limit on the support of the prior"
    )
    parser.add_argument(
        "--same_prior",
        type=bool,
        default=False,
        help="output dir passed by LLMapReduce",
    )

    # Let's also add randomness for thetamax.
    parser.add_argument(
        "--theta_max_israndom", action="store_true", help="are thetamax random?"
    )
    parser.add_argument(
        "--uniform_prior",
        action="store_true",
        help="Simplistic generation where thetas are sampled from a uniform prior",
    )
    parser.add_argument(
        "--worst_prior", action="store_true", help="Trying out worst prior"
    )
    parser.add_argument(
        "--seqlen", type=int, default=512, help="maximal length of the input"
    )
    parser.add_argument("--batch", type=int, default=100, help="number of batches")

    # 3. seed
    parser.add_argument("--seed", type=int, default=10)
    # 4. num_samples
    parser.add_argument("--num_samples", type=int, default=5000)
    # 6. output_dir
    parser.add_argument("--output_dir", type=str, default="output")

    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"
    args.dtype = torch.float32
    model = args.model
    mdl_name = model.split("/")[-1]
    model_dict = load_model_dict(model, args.device)
    model = model_dict["model"]
    model.args.device = args.device

    set_seed(args.seed)

    data_seeds = [
        random.randrange(1000000000) for i in range(args.num_samples // args.batch)
    ]
    print("data_seeds", data_seeds)
    mdl_dir = os.path.join(args.output_dir, mdl_name)
    os.makedirs(mdl_dir, exist_ok=True)
    for layer in range(model.args.layers + 1):
        out_file = os.path.join(mdl_dir, f"{layer}.pkl")
        save_layer_outpus(data_seeds, layer, model, args, out_file)
