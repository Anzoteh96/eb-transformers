import argparse
import glob
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import torch

from eb_arena_mapper import TorchCpuUnpickler

# Proves the activation linearly


def standarize_inputs(experiment_list):
    print(experiment_list[0].shape)
    if not isinstance(experiment_list, torch.Tensor):
        results = torch.stack(experiment_list)
    else:
        results = experiment_list
    results = results.view(-1, results.size(-1))
    return results.numpy()


def get_r2_and_coefs(x, y):
    reg = sklearn.linear_model.LinearRegression()
    reg.fit(x, y)
    return reg.score(x, y), reg.coef_


ATTRIBUTE_COMPUTERS = {
    "erm": lambda d: d["erm"],
    "freqn": lambda d: d["freqn"],
    "freqnplus1": lambda d: d["freqnplus1"],
    "ratio": lambda d: d["freqnplus1"] / d["freqn"],
    "robins": lambda d: (d["x"] + 1) * d["freqnplus1"] / (d["freqn"]),
}

TARGETS = [
    "y",
    "y_hat",
    "erm",
    "freqn",
    "freqnplus1",
    "ratio",
    "robins",
    "x",
]  # , "npmle"]

FEATURES = ["out", "mlp_step", "attn_step", "activations"]
FEATURE_NAMES = ["decoded activations", "MLP step", "Attention step", "activations"]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="probe_activations")
    parser.add_argument("--inputdir", type=str, required=True)
    parser.add_argument("--outputname", type=str, required=True)
    parser.add_argument(
        "--attributedir", type=str, default="./missing_attributes_neural"
    )
    parser.add_argument("--add_corr", type=bool, default=True)

    args = parser.parse_args()

    files = sorted(glob.glob(args.inputdir + "/*.pkl"))
    # prepare y and x
    num_layers = len(files)  # updated below
    default_seeds = [
        715393164,
        84670904,
        872369255,
        949622851,
        805315292,
        707275517,
        636849157,
        336928280,
        451453725,
        216388192,
        916276168,
        136545221,
        98052060,
        648086751,
        366781926,
        904876033,
        988719428,
        262948609,
        914821058,
        289783310,
        52278999,
        845605560,
        308999736,
        296306123,
        777841764,
        316929251,
        699247120,
        718514585,
        522280770,
        915675869,
        984775207,
        72025026,
        10681500,
        403941473,
        510146937,
        436808402,
        95597108,
        331577249,
        962970448,
        411773425,
        16759924,
        49952492,
        432079842,
        171277678,
        705224444,
        914415652,
        425835549,
        897944141,
        960917033,
        697199448,
        639936328,
        727484960,
        80043846,
        396846444,
        94935248,
        751387787,
        659129257,
        406409069,
        961954982,
        828007425,
        512549369,
        420563412,
        331264023,
        716427143,
        926964506,
        482820150,
        796625975,
        538770909,
        872442977,
        838804621,
        660981320,
        929117015,
        108902176,
        324200685,
        695409542,
        952126954,
        410195385,
        156405,
        549755774,
        193609983,
        190455298,
        424172299,
        315067586,
        690901466,
        471861233,
        343411039,
        785603900,
        931751832,
        588660688,
        220045647,
        452024363,
        509611213,
        898829288,
        52539626,
        278055446,
        310279056,
        20395159,
        852649769,
        970795307,
        141683866,
    ]
    d = {}
    seeds = []
    available_attributes = list(os.listdir(args.attributedir))
    r2s = defaultdict(list)
    coefs = defaultdict(list)
    for layer in range(num_layers):
        print("layer", layer)
        print("------------------------------------------")
        filename = os.path.join(args.inputdir, f"{layer}.pkl")
        with open(filename, "rb") as f:
            result = TorchCpuUnpickler(f).load()
            for k in result:
                if "seed" in k or result[k][0] is None:
                    if "seed" not in d:
                        print("not available", k, layer)
                    continue
                print(k)
                d[k] = standarize_inputs(result[k])
            if "seed" in result:
                seeds = result["seed"]
            else:
                seeds = default_seeds
        for attribute in available_attributes:
            if attribute not in d:
                attribute_dir = os.path.join(args.attributedir, attribute)
                d[attribute] = []
                for seed in seeds:
                    filename = os.path.join(attribute_dir, f"{seed}.pkl")
                    with open(filename, "rb") as f:
                        d[attribute].append(TorchCpuUnpickler(f).load())
                print("attribute", attribute)
                d[attribute] = standarize_inputs(d[attribute])
        for attribute, comp in ATTRIBUTE_COMPUTERS.items():
            print("attribute", attribute)
            d[attribute] = comp(d)
        for x in FEATURES:
            for y in TARGETS:
                if x not in d or y not in d:
                    print("skipping", x, y, layer)
                    continue
                print(f"regressing {y} on {x}")
                r2, coef = get_r2_and_coefs(d[x], d[y])
                r2s[f"{x}->{y}"].append(r2)
                coefs[f"{x}->{y}"].append(coef)
        print("------------------------------------------")

    output = {"r2s": r2s, "coefs": coefs}
    print(r2s)
    if args.add_corr:
        variables_2_corr = TARGETS + ["x"]
        print("corr")
        for k in variables_2_corr:
            print(k, d[k].shape)
        df = pd.DataFrame({k: d[k].flatten() for k in variables_2_corr})
        output["corr"] = df.corr()
    results_path = os.path.join(args.outputname, "results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(output, f)
    for k in r2s.keys():
        image_name = os.path.join(args.outputname, f"{k}.png")
        plt.plot(list(range(num_layers))[-len(r2s[k]) :], r2s[k])
        plt.axhline(0, color="r")
        plt.axhline(1, color="r")
        plt.xlabel("Layer")
        plt.ylabel("R2")
        plt.grid()
        plt.title(f"{num_layers - 1} {k}")
        plt.savefig(image_name)
        plt.clf()
    print(":((((((((((")
    for f, fname in zip(FEATURES, FEATURE_NAMES):
        for t in TARGETS:
            print(".........................")
            print(f, t, f"{f}->{t}" in r2s)
            print(list(range(num_layers))[-len(r2s[f"{f}->{t}"]) :])
            print(r2s[f"{f}->{t}"])
            print(".........................")
            plt.plot(
                list(range(num_layers))[-len(r2s[f"{f}->{t}"]) :],
                r2s[f"{f}->{t}"],
                label=f"{t}",
            )
        plt.xlabel("Layer")
        plt.ylabel("r2")
        plt.axhline(0, color="red")
        plt.axhline(1, color="red")
        plt.axvline((num_layers - 1) // 2, color="red", label="seccond attn head")
        plt.title(f"{num_layers - 1} r2 {fname}")
        plt.grid()
        plt.legend()
        image_name = os.path.join(args.outputname, f"{f} all targets.png")
        plt.savefig(image_name)
        image_name = f"{args.outputname}-{f}.png"
        plt.savefig(image_name)
        plt.clf()
