import argparse
import os
import pickle
import sys
from collections import defaultdict
from tqdm import tqdm

import torch

from eb_arena_mapper import TorchCpuUnpickler
from algo_helpers import erm, npmle


def _get_key(x, m):
    return tuple(int(i) for i in (x + m).flatten())


def freqpnlusm(x, m):
    # Round x to integers without modifying x in place
    x_rounded = torch.round(x).to(torch.int64)
    B, N, D = x_rounded.shape
    ret = torch.zeros(B, N, 1)

    for b in tqdm(range(B)):
        counter_dict = defaultdict(lambda: 0)
        for n in range(N):
            counter_dict[_get_key(x_rounded[b, n, :], 0)] += 1

        for n in range(N):
            k = _get_key(x_rounded[b, n, :], m)
            ret[b, n, 0] = counter_dict[k]
    return ret


def freqn(x):
    return freqpnlusm(x, 0)


def freqnplus1(x):
    return freqpnlusm(x, 1)


attributes = {"erm": erm, "freqn": freqn, "freqnplus1": freqnplus1}  # , "npmle": npmle}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eb_arena")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--attribute", type=str, default="erm")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    dir_out = os.path.join(args.output, args.attribute)
    out_f = os.path.join(dir_out, os.path.basename(args.input))
    os.makedirs(dir_out, exist_ok=True)
    attribute = attributes[args.attribute]
    if not os.path.exists(out_f):
        with open(args.input, "rb") as f:
            data = TorchCpuUnpickler(f).load()["x"]
            out = attribute(data)
            with open(
                out_f,
                "wb",
            ) as f:
                pickle.dump(out, f)
