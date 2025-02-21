import io
import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from eb_train import EBTransformer, get_n_params


class TorchCpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def load_model_dict(filename, device):
    file = open(filename, "rb")
    if device == "cpu":
        return TorchCpuUnpickler(file).load()
    elif device == "cuda":
        return pickle.load(file)
    else:
        raise ValueError("Unknown device")


act_map = {"gelu": "g", "relu": "r"}


def get_model_plot_name(
    output_file_name,
    eb_hscan_path="./eb_hscan/",
    args=["layers", "heads", "dmodel", "activation"],
):
    if output_file_name == "identity":
        return "MLE"
    elif ".pkl" == output_file_name[-4:]:
        mdl_dict_path = os.path.join(eb_hscan_path, output_file_name)
        mdl_dict = load_model_dict(mdl_dict_path, "cpu")
        n_params = get_n_params(mdl_dict["model"])
        s = f"{math.floor(n_params / 1000)}.{round(n_params / 100) - 10 * math.floor(n_params / 1000)}K"
        for arg in args:
            val = getattr(mdl_dict["args"], arg)
            if arg == "layers":
                s += "-L"
                s += f"{val}"
            elif arg == "dmodel":
                s += "-d"
                s += f"{val}"
            elif arg == "train_steps":
                s += "-S"
                s += f"{round(val / 1000)}K"

        # print(mdl_dict["args"])
        # n_layers = mdl_dict["args"].layers
        # imbed_dim = mdl_dict["args"].dmodel
        # train_steps = mdl_dict["args"].train_steps
        # mdl_name = f"{n_params}-{n_layers}-{imbed_dim}-{train_steps}"
        return s  # mdl_name
    else:
        return output_file_name


# This basically compares between two regret / MSE stats (e.g. different priors, or different number of samples).
def plot(file1, file2, xlabel, ylabel, key, threshold=None, savefile=None):
    with open(file1, "rb") as f1:
        with open(file2, "rb") as f2:
            lst1 = pickle.load(f1)[0.0]
            lst2 = pickle.load(f2)[0.0]
            dict1 = dict([(item["model"], item[key]) for item in lst1])
            dict2 = dict([(item["model"], item[key]) for item in lst2])
            lst_plot_x = []
            lst_plot_y = []
            lst_model = []
            for key in dict1:
                if key in ["robbins", "erm", "mle"]:
                    continue
                if key in dict2:
                    if (threshold is not None) and np.sqrt(dict2[key]) > threshold:
                        continue
                    lst_model.append(get_model_plot_name(key))
                    lst_plot_x.append(np.sqrt(dict1[key]))
                    lst_plot_y.append(np.sqrt(dict2[key]))
            plt.scatter(lst_plot_x, lst_plot_y)
            plt.xlabel(xlabel)  #
            plt.ylabel(ylabel)  #
            for mod, x, y in zip(lst_model, lst_plot_x, lst_plot_y):
                plt.annotate(mod, (x, y), fontsize=5)
    if savefile is not None:
        plt.savefig(savefile)
    plt.clf()


# E.g. plot(file1 = 'regret512.pkl', file2 = 'regret1024.pkl', xlabel = 'Neural prior sqrt mse (seqlen=512)', ylabel = 'Neural prior sqrt regret (seqlen=1024)', key='mse', threshold=0.6),

# possible args:
# dmodel=32, dinput=1, batch=192, theta_max=50, seqlen=512, step=0.5, layers=18, heads=4, activation='gelu', train_steps=15000, train_lr=0.007, train_lr_epoch=300, train_lr_gamma=0.9, uniform_prior=False, nohist_thetas=True, keep_stdout=False, tqdm_disable=True, device='cuda', dtype=torch.float32, fname_prefix='eb_2023_08_14-14_17_7ws'

if __name__ == "__main__":
    print(get_model_plot_name("eb_2023_08_14-14_17_7ws.pkl"))
    print(get_model_plot_name("identity"))
    print(get_model_plot_name("ERM"))
