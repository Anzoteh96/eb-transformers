import argparse
import glob
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim

from eb_arena_mapper import TorchCpuUnpickler
from eb_train import EBTransformer

# Probe model activations with a decoder


class EBDecoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        factory_kwargs = {"device": args.device, "dtype": args.dtype}
        self.args = args
        self.mask = None

        if args.decoding_layer_norm:
            self.norm3 = torch.nn.LayerNorm(args.dmodel, **factory_kwargs)

        self.decoder = torch.nn.Linear(args.dmodel, args.targetsdim, **factory_kwargs)
        return None

    def forward(self, inputs):
        # Add dummy dimension to help avoid killing input's magnitude by the LayerNorm.
        # inputs is BxTxd_in. inputs_aux is BxTx(d_in+1).
        inputs = inputs.to(self.args.device)
        if self.args.decoding_layer_norm:
            emb = self.norm3(inputs)
        out = self.decoder(emb)

        gelu = torch.nn.GELU()
        out = gelu(out)
        return out

    def eval_loss(self, inputs, labels):
        inputs = inputs.to(self.args.device)
        labels = labels.to(self.args.device)
        out = self.forward(inputs)
        loss = ((out - labels) ** 2).mean()
        return loss

    def eval_r2(self, inputs, labels):
        inputs = inputs.to(self.args.device)
        labels = labels.to(self.args.device)
        mean_dims = [i for i in range(len(labels.shape) - 1)]
        out = self.forward(inputs)
        print("out shape", out.shape)
        loss = ((out - labels) ** 2).mean(dim=mean_dims)
        var = labels.var(dim=mean_dims)
        loss = 1 - loss / (var + 1e-11)
        print("loss", loss)
        return loss


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


def standarize_inputs(experiment_list):
    print(experiment_list[0].shape)
    if not isinstance(experiment_list, torch.Tensor):
        results = torch.stack(experiment_list)
    else:
        results = experiment_list
    results = results.view(-1, results.size(-1))
    return results.numpy()


FEATURES = ["out", "mlp_step", "attn_step", "activations"]
FEATURE_NAMES = ["decoded activations", "MLP step", "Attention step", "activations"]
FEATURE_NAME_MAPPING = {f: fn for f, fn in zip(FEATURES, FEATURE_NAMES)}


def train(
    model,
    inputs,
    outputs,
    lr=0.05,
    max_epochs=int(1e4),
    grad_norm_threshold=1e-7,
    epoch_size=int(1e2),
):
    """
    Train a single-layer perceptron using Adam optimizer with decaying learning rate.

    Args:
        model (nn.Module): The perceptron model.
        inputs (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        outputs (torch.Tensor): Target tensor of shape (batch_size, output_dim).
        loss_fn (nn.Module): Loss function (e.g., MSELoss).
        lr (float): Initial learning rate for the optimizer.
        max_epochs (int): Maximum number of training epochs.
        grad_norm_threshold (float): Threshold for gradient norm to stop training early.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.9
    )  # Decay LR by 10% every epoch

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        # Compute loss and backpropagate
        loss = model.eval_loss(inputs, outputs)
        loss.backward()

        # Compute total gradient norm
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm**0.5

        # Check stopping criterion
        if total_grad_norm < grad_norm_threshold:
            print(
                f"Stopping early at epoch {epoch}, gradient norm {total_grad_norm:.6f}"
            )
            break

        # Update weights and decay learning rate
        optimizer.step()

        # Print progress
        if epoch % epoch_size == 0 or epoch == max_epochs - 1:
            scheduler.step()
            print("----------------------------------")
            print(
                f"Epoch {epoch}, Loss: {loss.item():.6f}, Grad Norm: {total_grad_norm:.6f}"
            )
            with torch.no_grad():
                r2 = model.eval_r2(inputs, outputs)
                print("r2", r2)
                print("TARGETS", TARGETS)
            print("----------------------------------")

    print("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train_layer_decoder")

    parser.add_argument("--inputactivations", type=str, required=True)
    parser.add_argument("--outputname", type=str, required=True)
    parser.add_argument("--attributedir", type=str, required=True)
    parser.add_argument("--feature", type=str, default="activations")
    parser.add_argument("--model_path", type=str, required=True)
    decoder_args = parser.parse_args()

    # Load Transformer to get decoder args args
    with open(decoder_args.model_path, "rb") as f:
        model_dict = TorchCpuUnpickler(f).load()
    args = model_dict["model"].args
    # Set Device properly
    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"
    del model_dict

    # Load data
    with open(decoder_args.inputactivations, "rb") as f:
        d = TorchCpuUnpickler(f).load()

    for k in d.keys():
        if "seed" in k or d[k][0] is None:
            continue
        print("standarizing", k)
        d[k] = standarize_inputs(d[k])

    # Load attributes

    available_attributes = list(os.listdir(decoder_args.attributedir))
    for attribute in available_attributes:
        print("attribute", attribute)
        if attribute not in d and attribute in TARGETS:
            attribute_dir = os.path.join(decoder_args.attributedir, attribute)
            fs = glob.glob(os.path.join(attribute_dir, "*.pkl"))
            assert len(fs) == 1
            f = fs[0]
            d[attribute] = []
            with open(f, "rb") as f:
                d[attribute] = standarize_inputs(TorchCpuUnpickler(f).load())

    for attribute, comp in ATTRIBUTE_COMPUTERS.items():
        print("attribute comp", attribute)
        d[attribute] = comp(d)

    # prepare targets
    to_free = [k for k in d.keys() if k not in TARGETS and k != decoder_args.feature]
    for k in to_free:
        del d[k]

    labels = torch.concat([torch.from_numpy(d[t]) for t in TARGETS], -1)
    inputs = torch.from_numpy(d[decoder_args.feature])
    print(".......................................")
    print("label shape", labels.shape)
    print("inputs shape", inputs.shape)
    # train decoder
    args.targetsdim = labels.shape[-1]
    args.targets = TARGETS
    args.decoeder_args = decoder_args
    decoder = EBDecoder(args)
    train(
        decoder,
        inputs,
        labels,
        lr=0.05,
        max_epochs=1000,  # int(1e4),
        grad_norm_threshold=1e-7,
        epoch_size=100,  # int(1e2),
    )
    # save output
    out_path = decoder_args.outputname
    r2s = decoder.eval_r2(inputs, labels)
    r2_dict = {k: v for k, v in zip(TARGETS, r2s)}
    pkl_save = {"decoder": decoder, "r2": r2s, "r2_dict": r2_dict}
    with open(out_path, "wb") as f:
        pickle.dump(pkl_save, f)
    print("saved to", out_path)
