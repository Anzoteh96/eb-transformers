import argparse
import glob
import os
import pickle
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader # Ways to split up into batches. 
from torcheval.metrics import R2Score
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from eb_arena_mapper import TorchCpuUnpickler
from eb_transformer import EBTransformer

PYTORCH_NO_CUDA_MEMORY_CACHING=1


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
        # To ensure that we only prediction nonnegative values, we add a final Gelu step.
        # Could have used ReLU, but the network will be bad if we initialized the network to produce only negative values. 
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
        r2 = 1 - loss / (var + 1e-11)
        print("loss", loss)
        return loss, r2


ATTRIBUTE_COMPUTERS = {
    "erm": lambda d: d["erm"],
    "freqn": lambda d: d["freqn"],
    "freqnplus1": lambda d: d["freqnplus1"],
    "ratio": lambda d: d["freqnplus1"] / d["freqn"],
    "robbins": lambda d: (d["x"] + 1) * d["freqnplus1"] / (d["freqn"]),
}

TARGETS = [
    "y",
    "y_hat",
    "erm",
    "freqn",
    "freqnplus1",
    "ratio",
    "robbins",
    'weight', # For multinomial ones. 
    "x",
    "npmle", 
    "posterior", 
    "posterior_next"]


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
    
    start_loss, start_r2 = model.eval_r2(inputs, outputs)
    print("Start loss: {}".format(start_loss))
    print("Start r2: {}".format(start_r2))

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
        #print(model(inputs))
        # Update weights and decay learning rate
        optimizer.step()
        #print(model(inputs))
        if (epoch + 1) % epoch_size == 0:
            scheduler.step()


        # Print progress
        if (epoch + 1) % epoch_size == 0 or epoch == max_epochs - 1:
            print("----------------------------------")
            print(
                f"Epoch {epoch}, Loss: {loss.item():.6f}, Grad Norm: {total_grad_norm:.6f}"
            )
            with torch.no_grad():
                loss, r2 = model.eval_r2(inputs, outputs)
                print("r2", r2)
                print("TARGETS", TARGETS)
            print("----------------------------------")

    print("Training completed.")

# Now we define a tool to batch / split up all the inputs and labels. 
class ProbeDecoderDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs 
        self.labels = labels
    
    def __len__(self):
        return self.inputs.shape[0] 
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


# TODO: might be better to move this later to a different file or so. 
# Recall that the mapping has the form (X^TX)^{-1}X^Ty
# But each X is too big, so we have to comput X^TX and X^Ty in batches. 
def linear_fit(model, lyr_num, inputs, labels, batch_size):
    """
        Args:
            Model: the transformer we're using; 
            lyr_num: the layer where we're getting the activation from. 
            inputs: a giant list of inputs (which we know for sure we need to split up). 
    """
    device = model.args.device
    datasets = ProbeDecoderDataset(inputs, labels)
    dataloader = DataLoader(datasets, batch_size = batch_size, shuffle=False)
    dim = model.args.dmodel + 1 # The +1 for bias. 
    XTX = torch.zeros((dim, dim)).to(device)
    XTy = torch.zeros((dim, labels.shape[-1])).to(device)
    model = model.eval()
    
    with torch.no_grad(): # Important to make sure we do not run into memory issue. 
        for inp, lab in tqdm(dataloader):
            inp = inp.to(model.args.device)
            lab = lab.to(model.args.device)
            B, N, _ = inp.shape
            #emb = model.get_activations(inp)["activations"][lyr_num]
            emb = model.get_layer_activation(raw, lyr_num)["activations"]
            D = emb.shape[-1]
            emb = emb.reshape(B * N, D)

            emb_bias = torch.cat((emb, torch.ones(B * N, 1).to(device)), axis = 1) # (BN) x (D + 1)
            xtx_sub = torch.matmul(emb_bias.T, emb_bias) # (D + 1) x (D + 1)
            xty_sub = torch.matmul(emb_bias.T, lab.reshape(B * N, labels.shape[-1])) # (D + 1) x 1
            XTX.add_(xtx_sub)
            XTy.add_(xty_sub)
    
    w_hat = np.linalg.solve(XTX.cpu().numpy(), XTy.cpu().numpy()) # Numpy seems to be more numerically stable. 
    w_hat = torch.from_numpy(w_hat).to(device)
    y_all = []
    y_pred_all = []
    # Next we will compile all the predictions. 
    with torch.no_grad():
        for inp, lab in tqdm(dataloader): 
            inp = inp.to(model.args.device)
            lab = lab.to(model.args.device)
            B, N, _ = inp.shape
            #emb = model.get_activations(inp)["activations"][lyr_num]
            emb = model.get_layer_activation(raw, lyr_num)["activations"]
            # Check dimension. 
            D = emb.shape[-1]
            emb_ = emb.reshape(B * N, D)
            emb_bias = torch.cat((emb_, torch.ones(B * N, 1).to(device)), axis = 1)
            y_pred = torch.matmul(emb_bias, w_hat)
            y_all.append(lab)
            y_pred_all.append(y_pred)

            
    y_all = torch.cat(y_all)
    B, N, D = y_all.shape 
    y_all = y_all.reshape(B * N, D)
    y_pred_all = torch.cat(y_pred_all)
    
    r2_all = []
    for i in range(D):
        metric = R2Score().to(device)
        metric.update(y_pred_all[:,i], y_all[:,i])
        r2 = metric.compute().item()
        r2_all.append(r2)
    return w_hat, r2_all
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train_layer_decoder")

    parser.add_argument("--inputactivations", type=str, required=True, help = "the file where the activation is stored")
    parser.add_argument("--outputname", type=str, required=True, help = "places we store the output")
    parser.add_argument("--attributedir", type=str, required=True, help = "attributes like x, y, freq, etc")
    parser.add_argument("--feature", type=str, default="activations")
    parser.add_argument("--lyr_num", type=int, default=-1, help="Layer number; -1=None")
    parser.add_argument("--model_path", type=str, required=True, help = "Path where the model is stored")
    parser.add_argument("--norm_label", action = "store_true", help="do we want to normalize the labels?")
    parser.add_argument("--job_name", type=str, default="")
    decoder_args = parser.parse_args()
    print("--------------------------------")
    print(decoder_args.job_name)
    print("--------------------------------")
    # Load Transformer to get decoder args args
    with open(decoder_args.model_path, "rb") as f:
        model_dict = TorchCpuUnpickler(f).load()
    args = model_dict["model"].args
    # Set Device properly
    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"
    model = model_dict["model"].to(args.device)
    del model_dict

    # Load data
    stored_inputs = None
    with open(decoder_args.inputactivations, "rb") as f:
        # keys: 'seed', 'x', 'y', 'y_hat', 'activations', 'mlp_step', 'attn_step', 'out'
        d = TorchCpuUnpickler(f).load()

    for k in d.keys():
        if "seed" in k or d[k][0] is None:
            continue
        if isinstance(d[k], list):
            seqlen = d[k][0].shape[1]
        else:
            seqlen = d[k].shape[1]
        print("standarizing", k)
        if isinstance(d[k], np.ndarray):
            d[k] = torch.from_numpy(d[k])
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
    #to_free = [k for k in d.keys() if k not in TARGETS and k != decoder_args.feature]
    #for k in to_free:
    #    del d[k]

    target_ = [t for t in set(TARGETS) and set(d.keys())]
    print(target_)
    
    labels = torch.concat([torch.from_numpy(d[t]) for t in set(d.keys())], -1).to(args.device)
    
    if decoder_args.feature in d:
        inputs = torch.from_numpy(d[decoder_args.feature])
    else:
        # Need to get from activations?
        raw_inputs = torch.from_numpy(d["x"].reshape(d["x"].shape[0] // seqlen, seqlen, 1)).to(args.device)
        raw_chunks = torch.chunk(raw_inputs, chunks = 50) # 30 is dummy here, will change that to whichever fit. 
        with torch.no_grad():
            inp_all = []
            for raw in tqdm(raw_chunks):
                inp = model.get_layer_activation(raw, decoder_args.lyr_num)["activations"]
                # inp = model.get_activations(raw)["activations"][decoder_args.lyr_num]
                inp_all.append(inp)
            inputs = torch.cat(inp_all)

        inputs = inputs.reshape(inputs.shape[0] * seqlen, inputs.shape[-1]).detach()
    
    print(".......................................")
    print("label shape", labels.shape)
    print("inputs shape", inputs.shape)
    lr = 0.0000005
    if decoder_args.norm_label:
        labels = labels / labels.mean(axis=0)
        lr = 0.00005

    from sklearn.linear_model import LinearRegression
    args.targetsdim = labels.shape[-1]
    args.targets = TARGETS
    args.decoeder_args = decoder_args
    decoder = EBDecoder(args).to(args.device)
    inp1 = decoder.norm3(inputs.to(args.device))
    inp_bias = torch.cat((inp1, torch.ones(inp1.shape[0], 1).to(inp1.device)), axis = 1)
    XTX = torch.matmul(inp_bias.T, inp_bias)
    XTy = torch.matmul(inp_bias.T, labels.float().to(inp1.device))
    lr_model = LinearRegression().fit(inp1.detach().cpu(), labels.cpu())
    w_init = torch.from_numpy(lr_model.coef_).to(args.device)
    w_interc = torch.from_numpy(lr_model.intercept_).to(args.device)

    decoder.decoder.weight.data = w_init
    decoder.decoder.bias.data = w_interc
 
    #from IPython import embed; embed()
    train(
        decoder,
        inputs,
        labels,
        lr=lr, # Note: feel free to toggle this. 
        max_epochs=1000,  # int(1e4),
        grad_norm_threshold=1e-7,
        epoch_size=100,  # int(1e2),
    )
    # save output
    out_path = decoder_args.outputname
    loss, r2s = decoder.eval_r2(inputs, labels)
    r2_dict = {k: v for k, v in zip(target_, r2s)}
    pkl_save = {"decoder": decoder, "r2": r2s, "r2_dict": r2_dict}
    with open(out_path, "wb") as f:
        pickle.dump(pkl_save, f)
    print("saved to", out_path)
