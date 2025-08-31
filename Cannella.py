# What do I need to get done in total for this project? Let's break it into parts
# 1. Implement Han's Algorithm from notes pdf REPEAT ? TIMES
# 1a. Generate support: lambda_i (i \in [1, ... ,m]) from Unif[0,O(1) = A] with m = Clogn, C>0 a large constant
# 1b. Generate weights: w_i (i \in [1, ..., m]) from Dir(1, ... , 1)
# 1c. Generate true parameters: theta_i (i \in [1, ..., n]) ~ iid from \Sum_j w_j * lambda_j
# and observations X_i from Poisson(theta_i), 1 <= i <= n
# 1d. Train transformer to find: argmin_{f: R^n \to R^n} 1/m \sum_{j=1}^m ||f(x^(j) - \theta^{(j)})||^2

import math
import numpy as np
import torch
from eb_transformer import EBTransformer
from eb_train import train
import argparse

A = 10
C = 10
DIM = 20

def generate_support(m, DIM):
    m = int(C * math.log(DIM))
    return np.random.uniform(0, A, m)

def generate_weights(m):
    return np.random.dirichlet(np.ones(m))

def generate_true_parameters(DIM, weights, support):
    true_parameters = np.zeros(DIM)
    for i in range(DIM):
        true_parameters[i] = np.sum(weights * support)
    return true_parameters

def generate_observations(DIM, true_parameters):
    observations = np.zeros(DIM)
    for i in range(DIM):
        observations[i] = np.random.poisson(true_parameters[i])
    return observations

# Generate synthetic data
m = int(C * math.log(DIM))
support = generate_support(m, DIM)
weights = generate_weights(m)
true_parameters = generate_true_parameters(DIM, weights, support)
observations = generate_observations(DIM, true_parameters)

inputs = torch.tensor(observations, dtype=torch.float32).reshape(1, DIM, 1)
labels = torch.tensor(true_parameters, dtype=torch.float32).reshape(1, DIM, 1)

args = argparse.Namespace(
    dmodel=32,
    dinput=1,
    batch=1,
    theta_max=50,
    seqlen=20,  # Should match DIM
    weight_share=1,
    prior="neural",
    alpha=None,
    dirich_prob=None,
    mixture_level="batch",
    theta_max_israndom=True,
    step=0.5,
    layers=12,
    heads=4,
    activation="gelu",
    norm_share=True,
    decoding_layer_norm=True,
    att_activ="softmax",
    attn_only=True,
    no_prenorm=True,
    no_postnorm=True,
    train_steps=100_000,
    train_lr=0.007,
    train_lr_epoch=400,
    train_lr_gamma=0.95,
    uniform_prior=True,
    nohist_thetas=True,
    num_padding=0,
    keep_stdout=True,
    tqdm_disable=True,
    store_temp_model=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.float32,
    fname_prefix="test_run"
)

# Initialize model
model = EBTransformer(args).to(args.device)

print("inputs:", inputs)
print("labels:", labels)

# Use the workspace's train function
train(args, model)

# Evaluation using model's eval_loss method
model.eval()
with torch.no_grad():
    eval_loss = model.eval_loss(inputs.to(args.device), labels.to(args.device))
    print(f"Evaluation MSE: {eval_loss.item():.4f}")

    print("Predicted vs. True values (first 5):")
    preds = model(inputs.to(args.device))
    for i in range(min(5, DIM)):
        pred_val = preds[0, i, 0].item()
        true_val = labels[0, i, 0].item()
        print(f"  {i}: Predicted={pred_val:.3f}, True={true_val:.3f}")