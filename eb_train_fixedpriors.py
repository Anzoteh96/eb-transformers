# Train with fixed set of priors. 
import argparse
import datetime
import gc
import math
import os
import pickle
import random
import string
import sys
import time
import timeit

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import torch.nn.functional as F
from eb_transformer import EBTransformer
from gen_priors import DirichletProcess, RandMultinomial, NeuralPrior, ExponentialPrior, Multinomial
from channels import slcp_channel, slcp_mle, two_moons_channel, inv_kinematrics_channel
from gen_priors import Multinomial
from eb_train import get_batch_from_prior
from algo_helpers import eval_regfunc_poisson


# Helper function that calculates the number of parameters.
def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp



# Here, we want to train our EB transformer. 
def train(args, model=None):
    if model is None:
        model = EBTransformer(args)
        model.nr_steps = 0
    
    # Add a mechanism to temporarily store outputs every 300 (or so) epochs. 
    if args.store_temp_model:
        store_filename = args.fname_prefix + "_temp_model.pkl" if args.save_dir is None else os.path.join(args.save_dir, args.fname_prefix + "_temp_model.pkl")
        outdict = {
            "args": args,
        }
    
    num_params = get_n_params(model)
    print("Number of parameters: {}".format(num_params))

    lr = args.train_lr
    ad_eps = 0.01
    print(
        f"EB trans: Using Adam, initial rate={lr:.3g}, eps = {ad_eps:.3g}, interval={args.train_lr_epoch}, gamma={args.train_lr_gamma}"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=ad_eps)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 1.0, gamma=args.train_lr_gamma
    )
    real_start_time = time.time()
    start_time = time.time()

    model.train()  # turn on train mode
    log_interval = args.train_lr_epoch // 5
    # log_interval = 10;

    max_loss = 100 * args.dinput * args.theta_max**2
    total_loss = 0.0
    total_mle_loss = 0.0
    total_grad_norm = 0.0
    clip_grad = 0
    loss_list = []
    norm_loss_list = [] # divided by MLE loss 

    # In some instances, we also want to store our priors. 
    pickle_dict = pickle.load(open(args.prior_file, "rb"))
    atoms_all = pickle_dict["atoms"]
    probs_all = pickle_dict["probs"]
    priors = []
    for atoms, probs in zip(atoms_all, probs_all):
        args.atoms = torch.tensor(atoms).to(args.device)
        args.probs = torch.tensor(probs).to(args.device)
        prior = Multinomial(args)
        priors.append(prior)

    for step in tqdm.tqdm(range(args.train_steps), disable=args.tqdm_disable):
        # model.param_report();

        prior = priors[step % len(priors)]
        (inputs, thetas) = get_batch_from_prior(prior, args.channel)
        
        
        if args.train_on_bayes:
            labels = prior.gen_bayes_est(inputs, channel = args.channel)
        else:
            labels = thetas
        loss = model.eval_loss(inputs, labels, args.num_padding)
        optimizer.zero_grad()
        loss.backward()
        norm_type = 2
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type) for p in model.parameters()]
            ),
            norm_type,
        )
        total_grad_norm += total_norm
        optimizer.step()

        total_loss += loss.item()
        loss_list.append(loss.item())
        if args.channel in ["gaussian", "poisson"]:
            mle_ans = inputs
        elif args.channel == "slcp":
            mle_ans = slcp_mle(inputs, clamp = args.theta_max)
        else: # No-op
            mle_ans = torch.zeros_like(labels)
        mle_loss = ((mle_ans - labels) ** 2).sum() / mle_ans.numel()
        norm_loss_list.append(loss.item() / mle_loss.item())
        total_mle_loss += mle_loss
        if step % args.train_lr_epoch == 0 and step > 0:
            scheduler.step()

        if step % log_interval == 0 and step > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            mle_loss = total_mle_loss / log_interval
            tot_tim = time.time() - real_start_time
            avg_grad_norm = total_grad_norm / log_interval
            print(
                f"TRN | time {tot_tim / 60:4.1f} m | step {step:7d} | "
                f"lr {lr:.4g} | ms/batch {ms_per_batch:5.2f} | "
                f"norm grad = {avg_grad_norm:.3g} | "
                f"loss {cur_loss:5.2f} = {cur_loss / mle_loss:1.4f} MLE"
            )
            total_loss = 0.0
            total_grad_norm = 0.0
            total_mle_loss = 0.0
            start_time = time.time()
            if math.isnan(avg_grad_norm) or math.isnan(cur_loss) or cur_loss > max_loss:
                print("... ABORTING THIS RUN ...")
                return None
            if avg_grad_norm < 1e-5:
                clip_grad += 1
                if clip_grad >= 3:
                    print("... STOPPING EARLY DUE TO ZERO GRADIENTS ...")
                    break
            if args.store_temp_model:
                print(f"Storing temporary model to {store_filename}")
                with open(store_filename, "wb") as f:
                    outdict.update({"model": model})
                    outdict.update({"step": step})
                    outdict.update({"loss": np.array(loss_list)})
                    outdict.update({"norm_loss": np.array(norm_loss_list)})
                    outdict.update({"loss_ratio": cur_loss/mle_loss})
                    pickle.dump(outdict, f)

    # model.param_report();
    outdict = {"model": model, "loss": np.array(loss_list), "norm_loss": np.array(norm_loss_list), "priors": priors}
    return outdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eb_train")
    parser.add_argument(
        "--dmodel", type=int, default=32, help="dimensionality of each token"
    )
    parser.add_argument(
        "--dinput", type=int, default=1, help="dimensionality of inputs"
    )
    parser.add_argument(
        "--dlabel", type=int, default=None, help="dimensionality of labels"
    )
    parser.add_argument("--batch", type=int, default=192, help="number of batches")
    parser.add_argument(
        "--seqlen", type=int, default=512, help="maximal length of the input"
    )
    parser.add_argument(
        "--one_hot", type=int, default=None, help="One hot encoding"
    )
    # For weight sharing, there are three versions that we're thinking.
    # no weight share (one weight throughout), completely different (multiple weights), or two weights (first half, second half).
    # So we encode this information as a number): 0 for no weight share (all different), N for uniformly divided into N weights.
    parser.add_argument(
        "--weight_share",
        type=int,
        default=1,
        help="how many different weights do we use?",
    )


    # Below, we will consider Gaussian channels, if applicable. 
    parser.add_argument(
        "--channel",
        type=str,
        default="poisson",
        help="channel type (poisson|gaussian)",
    )
    parser.add_argument(
        "--theta_max", type=float, default=50, help="limit on the support of the prior"
    )

    parser.add_argument("--train_bayespmf", action="store_true", help="train via bayes and PMF")

    ### Neural Net hyperparams
    parser.add_argument("--step", type=float, default=0.5, help="Layer gain")
    parser.add_argument("--layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--heads", type=int, default=4, help="Number of layers")
    #    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout');
    parser.add_argument(
        "--activation", type=str, default="gelu", help="Activation (gelu|relu)"
    )
    # We'll also need to add whether we need norm_share and decoding_layer_norm,
    # for purpose of backwards compability (hopefully we can soon remove this).
    parser.add_argument(
        "--norm_share",
        action="store_true",
        help="are we doing layer sharing for layer norm?",
    )
    parser.add_argument(
        "--decoding_layer_norm",
        action="store_true",
        help="are we adding layer norm before decoding?",
    )
    # Next, we customize and see if we want our attention's activation to be softmax or something else. 
    parser.add_argument(
        "--att_activ", type=str, default="softmax", help="custom activation for attention"
    )
    parser.add_argument(
        "--attn_only", action="store_true", help = "Do we want MLP in between"
    )

    parser.add_argument(
        "--no_prenorm", action="store_true", help = "Do we want pre-norm or not"
    )

    parser.add_argument(
        "--no_postnorm", action="store_true", help = "Do we want post-norm or not"
    )

    ### Training hyper params
    parser.add_argument(
        "--train_steps", type=int, default=100_000, help="Number of training steps"
    )


    parser.add_argument(
        "--train_lr", type=float, default=0.007, help="Initial learning rate"
    )
    parser.add_argument(
        "--train_lr_epoch", type=int, default=400, help="Step down rate period"
    )
    parser.add_argument(
        "--train_lr_gamma", type=float, default=0.95, help="Step down multiplier"
    )
    parser.add_argument(
        "--uniform_prior",
        action="store_true",
        help="Simplistic training where thetas are sampled from a uniform prior",
    )
    parser.add_argument(
        "--nohist_thetas",
        action="store_true",
        help="Do not generate a sample histogram of thetas",
    )
    parser.add_argument(
        "--num_padding", type=int, default=0, help="number of padding dimension"
    )
    parser.add_argument(
        "--keep_stdout", action="store_true", help="Do not redirect to log file"
    )
    parser.add_argument("--tqdm_disable", action="store_true", help="disable tqdm")
    parser.add_argument(
        "--store_temp_model",
        action="store_true",
        help="Store temporary model every N epochs",
    )


    # When training, we can also determine if we want to train the via (input, theta) pairs, or (input, bayes_est) pairs.
    parser.add_argument(
        '--train_on_bayes', action="store_true", help="train on (input, bayes_est) pairs instead of (input, theta) pairs"
    )

    parser.add_argument(
        "--model_file", type=str, default=None, help="pretrained model file"
    )

    parser.add_argument(
        "--save_dir", type=str, default=None, help="directory to save/load models"
    )

    parser.add_argument(
        "--prior_file", type=str, default=None, help="where you get the file for priors"
    )

    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"

    args.dtype = torch.float32

    if args.save_dir is not None: 
        os.makedirs(args.save_dir, exist_ok=True)
    if args.dlabel is None:
        args.dlabel = args.dinput
    outdict = {
        "args": args,
    }
    salt = "".join(random.choices(string.ascii_letters + string.digits, k=3))
    fname_prefix = datetime.datetime.now().strftime("eb_%Y_%m_%d-%H_%M_" + salt)
    args.fname_prefix = fname_prefix

    if not args.keep_stdout:
        log_filename = fname_prefix + ".log" if args.save_dir is None else os.path.join(args.save_dir, fname_prefix + ".log")
        print(f"Using {log_filename} for stdout")
        sys.stdout = open(log_filename, "wt")

    start_time = time.time()


    if True:
        print("Using the following settings:\n", args)
        if args.model_file is not None:
            print(f"Loading model from {args.model_file}")
            with open(args.model_file, "rb") as f:
                tmpdict = pickle.load(f)
                model = tmpdict["model"]
            if args.train_bayes:
                main_res = train_getbayes(args, model=model)
            else:
                main_res = train(args, model=model)
        else:
            if args.train_bayespmf:
                main_res = train_getbayes(args)
            else:
                main_res = train(args)
        outdict.update(main_res)
        save_file = fname_prefix + ".pkl" if args.save_dir is None else os.path.join(args.save_dir, fname_prefix + ".pkl")
        print(f"Storing final model to {save_file}")
        with open(save_file, "wb") as f:
            pickle.dump(outdict, f)

        # Insert here something that generates validation plots (e.g. on hockey data, vs NPMLE, Robbins etc)
        # plot_pickle(fname_prefix + '.pkl');

        end_time = time.time()
        print(f"Total time: {(end_time - start_time) / 60:.1f} min")