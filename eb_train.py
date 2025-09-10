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
from gen_priors import DirichletProcess, GoofyPrior, Multinomial, RandMultinomial, NeuralPrior

# Helper function that calculates the number of parameters.
def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


# import ipdb
def plot_thetas(args):
    print(f"Generating a few samples of the distributions of Thetas used")
    fig, axs = plt.subplots(4, 4)
    for ax in axs.flatten():
        _, prior = get_batch(args, return_prior = True)
        thetas = prior.gen_thetas().cpu() # Typerror: NeuralPrior.gen_thetas() takes 1 positional argument but 2 were given. I don't think we need to pass args through gen.thetas...?
        ax.hist(thetas.flatten().numpy(), bins=500, density=True)
        ax.set_ylim([0, 0.2])
        ax.set_xlim([0, args.theta_max])
    # ipdb.set_trace();

    axs.flatten()[0].set_title(f"B = {args.batch}, S={args.seqlen}, dim={args.dinput}")
    # plt.show();
    fig.savefig(args.fname_prefix + "_thetas.png")
    plt.close(fig)


def get_batch(args, return_prior=False):
    # Generate batch via:
    # Step 1. Sample a random prior for thetas
    #       TODO: currently just sampling thetas from a uniform prior
    # Step 2. Sample thetas and inputs
    # Step 3. Return inputs BxT

    # We give the users to retrieve prior; this is for the use of the BayesEst
    # (not the best codestyle but what other alternatives?)

    batch, seqlen, theta_max, dinput = (
        args.batch,
        args.seqlen,
        args.theta_max,
        args.dinput,
    )
    # How do we get priors?
    if args.prior == "neural":
        prior = NeuralPrior(args)
        thetas = prior.gen_thetas()

    elif args.prior == "goofy":
        prior = GoofyPrior(args)
        thetas = prior.gen_thetas()

    elif args.prior == "dirichlet":
        assert args.alpha is not None, "alpha must not be None for dirichlet process"
        prior = DirichletProcess(args)
        thetas = prior.gen_thetas()

    elif args.prior == "multinomial":
        # Two cases: prior_file is given (in which case we just use it),
        # or not given (then we use RandMultinomial)
        if args.prior_file is not None:
            # Now read the file. 
            lst = np.load(args.prior_file)
            atoms, probs = (torch.from_numpy(lst['atoms']).to(args.device), 
                            torch.from_numpy(lst['probs']).to(args.device)
            )
            args.atoms = atoms
            args.probs = probs
            prior = Multinomial(args)
        else:
            prior = RandMultinomial(args)
        thetas = prior.gen_thetas()
    # Maybe we want to add support for our goofyprior?
    elif args.prior == "mixture":
        # Here we only support Dirichlet and neural mixture, for now.
        assert args.dirich_prob is not None, (
            "dirichlet probability cannot be None for mixture"
        )
        assert args.alpha is not None, "alpha must not be None for dirichlet mixture"
        neural_prior = NeuralPrior(args)
        theta_neural = neural_prior.gen_thetas()
        mydirich = DirichletProcess(args)
        theta_dirichlet = mydirich.gen_thetas()
        assert args.mixture_level in ["batch", "token"]
        if args.mixture_level == "batch":
            # Here, we really want things to be at batch level.
            bern_input = torch.full(size=[args.batch], fill_value=args.dirich_prob).to(
                args.device
            )
            mask = torch.bernoulli(bern_input).bool().to(args.device)
            mask = mask.reshape(
                args.batch, 1, 1
            )  # Just use B x 1 x 1 to broadcast to the rest later.
            thetas = torch.where(mask, theta_dirichlet, theta_neural)
        else:
            # Here we're looking more in distribution shift.
            bern_input = torch.full(
                size=theta_neural.shape, fill_value=args.dirich_prob
            ).to(args.device)
            mask = (
                torch.bernoulli(bern_input).bool().to(args.device)
            )  # This is gonna be B x seqlen x dinput
            thetas = torch.where(mask, theta_dirichlet, theta_neural)
        prior = [(neural_prior, 1 - args.dirich_prob), (mydirich, args.dirich_prob)]

    inputs = torch.poisson(thetas).to(args.device)
    labels = thetas
    if return_prior:
        return (inputs, labels), prior
    else:
        return (inputs, labels)


def train(args, model=None):
    if model is None:
        model = EBTransformer(args)
        model.nr_steps = 0
    
    # Add a mechanism to temporarily store outputs every 300 (or so) epochs. 
    if args.store_temp_model:
        store_filename = args.fname_prefix + "_temp_model.pkl"
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
    for step in tqdm.tqdm(range(args.train_steps), disable=args.tqdm_disable):
        # model.param_report();

        (inputs, labels) = get_batch(args)
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
        total_mle_loss += ((inputs - labels) ** 2).sum() / inputs.numel()
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
            if args.store_temp_model:
                print(f"Storing temporary model to {store_filename}")
                with open(store_filename, "wb") as f:
                    outdict.update({"model": model})
                    outdict.update({"step": step})
                    outdict.update({"loss_ratio": cur_loss/mle_loss})
                    pickle.dump(outdict, f)

    # model.param_report();

    return {"model": model}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eb_train")
    parser.add_argument(
        "--dmodel", type=int, default=32, help="dimensionality of each token"
    )
    parser.add_argument(
        "--dinput", type=int, default=1, help="dimensionality of inputs and labels"
    )
    parser.add_argument("--batch", type=int, default=192, help="number of batches")
    parser.add_argument(
        "--theta_max", type=float, default=50, help="limit on the support of the prior"
    )
    parser.add_argument(
        "--seqlen", type=int, default=512, help="maximal length of the input"
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

    # Prior-related params, some optional but required for dirichlet distribution.
    parser.add_argument(
        "--prior", type=str, default="neural", help="prior we use for training"
    )
    parser.add_argument(
        "--alpha", type=float, default=None, help="alpha param for dirichlet"
    )
    parser.add_argument(
        "--dirich_prob", type=float, default=None, help="Dirichlet mixture probability"
    )
    parser.add_argument(
        "--mixture_level",
        type=str,
        default="batch",
        help="how should the mixture be dealt with",
    )
    parser.add_argument("--m", type=int, default=20, help="m parameter in goofy prior")
    # Let's also add randomness for thetamax.
    parser.add_argument(
        "--theta_max_israndom", action="store_true", help="are thetamax random?"
    )

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
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"

    args.dtype = torch.float32
    outdict = {
        "args": args,
    }
    salt = "".join(random.choices(string.ascii_letters + string.digits, k=3))
    fname_prefix = datetime.datetime.now().strftime("eb_%Y_%m_%d-%H_%M_" + salt)
    args.fname_prefix = fname_prefix
    if args.nohist_thetas == False:
        plot_thetas(args) # Similar issue here, I don't think we need to pass args through gen.thetas...? Nvm we do!

    if not args.keep_stdout:
        print(f"Using {fname_prefix}.log for stdout")
        sys.stdout = open(fname_prefix + ".log", "wt")

    start_time = time.time()

    # torch.autograd.set_detect_anomaly(True);

    if False:
        # Testing dumping
        model = EBTransformer(args)
        outdict.update({"model": model})
        with open("test.pkl", "wb") as f:
            pickle.dump(outdict, f)

    if True:
        print("Using the following settings:\n", args)
        main_res = train(args)
        outdict.update(main_res)
        with open(fname_prefix + ".pkl", "wb") as f:
            pickle.dump(outdict, f)

        # Insert here something that generates validation plots (e.g. on hockey data, vs NPMLE, Robbins etc)
        # plot_pickle(fname_prefix + '.pkl');

        end_time = time.time()
        print(f"Total time: {(end_time - start_time) / 60:.1f} min")
