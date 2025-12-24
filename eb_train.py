## Nb. This is exclusively for training EB transformer models. 

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
from gen_priors import DirichletProcess, RandMultinomial, NeuralPrior, ExponentialPrior
from channels import slcp_channel, slcp_mle, two_moons_channel, inv_kinematrics_channel
from gen_priors import Multinomial
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


def plot_thetas(args):
    print(f"Generating a few samples of the distributions of Thetas used")
    # Idea: plot histograms of theta samples from the prior when dinput = 1, 
    # and scatterplot when dinput = 2 (idk how to do for dinput > 2 that's for future.)
    fig, axs = plt.subplots(4, 4)
    assert (args.dlabel <= 2), "dlabel > 2 is not supported for plotting yet."
    for ax in axs.flatten():
        _, prior = get_batch(args, return_prior = True)
        thetas = prior.gen_thetas().cpu()
        if args.dlabel == 2:
            ax.scatter(thetas[:, :, 0].flatten().numpy(), thetas[:, :, 1].flatten().numpy(), alpha=0.1, s=1)
            #ax.set_xlim([0, args.theta_max])
            #ax.set_ylim([0, args.theta_max])
            ax.set_aspect('equal', 'box')
            continue
        else:
            ax.hist(thetas.flatten().numpy(), bins=500, density=True)
            #ax.set_ylim([0, 0.2])
            ax.set_xlim([0, args.theta_max])

    axs.flatten()[0].set_title(f"B = {args.batch}, S={args.seqlen}, dim={args.dlabel}")
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

    batch, seqlen, theta_max, dlabel= (
        args.batch,
        args.seqlen,
        args.theta_max,
        args.dlabel,
    )
    
    args.has_negative = (args.channel == "gaussian")
   
    # How do we get priors?
    if args.prior == "neural":
        prior = NeuralPrior(args)
        
        thetas = prior.gen_thetas()

    elif args.prior == "dirichlet":
        assert args.alpha is not None, "alpha must not be None for dirichlet process"
        prior = DirichletProcess(args)
        thetas = prior.gen_thetas()

    elif args.prior == "multinomial":
        # Two cases: prior_file is given (in which case we just use it),
        # or not given (then we use RandMultinomial)
        if "prior_file" in args and args.prior_file is not None:
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
    
    elif args.prior == "exponential":
        prior = ExponentialPrior(args)
        thetas = prior.gen_thetas()

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
    
    # Think of how we want to support non-Poisson models. 
    channel = "poisson" if not ("channel" in args) else args.channel
    assert channel in ["poisson", "gaussian", "slcp", "two_moons", "inverse_kinematics"], (
        f"Channel {channel} is not supported, only poisson, gaussian, slcp, two_moons, inverse_kinematics are supported"
    )
    if channel == "gaussian":
        # For Gaussian, we sample from a normal distribution with mean = thetas and std = 1.
        inputs = torch.normal(mean=thetas, std=1.0).to(args.device)
    elif channel == "slcp":
        inputs = slcp_channel(thetas).to(args.device)
    elif channel == "two_moons":
        inputs = two_moons_channel(thetas).to(args.device)
    elif channel == "inverse_kinematics":
        inputs = inv_kinematrics_channel(thetas).to(args.device)
    else:
        # Here there will be a need to check that all thetas are nonnegative. 
        if torch.any(thetas < 0):
            raise ValueError(
                "Poisson channel requires all thetas to be nonnegative, but found negative thetas."
            )
        inputs = torch.poisson(thetas).to(args.device)
    labels = thetas

    if args.prior != "mixture":
        #cov_est, corr_est = prior.get_cov_est()
        #print(cov_est, corr_est)
        #bayes_est = prior.gen_bayes_est(inputs)
        pass

    if return_prior:
        return (inputs, labels), prior
    else:
        return (inputs, labels)

# Here, if we already have a prior, we can get it to generate arguments. 
def get_batch_from_prior(prior, channel):
    thetas = prior.gen_thetas()
    if channel == "gaussian":
        # For Gaussian, we sample from a normal distribution with mean = thetas and std = 1.
        inputs = torch.normal(mean=thetas, std=1.0).to(args.device)
    elif channel == "slcp":
        inputs = slcp_channel(thetas).to(args.device)
    elif channel == "two_moons":
        inputs = two_moons_channel(thetas).to(args.device)
    elif channel == "inverse_kinematics":
        inputs = inv_kinematrics_channel(thetas).to(args.device)
    else:
        # Here there will be a need to check that all thetas are nonnegative. 
        if torch.any(thetas < 0):
            raise ValueError(
                "Poisson channel requires all thetas to be nonnegative, but found negative thetas."
            )
        inputs = torch.poisson(thetas).to(args.device)
    return inputs, thetas

# Here, we want to train our EB transformer. 
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
    loss_list = []
    norm_loss_list = [] # divided by MLE loss 

    # In some instances, we also want to store our priors. 
    reuse_prior = (args.num_priors is not None)
    if reuse_prior:
        priors = []

    for step in tqdm.tqdm(range(args.train_steps), disable=args.tqdm_disable):
        # model.param_report();

        if reuse_prior:
            if step < args.num_priors:
                (inputs, labels), prior = get_batch(args, return_prior = True)
                priors.append(prior)
            else:
                prior = priors[step % args.num_priors]
                (inputs, labels) = get_batch_from_prior(prior, args.channel)
        else:
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
    outdict = {"model": model, "loss": np.array(loss_list), "norm_loss": np.array(norm_loss_list)}
    if reuse_prior:
        outdict.update({"priors": priors})
    return outdict

def train_getbayes(args, model=None):
    assert(args.channel == "poisson"), "Currently only support poisson channel for train_getbayes"
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
    total_loss = 0.0
    total_mle_loss = 0.0
    total_grad_norm = 0.0
    clip_grad = 0
    loss_list = []
    norm_loss_list = [] # divided by MLE loss 
    for step in tqdm.tqdm(range(args.train_steps), disable=args.tqdm_disable):
        # model.param_report();

        (_, labels), prior = get_batch(args, return_prior = True)
        max_tokens = 2 * (labels.max().int() + 1)
        inputs = torch.arange(max_tokens).float().to(args.device).reshape(1, -1, 1)
        # bayes_est = prior.gen_bayes_est(inputs.reshape(-1, 1), channel = args.channel)
        lambdas = labels.flatten()
        bayes_est = eval_regfunc_poisson(lambdas, torch.ones_like(lambdas)/lambdas.numel(), inputs.reshape(-1, 1))
        # Now we need to get the PMFs too. 
        # Now get the log PMF (scaled by constant).
        log_pmf_lst = [0]
        for i, p in zip(inputs[0, :-1], bayes_est[:-1]):
            ell = log_pmf_lst[-1]
            # p = (i + 1) * pmf[i+1] / pmf[i] 
            # => pmf[i+1] = pmf[i] * p / (i + 1)
            ell_next = ell + torch.log(p) - torch.log(i + 1)
            log_pmf_lst.append(ell_next.item())
        
        log_pmf = torch.Tensor(log_pmf_lst).reshape(1, -1, 1).to(args.device)
        # Now let's normalize the PMF. 
        log_pmf = log_pmf - torch.logsumexp(log_pmf, dim=1, keepdim=True)
        attn_mask = log_pmf[:,:,0].unsqueeze(-2).expand(-1, log_pmf.shape[1], -1)
        outputs = model(inputs, weights=torch.exp(attn_mask))
        weighted_loss = torch.sum((outputs - bayes_est) ** 2 * torch.exp(log_pmf))
        #from IPython import embed; embed()
        
        optimizer.zero_grad()
        weighted_loss.backward()
        norm_type = 2
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type) for p in model.parameters()]
            ),
            norm_type,
        )
        total_grad_norm += total_norm
        optimizer.step()

        total_loss += weighted_loss.item()
        loss_list.append(weighted_loss.item())
        mle_loss = ((inputs - bayes_est) ** 2 * torch.exp(log_pmf)).sum()
        norm_loss_list.append(weighted_loss.item() / mle_loss.item())
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
            if math.isnan(avg_grad_norm) or math.isnan(cur_loss):
                print("... ABORTING THIS RUN ...")
                return None
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

    return {"model": model, "loss": np.array(loss_list), "norm_loss": np.array(norm_loss_list)}

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
        "--theta_max", type=float, default=50, help="limit on the support of the prior"
    )
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
    # Let's also add randomness for thetamax.
    parser.add_argument(
        "--theta_max_israndom", action="store_true", help="are thetamax random?"
    )

    # Below, we will consider Gaussian channels, if applicable. 
    parser.add_argument(
        "--channel",
        type=str,
        default="poisson",
        help="channel type (poisson|gaussian)",
    )

    parser.add_argument("--train_bayes", action="store_true", help="train to get bayes estimator")

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
        "--num_priors", type=int, default=None, help="Number of priors to reuse during training"
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
    parser.add_argument(
        "--rotate",  action="store_true", help="apply random rotation to inputs and labels"
    )
    parser.add_argument(
        "--atoms_locations", type=str, default=None, help="locations of atoms for multinomial prior"
    )
    parser.add_argument(
        "--num_atoms", type=int, default=0, help="number of atoms for multinomial prior"
    )

    parser.add_argument(
        "--multin_dirich_param", type=float, default=1.0, help="Dirichlet parameter for multinomial prior"
    )

    parser.add_argument(
        "--model_file", type=str, default=None, help="pretrained model file"
    )

    parser.add_argument(
        "--save_dir", type=str, default=None, help="directory to save/load models"
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
    if args.nohist_thetas == False:
        plot_thetas(args)

    if not args.keep_stdout:
        log_filename = fname_prefix + ".log" if args.save_dir is None else os.path.join(args.save_dir, fname_prefix + ".log")
        print(f"Using {log_filename} for stdout")
        sys.stdout = open(log_filename, "wt")

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
            if args.train_bayes:
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
