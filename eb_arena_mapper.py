import pickle
import torch
import io
import glob
import argparse
import eb_train
import os
import numpy as np
from tqdm import tqdm
import math
import copy
from eb_transformer import EBTransformer, custom_transformer
from eb_transformer.temp_mha import TempMHA,  convert_model_mha_to_temp
from eb_train import get_n_params
#from mlp import MLP
from gen_priors import NeuralPrior, DirichletProcess, WorstPrior
import random
from algo_helpers import robbins, erm_helper, erm, fixed_grid_npmle, eval_regfunc, npmle, james_stein
import sys
import time
from utils import load_model_dict, convert_tensor_to_bincount, model_input_bincounts


def set_seed(seed: int = 42) -> None:
    #https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

# Sanity check: make sure that the MLP can also overfit into one prior. 
# To do so we regurgitate the input-output pair from the same prior. 
def train_overfit(model, inputs, labels, num_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, eps=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 1.0, gamma=0.9
    )
    for step in tqdm(range(num_epochs)):
        # model.param_report();
        loss = model.eval_loss(inputs, labels)
        #if (step+1) % 100 == 0:
        #    from IPython import embed; embed()
        optimizer.zero_grad()
        # print(loss)
        loss.backward()
        norm_type = 2
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type) for p in model.parameters()]
            ),
            norm_type,
        )
        # total_grad_norm += total_norm
        optimizer.step()
    return model

def gen_batch_from_seed(args, return_prior = False):
    """
        Returns:
            inputs: integer labels of the observations. 
            labels: the ground truth thetas. 
            return_prior: prior of our generating process or NONE if mixture (or if get_prior is False). 
    """
    # The purpose of returning prior is so that we can calculate BayesEst. 
    # For now this is not supported by mixtures.
    assert args.channel in ["poisson", "gaussian"], "only Poisson and Gaussian channels are supported for now"
    set_seed(args.seed)
    prior = None # The prior we use. 
    if "worst_prior" in args and args.worst_prior:
        wp = WorstPrior(args, save_file = 'worst_priors/theta50_seed19.npz')
        labels = wp.gen_thetas()
        prior = wp
    elif ("same_prior" in args and args.same_prior):
        labels = args.prior()
        prior = args.prior
    else:
        if return_prior:
            (_, labels), prior = eb_train.get_batch(args, True)
        else:
            (_, labels) = eb_train.get_batch(args, False)
    if args.uniform_percentage > 0.0:
        assert(not return_prior), "Bayes est is not supported for uniform mixture / distribution shift yet"
        # We note here that we shouldn't mix uniform with Dirichlet prior/mixture, otherwise things will get messy. 
        assert(args.prior == 'neural'), "nonzero uniform precentage is not supported yet for other priors"
        set_seed(args.seed)
        mask = torch.rand(*labels.shape).to(labels.device) < args.uniform_percentage
        uniform = torch.rand(*labels.shape).to(labels.device) * args.theta_max
        labels[mask] = uniform[mask]

    if args.channel == "poisson":
        inputs = torch.poisson(labels)
    else:
        inputs = torch.normal(mean = labels, std = 1)

    #print(f"input share {inputs.shape}, labels shape {labels.shape}")
    if return_prior:
        return (inputs, labels), prior
    else:
        return (inputs, labels)


def get_batch_loss(model, args):
    if model == "bayes":
        (inputs, labels_raw), prior = gen_batch_from_seed(args, True)
    else:
        (inputs, labels_raw) = gen_batch_from_seed(args)
    # There are a few things we're trying to get: 
    # - The MSE of the model; 
    # - The normalized MSE of the model (normalize by the variance of the labels)
    # - The Bayes estimator of the model; 
    labels = args.func(labels_raw)
    del labels_raw
    
    # We first encode the variance. 
    labels_var = torch.var(labels)
    

    if model == npmle:
        outputs = model(inputs)
    elif model == "bayes": 
        # If f is not identity, we apply f to Bayes estimator. 
        # This is not the real Bayes estimator, but a reasonable proxy that we may compare against. 
        raw_outputs = prior.gen_bayes_est(inputs, args.channel)
        outputs = args.func(raw_outputs)
        inference_time = None
    elif model == "worst_prior":
        # Load model, for now we just have to hard code. 
        # save_file = "worst_priors/theta_{}.npz".format(int(args.theta_max))
        save_file = "worst_priors/theta50_seed19.npz" # Might be worth passing in as prior? 
        lst = np.load(save_file)
        atoms, probs = torch.from_numpy(lst['atoms']).to(args.device), torch.from_numpy(lst['probs']).to(args.device)
        outputs = eval_regfunc(atoms, probs, inputs.flatten()).reshape(inputs.shape)
        inference_time = None
    else:
        if args.prior_fit:
            import copy
            model_start = copy.deepcopy(model)
            b = args.batch
            args.batch = 10 * b
            (inputs_new, labels_new) = gen_batch_from_seed(args)
            model = train_overfit(model, inputs_new, labels_new, 200)
            args.batch = b
        with torch.inference_mode():
            # There are two different ways of measuring time: CUDA vs CPU. 
            # For CUDA, we'll make sure that CUDA synchronizes. 
            # For CPU, we can just use perf counter. 
            
            if args.device == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                if args.shrink_input_bincount:
                    outputs = args.func(model_input_bincounts(model, inputs))
                else:
                    outputs = args.func(model(inputs))
                end.record()
                torch.cuda.synchronize()
                inference_time = start.elapsed_time(end) / 1000

            else:
               start = time.perf_counter()
               outputs = model(inputs)
               end = time.perf_counter()
               inference_time = end - start
        if args.prior_fit:
            #import copy
            model = copy.deepcopy(model_start)

    if args.save_random_input:
        out_name = f"{args.dbg_file}/{args.mdl_name}{args.start}{args.end}"
        with open(out_name, "wb") as f:
            pickle.dump((inputs, labels), f)
    cur_mse = torch.mean((outputs - labels) ** 2)
    # print(cur_mse)
    normalized_mse = cur_mse / labels_var
    return cur_mse, normalized_mse, inference_time 

def get_mses(model, args, seed):
    set_seed(seed)
    data_seeds = [random.randrange(1000000000) for i in range(args.end)][args.start : args.end] 
    mses = np.zeros(args.end- args.start)
    norm_mses = np.zeros(args.end- args.start) # MSEs normalized by the variances of labels. 
    inference_times = np.zeros(args.end- args.start)
    #TODO: vectorize?
    for i, data_seed in  enumerate(data_seeds):
        args.seed = data_seed
        mses[i], norm_mses[i], inference_times[i] = get_batch_loss(model, args)
    return mses, norm_mses, inference_times#, batch_outs, batch_labels




def main():
    print(sys.argv)
    parser = argparse.ArgumentParser(description='eb_arena')
    parser.add_argument('--model', help='name of model to use')
    parser.add_argument('--temperature', type = float, default=None, help='temperature for sampling from the model')
    parser.add_argument('--seed', type = int, help='seed used to generate inputs')
    parser.add_argument('--start', type = int, help='first input this job is responsible for')
    parser.add_argument('--end', type = int, help='last input this job is responsible for')
    parser.add_argument('--llmap_out', help='output dir passed by LLMapReduce')
    parser.add_argument('--save_random_input', type=bool, default=False, help='output dir passed by LLMapReduce')
    parser.add_argument('--same_prior', type=bool, default=False, help='output dir passed by LLMapReduce')
    parser.add_argument('--prior_seed', type=int, default=10)
    parser.add_argument('--shrink_input_bincount',  action='store_true', help='whether to shrink input via bincounts before feeding into the model')
    
    parser.add_argument('--dbg_file', help='path to debug stuff')
    
    # Here, we also consider some priors we want to evaluate them on. 
    # We have uniform distribution here, but let's also consider Dirichlet prior. 
    # This is also copied from eb_train. 
    parser.add_argument('--prior', type=str, default='neural', help='prior we use for training')
    parser.add_argument('--prior_file', type=str, default=None, help='file we load prior from')
    parser.add_argument('--alpha', type=float, default=None, help='alpha param for dirichlet')
    parser.add_argument('--dirich_prob', type=float, default=None, help = 'Dirichlet mixture probability')
    parser.add_argument('--uniform_percentage', type=float, default=0.0, help='percentage that')
    parser.add_argument('--prior_fit', action='store_true', help='do we want to overfit to a prior?')
    parser.add_argument(
        "--channel",
        type=str,
        default="poisson",
        help="channel type (poisson|gaussian)",
    )
    
    parser.add_argument('--dmodel', type=int, default=32, help='dimensionality of each token')
    parser.add_argument('--dinput', type=int, default=1, help='dimensionality of inputs and labels')
    parser.add_argument('--batch', type=int, default=192, help='number of batches')
    parser.add_argument('--theta_max', type=float, default=50, help='limit on the support of the prior')
    # Let's also add randomness for thetamax. 
    # In practice this is not used, since we usually evaluate it on priors with fixed thetamax. 
    parser.add_argument('--theta_max_israndom', action="store_true", help = "are thetamax random?")
    parser.add_argument('--seqlen', type=int, default=512, help='maximal length of the input')
    parser.add_argument('--uniform_prior', action='store_true', help='Simplistic generation where thetas are sampled from a uniform prior')
    parser.add_argument('--worst_prior', action='store_true', help='Trying out worst prior')
    parser.add_argument("--func", type=str, help="function of theta (default: identity)")
    parser.add_argument('--out_file', type=str, default=None, help='file to output results to')

    args = parser.parse_args()
    args.mdl_name = args.model
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    args.dtype = torch.float32

    if args.same_prior:
        set_seed(args.prior_seed)
        args.prior = eb_train.gen_random_prior(args)

    assert(args.end >= args.start), "End seed must be greater than or equal to start seed"
    benchmarks = {}

    # Process the functions first. 
    # Now process the function of the labels. 
    if args.func == "square":
        args.func = lambda x: x ** 2
    elif args.func == "cube":
        args.func = lambda x: x ** 3
    elif args.func == "sqrt":
        args.func = lambda x: torch.sqrt(x)
    elif args.func == "log": # Get log(1+x) to avoid issues with log(0).
        args.func = lambda x: torch.log(x + 1)
    else:
        args.func = lambda x: x

    benchmarks['mle'] = args.func
    benchmarks['robbins'] = robbins
    benchmarks["erm"] = erm
    benchmarks["npmle"] = npmle
    benchmarks["fixed_grid_npmle"] = (lambda x: fixed_grid_npmle(x, args.channel))
    benchmarks["bayes"] = "bayes"
    benchmarks["worst_prior"] = "worst_prior"
    benchmarks["james_stein"] = james_stein

    model = args.model
    mdl_name = ""
    if model in benchmarks:
        mdl_name =  model
        model = benchmarks[model]
        model_args = None 
    else:
        mdl_name =  model.split("/")[-1]
        model = load_model_dict(model, args.device)['model']
        model_args = None
        if args.device == 'cuda':
            model = model.cuda()
        if isinstance(model, EBTransformer):
            model.args.device = args.device
            model_args = model.args
            model_args.num_params = get_n_params(model)
            if args.temperature is not None:
                model.temperature = args.temperature
                model_old = copy.deepcopy(model)
                model = convert_model_mha_to_temp(model, default_temperature = args.temperature)
                model.args.temperature = args.temperature
                #input_dummy = torch.randn(2, 50, args.dinput).to(args.device)
                #from IPython import embed; embed()


    mses, norm_mses, runtime = get_mses(model, args, args.seed)
    output = {"mses" : mses, "norm_mses": norm_mses, "time": runtime, "mdl_name": mdl_name, 
    "args": model_args, "start": args.start, "end" : args.end, "uniform_percentage": args.uniform_percentage, 
    "test_seqlen": args.seqlen, "test_batch": args.batch, "test_channel": args.channel}
    if args.out_file is not None:
        out_name = args.out_file
    else:
        out_name = f"{args.llmap_out}{mdl_name}_{args.start}_{args.end}__{args.uniform_percentage}"
    print("I am outputting here", out_name)
    with open(out_name, "wb") as f:
        pickle.dump(output, f)
    

if __name__ == '__main__':
    main()
