# This is to test some trained hierarchical bayesian models, including whether it's doing alpha-posterior. 

import numpy as np 
from code_submit.src.eb_transformer.eb_transformer import EBTransformer
import torch 
from gen_priors import Multinomial, NeuralPrior
from eb_transformer import EBTransformer, custom_transformer
from eb_transformer.temp_mha import TempMHA,  convert_model_mha_to_temp
import argparse
from utils import load_model_dict, model_input_bincounts, convert_tensor_to_bincount
from eb_train import get_batch
import pandas as pd
import seaborn as sns
import copy

def get_hierarchical_bayes(priors, inputs, channel, alpha = 1.0, shrink_input_bincount = False): 
    # B x N x 1? 
    # Part 1: get the Bayes estimator from each prior. 
    bayes_estimators = []
    for prior in priors:
        # Maybe generate enough thetas? But idk lol. 
        bayes_estimators.append(prior.gen_bayes_est(inputs, channel))
    # Next, need to calculate the log likelihoods of the inputs w.r.t. each prior. 
    if shrink_input_bincount:
        inputs_truncate, weights = convert_tensor_to_bincount(inputs)
    log_likelihoods = []
    for prior in priors:
        if shrink_input_bincount:
            # Warning: this assumes that inputs_truncate are [0, 1, ..., Xmax-1] across all batches. 
            lld = prior.eval_loglikelihood(inputs_truncate[0:1], channel) * weights * alpha
        else:
            lld = prior.eval_loglikelihood(inputs, channel) * alpha #keep in mind alpha-posterior
        # Might need to sum across dimensions?
        lld_sum = lld.sum(dim = 1)  # sum across data dimensions
        log_likelihoods.append(lld_sum)
        # from IPython import embed; embed()
    
    # Now we shall sum them up with the log-sum-exp trick. 
    if channel == "poisson":
        lld_stack = torch.stack(log_likelihoods, dim = 0) # num_priors x B
        lld_sum = torch.logsumexp(lld_stack, dim = 0)  # B

        # Now compute the weighted sum of the Bayes estimators. 
        bayes_stack = torch.stack(bayes_estimators, dim = 0)  # num_priors x B x N x 1
        scores = torch.log(bayes_stack) + lld_stack[:, :, None, None]
        scores_sum = torch.logsumexp(scores, dim = 0)
        log_bayesest = scores_sum - lld_sum[:, None, None]
        final_bayes_estimator = torch.exp(log_bayesest)  # B x N
        #from IPython import embed; embed()
        return final_bayes_estimator
    elif channel == "gaussian":
        # Need to normalize first. 
        lld_stack = torch.stack(log_likelihoods, dim = 0) # num_priors x B
        lld_max, _ = torch.max(lld_stack, dim = 0)
        lld_shifted = lld_stack - lld_max[None, :]  # num_priors x B
        weights = torch.exp(lld_shifted)
        weights_sum = torch.sum(weights, dim = 0)  # B
        weights_normalized = weights / weights_sum[None, :]  # num_priors x B
        # Now compute the weighted sum of the Bayes estimators.
        bayes_stack = torch.stack(bayes_estimators, dim = 0)  # num_priors x B x N x 1
        weighted_bayes_est = weights_normalized[:, :, None, None] * bayes_stack
        final_bayes_estimator = torch.sum(weighted_bayes_est, dim = 0)  # B x N x 1
        return final_bayes_estimator
    else:
        raise NotImplementedError("Currently only Poisson and Gaussian channels are supported in hierarchical_bayes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_hierarchical_bayes")
    parser.add_argument('--model', help='name of model to use')
    parser.add_argument('--test_seqlen', type=int, default=None, nargs = '*', help='sequence length')
    parser.add_argument('--random_inputs', action = 'store_true', help='whether to use random inputs')
    parser.add_argument('--batch', type=int, default=None, help='batch size')
    parser.add_argument('--plot_file', type=str, default=None, help='file to save plots to')
    parser.add_argument('--csv_file', type=str, default=None, help='file to save results to as csv')
    parser.add_argument('--temperature', type = float, default=None, help='temperature for sampling from the model')
    # For faster inference by transformers. 
    parser.add_argument('--shrink_input_bincount',  action='store_true', help='whether to shrink input via bincounts before feeding into the model')
    args = parser.parse_args()
    args.mdl_name = args.model
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    args.dtype = torch.float32

    # Now let's load the model. 
    model_all = load_model_dict(args.model, args.device)
    model = model_all['model']
    model_args = model_all['args']
    model.args.device = args.device
    model_priors = model_all['priors']
    print(len(model_priors) , "priors loaded for hierarchical Bayes model.")
    # Now get args. 
    for key in model_args.__dict__:
        if not hasattr(args, key):
            setattr(args, key, model_args.__dict__[key])
        if key == "batch":
            continue
        if key == "seqlen":
            train_seqlen = model_args.__dict__[key]
        if len(args.test_seqlen) == 0 and key == "seqlen":
            setattr(args, 'test_seqlen', [model_args.__dict__[key]])
    args.test_seqlen = sorted(args.test_seqlen)
    print(args)

    # Convert to temp mha if needed.
    if isinstance(model, EBTransformer):
        model.args.device = args.device
        model_args = model.args
        #model_args.num_params = get_n_params(model)
        if args.temperature is not None:
            model.temperature = args.temperature
            model_old = copy.deepcopy(model)
            model = convert_model_mha_to_temp(model, default_temperature = args.temperature)
            model.args.temperature = args.temperature

    alpha_list = sorted([a*(10**b) for b in range(-2, 2) for a in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.667, 7.5, 8.0]])
    alpha_smaller = sorted([a*(10**b) for b in range(-2, 2) for a in [1.0, 2.0, 5.0]])
    # However these are only candidates; we only pick those that are big/small enough. 
    max_alpha = train_seqlen / args.test_seqlen[0]
    min_alpha = train_seqlen / args.test_seqlen[-1]
    alpha_list = [a for a in alpha_list if a <= max_alpha * 3.0 and a >= min_alpha / 3.0]
    alpha_smaller = [a for a in alpha_smaller if a <= max_alpha * 3.0 and a >= min_alpha / 3.0]
    print("Using alphas:", alpha_list)

    # Now let's generate inputs from each of the priors. 
    results_df = pd.DataFrame(columns = ['prior_index', 'train_seqlen', 'alpha', 'mse_gap_hb', 'regret_model', 'regret_hb', 'test_seqlen'])
    for (i, prior) in enumerate(model_priors):
        # We need to get 50x such batches. 
        for test_seqlen in args.test_seqlen:
            prior.args.seqlen = test_seqlen
            theta_list = []
            if args.random_inputs:
                _, batch_thetas = get_batch(args)
            else:
                if args.batch is not None:
                    prior.args.batch = args.batch
                batch_thetas = prior.gen_thetas()
            theta_list.append(batch_thetas)
            thetas = torch.cat(theta_list, dim = 0)
            inputs = torch.poisson(thetas) if args.channel == "poisson" else torch.normal(thetas, 1)
            with torch.no_grad():
                if args.shrink_input_bincount:
                    outputs = model_input_bincounts(model, inputs)
                else:
                    outputs = model(inputs)
            lld = prior.eval_loglikelihood(inputs, args.channel).sum(dim = 1)
            print("log likelihood average:", lld.mean().item())
            bayes = prior.gen_bayes_est(inputs, args.channel)
            regret = torch.square(outputs - bayes).mean().item()
        
            for alpha in alpha_list:
                labels = get_hierarchical_bayes(model_priors, inputs, args.channel, alpha = alpha, shrink_input_bincount = args.shrink_input_bincount)
                hierachical_mse = torch.square(outputs - labels).mean().item()
                hierachical_regret = torch.square(labels - bayes).mean().item()
                next_index = results_df.index.max() + 1 if not results_df.empty else 0
                results_df.loc[next_index] = [i, train_seqlen, alpha, hierachical_mse, regret, hierachical_regret, test_seqlen]

        
            labels = get_hierarchical_bayes(model_priors, inputs, args.channel, alpha = 1.00)
            hierachical_mse = torch.square(outputs - labels).mean().item()
            hierachical_regret = torch.square(labels - bayes).mean().item()
            print("Test seqlen: {}, MSE (Output-HB): {:.7f}; regret (against Bayes): {:.7f}, HB's regret: {:.7f}".format(
                test_seqlen, hierachical_mse, regret, hierachical_regret))

        # Next, out of curiosity, let's see how other priors do. 
            regret_alt = []
            lld_diff = []
            for (j, prior2) in enumerate(model_priors):
                if i == j:
                    continue
                bayes2 = prior2.gen_bayes_est(inputs, args.channel)
                regret2 = torch.square(bayes - bayes2).mean().item()
                lld2 = prior2.eval_loglikelihood(inputs, args.channel).sum(dim = 1)
                lld_diff.append((lld - lld2).cpu().numpy())
                regret_alt.append(regret2)
        
            print("Alternative priors' average regrets (against true Bayes):", np.array(regret_alt).mean())
            print("Average log-likelihood differences against other priors:", np.stack(lld_diff).mean())
            del theta_list
            del thetas
            del inputs
            del labels
            del outputs
    
    results_df['test_seqlen'] = results_df['test_seqlen'].astype(int)
    results_df['prior_index'] = results_df['prior_index'].astype(int)
    results_df['train_seqlen'] = results_df['train_seqlen'].astype(int)
    print(results_df[results_df['alpha'] == 1.0][['test_seqlen', 'regret_hb']])
    if args.csv_file is not None:
        results_df.to_csv(args.csv_file, index = False)
    if args.plot_file is not None:
        sns.lineplot(data = results_df, x = 'alpha', y = 'mse_gap_hb', hue = 'test_seqlen')
        import matplotlib.pyplot as plt
        plt.title("Num priors: {}, Train seqlen: {}".format(len(model_priors), train_seqlen))
        plt.xscale('log')
        plt.yscale('log')
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.xticks(alpha_smaller, alpha_smaller)
        plt.xlabel('Alpha', fontsize=18)
        plt.ylabel('MSE(T, HB)', fontsize=18)
        plt.tight_layout()
        plt.savefig(args.plot_file)