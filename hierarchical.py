# This is to test some trained hierarchical bayesian models, including whether it's doing alpha-posterior. 

import numpy as np 
import torch 
from gen_priors import Multinomial, NeuralPrior
import argparse
from utils import load_model_dict 
from eb_train import get_batch

def get_hierarchical_bayes(priors, inputs, channel, alpha = 1.0): 
    # B x N x 1? 
    # Part 1: get the Bayes estimator from each prior. 
    bayes_estimators = []
    for prior in priors:
        # Maybe generate enough thetas? But idk lol. 
        bayes_estimators.append(prior.gen_bayes_est(inputs, channel))
    # Next, need to calculate the log likelihoods of the inputs w.r.t. each prior. 
    log_likelihoods = []
    for prior in priors:
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
    parser.add_argument('--seqlen', type=int, default=None, help='sequence length')
    parser.add_argument('--random_inputs', action = 'store_true', help='whether to use random inputs')
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
        if args.seqlen is None and key == "seqlen":
            setattr(args, key, model_args.__dict__[key])
    print(args)
    alpha_list = [0.05, 0.1, 0.2, 0.25, 0.5, 0.667, 1.0, 1.50, 2.0,4.0, 5.0, 10.0, 20.0]
    # Now let's generate inputs from each of the priors. 
    for (i, prior) in enumerate(model_priors):
        # We need to get 50x such batches. 
        prior.args.seqlen = args.seqlen
        theta_list = []
        for _ in range(10):
            if args.random_inputs:
                _, batch_thetas = get_batch(args)
            else:
                batch_thetas = prior.gen_thetas()
            theta_list.append(batch_thetas)
        thetas = torch.cat(theta_list, dim = 0)
        inputs = torch.poisson(thetas) if args.channel == "poisson" else torch.normal(thetas, 1)
        with torch.no_grad():
            outputs = model(inputs)
        lld = prior.eval_loglikelihood(inputs, args.channel).sum(dim = 1)
        print("log likelihood average:", lld.mean().item())
        bayes = prior.gen_bayes_est(inputs, args.channel)
        regret = torch.square(outputs - bayes).mean().item()
        for alpha in alpha_list:
            labels = get_hierarchical_bayes(model_priors, inputs, args.channel, alpha = alpha)
            hierachical_mse = torch.square(outputs - labels).mean().item()
            hierachical_regret = torch.square(labels - bayes).mean().item()
            print("Alpha: {:.2f}; Output's MSE against hierarchical Bayes (HB): {:.7f}; regret (against Bayes): {:.7f}, HB's regret: {:.7f}".format(
                alpha, hierachical_mse, regret, hierachical_regret))
        
        #labels = get_hierarchical_bayes(model_priors, inputs, args.channel)
        
        
        # from IPython import embed; embed()
        hierachical_mse = torch.square(outputs - labels).mean().item()
        
        hierachical_regret = torch.square(labels - bayes).mean().item()
        print("Output's MSE against hierarchical Bayes (HB): {:.7f}; regret (against Bayes): {:.7f}, HB's regret: {:.7f}".format(
            hierachical_mse, regret, hierachical_regret))

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