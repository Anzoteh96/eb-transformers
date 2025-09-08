import math
from numbers import Real
import numpy as np
import torch
import argparse
from algo_helpers import eval_regfunc
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.dirichlet import Dirichlet
from tqdm import tqdm
import scipy as sp

# Here, we consider both the neural priors and the dirichlet priors. 
# These are the things we are mainly working on. 
# For neural prior, we are doing random neural nets. 

def gen_random_nn(args):
    # TODO: might be better to separate this process: we want to keep our choices of activations, 
    # but at the same time still allows a dynamic amount of batches/seqlen to be passed in. 
    # Required for Bayesest. 
    batch, seqlen, theta_max, dinput = args.batch, args.seqlen, args.theta_max, args.dinput
    factory_kwargs = {'device': args.device, 'dtype': args.dtype}
    activations = [torch.nn.modules.GELU(), torch.nn.modules.ReLU(), 
            torch.nn.modules.SELU(), torch.nn.modules.CELU(),
            torch.nn.modules.SiLU(),
            torch.nn.modules.GELU(), torch.nn.modules.Tanh(),
            torch.nn.modules.Tanhshrink()];
    activation = np.random.choice(activations);
    D1 = dinput*4;
    #inputs = torch.rand(batch, seqlen, D1, requires_grad = False, **factory_kwargs);
    layer1 = torch.nn.Linear(D1,4*dinput, **factory_kwargs);
    layer2 = torch.nn.Linear(4*dinput, dinput, **factory_kwargs);
    #TODO:there's probably a better way to do this so we can save weights and inspect them later
    def nn():
        inputs = torch.rand(batch, seqlen, D1, requires_grad = False, **factory_kwargs);
        out = layer2(activation(layer1(inputs)));
        out = torch.sigmoid(10*out); # This ensures the output is [0,1]
        return out.detach();
    return nn

# Here we want to add a way to get random theta max. 
# For now, we just use N(mu, 0), Exp(mu), and Cauchy. 
def gen_random_thetamax(mu, std, size):
    # Mixture of 3
    thetamax1 = torch.rand(size) * (4.0 * mu) # I.e. normally (0, 200)
    # Now we need to sample from exponential and cauchy, but here's a catch: 
    # the heavy tail might give us unstable training results, so we need to clamp it. 
    thetamax2 = torch.clamp(torch.empty(size).exponential_(1 / mu), min = 0.00, max = 10.0 * mu)
    thetamax3 = torch.clamp(torch.empty(size).cauchy_(mu, std), min = 0.00, max = 10.0 * mu)
    inds = torch.multinomial(torch.tensor([0.75, 0.125, 0.125]), size, replacement=True)
    answer = torch.empty(size)
    answer[inds == 0] = thetamax1[inds == 0]
    answer[inds == 1] = thetamax2[inds == 1]
    answer[inds == 2] = thetamax3[inds == 2]
    return answer

# Here, we consolidate everything to make it a neural prior. 
class NeuralPrior():
    def __init__(self, args):
        self.batch, self.seqlen = args.batch, args.seqlen
        self.theta_max, self.dinput = args.theta_max, args.dinput
        if args.theta_max_israndom:
            mu, std = self.theta_max, 10
            self.theta_max = gen_random_thetamax(mu, std, args.batch).reshape((args.batch, 1, 1))
            self.theta_max = self.theta_max.to(args.device)
        self.nr_mixtures = 4
        self.nns = [gen_random_nn(args) for _ in range(self.nr_mixtures)]

    def gen_thetas(self):
        final = self.nns[0]()
        for i in range(1, self.nr_mixtures):
            # TODO: figure out if we can generate different number of batches or seqlens. 
            selector = torch.rand(self.batch, self.seqlen, self.dinput);
            mask = selector < (1/self.nr_mixtures);
            out = self.nns[i]()
            final[mask] = out[mask]
        thetas = final*self.theta_max
        #inputs = torch.poisson(thetas).to(args.device);
        labels = thetas
        return labels

    def gen_bayes_est(self, inputs):
        if self.dinput > 1:
            raise NotImplementedError
        # TODO: this is slow bc we gonna sample it repeatedly. 
        # Can we possibly pass in a new "batch" param into gen_thetas?
        labels = torch.concatenate([self.gen_thetas() for _ in range(1000)]) # Tried 20000 but got OOM. 
        mu = labels.flatten()
        lambdas = torch.ones(mu.shape[0]).to(labels.device) / mu.shape[0]
        bayes_est = eval_regfunc(mu, lambdas, inputs.flatten()).reshape(inputs.shape)
        return bayes_est

class DirichletProcess():
    def __init__(self, args):
        self.args = args
        self.alpha = args.alpha

    def gen_thetas(self, seed = None):
        """
            Args:
                theta_max: scalar
                batch: [B, N]
        """
        args = self.args
        alpha = self.alpha
        batch, seqlen, theta_max, dinput = args.batch, args.seqlen, args.theta_max, args.dinput
        # Same policy, if theta_max_israndom we want to generate random theta max too. 
        if args.theta_max_israndom:
            mu, std = theta_max, 10
            theta_max = gen_random_thetamax(mu, std, args.batch).reshape((args.batch, 1, 1))
        self.dinput = dinput
        device = args.device
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        base_unif = torch.rand(size = (batch, seqlen, dinput)) * theta_max
        dirich_bern_input = torch.arange(seqlen) / (torch.arange(seqlen) + self.alpha)
        # Now need to repeat this. 
        dirich_bern_input = dirich_bern_input.reshape(1, seqlen, 1).repeat(batch, 1, dinput)
        dirich_bern = torch.bernoulli(dirich_bern_input).bool()
        # Next, we actually do the dirichlet process. 
        samples = torch.clone(base_unif) # B x L x D
        samples_perm = torch.permute(samples, (0, 2, 1)).reshape(batch * dinput, seqlen) # (B * D) x L
        for i in range(1, seqlen):
            # Recall: with prob i/(alpha + i) we sample uniformly from previous index. 
            # Thus we need a Bernoulli variable for B. 

            # We also do random index 0, 1, ..., i - 1. 
            indices = torch.randint(i, size = (batch * dinput,)) # (B * D)
            replacement = (samples_perm[torch.arange(batch * dinput), indices]).reshape(batch, dinput) # B x D
            samples[:, i] = torch.where(dirich_bern[:, i], replacement, samples[:, i])
        samples = samples.to(args.device)
        return samples

    def gen_bayes_est(self, inputs):
        if self.dinput > 1:
            raise NotImplementedError
        labels = torch.concatenate([self.gen_thetas() for _ in range(1000)])
        mu = labels.flatten()
        lambdas = torch.ones(mu.shape[0]).to(labels.device) / mu.shape[0]
        bayes_est = eval_regfunc(mu, lambdas, inputs.flatten()).reshape(inputs.shape)
        return bayes_est

# Now we consider generating from multinomial distribution. 
# This can either be from a file, or from a probability distribution. 
class Multinomial():
    def __init__(self, args):
        self.args = args
        self.atoms = args.atoms.to(args.device)
        self.probs = args.probs.to(args.device)

    def gen_thetas(self, seed = None, with_prob = False):
        args = self.args 
        batch, seqlen, dinput = args.batch, args.seqlen, args.dinput  
        inds = torch.multinomial(self.probs, batch * seqlen * dinput, replacement = True).reshape(batch, seqlen, dinput)
        thetas = self.atoms[inds]
        if with_prob:
            probs = self.probs[inds]
            return thetas, probs
        else:
            return thetas 

    def gen_theta_with_probs(self, seed = None):
        return self.gen_thetas(self, seed, with_prob = True)

    def gen_bayes_est(self, inputs):
        # TODO: this is better approximated via direct computation instead of sampling a lot of those. 
        mu = self.atoms 
        lambdas = self.probs
        bayes_est = eval_regfunc(mu, lambdas, inputs.flatten()).reshape(inputs.shape)
        return bayes_est 
    
class GoofyPrior():
    def __init__(self, args, device = "cpu", dtype=torch.float32):
        # A should be theta_max, DIM should be seqlen
        # For our early experiments, I suggest to pass C * log DIM as a single argument m
        self.A = args.theta_max
        self.DIM = args.seqlen
        self.m = args.m
        self.device = device
        self.n = args.n
    # For consistency, use torch to generate random samples rather than numpy 
    def generate_support(self):
        return torch.rand(self.m, device = self.device, dtype = self.dtype) * self.A # torch.rand generates [0, 1) so we scale it.

    # Same for dirichlet
    def generate_weights(self):
        return torch.distributions.Dirichlet(torch.ones(self.m, device=self.device, dtype=self.dtype)).sample()

    def gen_thetas(self, seed=None): # Fixed bug here :D
        # Only need to generate supports and weights once
        support = self.generate_support()
        weights = self.generate_weights()

        # Sample n indices according to weights
        indices = torch.multinomial(weights, self.n, replacement=True) #apparently multinomial is more efficient

        # Select atoms for each theta
        thetas = support[indices]

        # Return as torch tensor, matching expected shape
        return thetas.reshape(1, self.n, 1).to(self.device)


# Below is Prior on prior for multinomials, where the atoms are fixed but the probability follows dirichlet distribution. 
class RandMultinomial(Multinomial):
    def __init__(self, args):
        args.atoms = torch.linspace(0.1, args.theta_max, steps = int(10 * args.theta_max))
        N = args.atoms.shape[0]
        args.probs = Dirichlet(torch.ones(N)).sample()
        super().__init__(args)

# For bookcorpus purpose: zipf law. 
class ZipfPrior(Multinomial):
    def __init__(self, args):
        self.args = args
        self.limit = args.limit
        self.alpha = args.alpha
        args.atoms = torch.arange(1, args.limit).to(args.device)
        args.probs = args.atoms ** (-self.alpha)
        args.probs = args.probs / torch.sum(args.probs)
        super().__init__(args)


# Here we have the generator fn for worstprior. This most likely would require loading from a file 
# as the worstprior has to be pre-computed. 
class WorstPrior():
    def __init__(self, args, save_file = None):
        if args.dinput > 1:
            raise NotImplementedError
        self.theta_max = args.theta_max
        self.args = args
        if save_file is not None:
            # Pre-load. 
            lst = np.load(save_file)
            self.support, self.probs = lst['atoms'], lst['probs']
        else:
            self.num_grids = args.num_grids
            self.__gen_prior()

    def __gen_prior(self):
        tot_iter = self.args.tot_iter
        theta_grid = torch.linspace(0, self.theta_max, self.num_grids) # start, end, steps
        model = nn.Linear(1, self.num_grids, bias = False)
        nn.init.xavier_uniform_(model.weight)

        X_max = int(max(10, self.theta_max * 10))
        D = torch.ones((1, 1))
        optimizer = Adam(model.parameters(), lr=5e-2)

        for it in tqdm(range(tot_iter)):
            model = model.train()
            optimizer.zero_grad()
            prob_dist = torch.nn.Softmax(dim =  1)(model(D))
            second_moment = torch.sum(prob_dist * (theta_grid ** 2))
            skewed_moments = torch.empty(X_max + 2)
            skewed_moments[0] = torch.logsumexp(torch.log(prob_dist) - theta_grid, axis = 1)
            for x in range(1, X_max + 2):
                skewed_moments[x] = torch.logsumexp(torch.log(prob_dist) - theta_grid + x * torch.log(theta_grid), axis = 1)
            log_minus_term = torch.logsumexp(torch.stack([2 * skewed_moments[x + 1] - skewed_moments[x] - sp.special.gammaln(x + 1) for x in range(X_max + 1)]), axis = 0)
            bayes_err = second_moment - torch.exp(log_minus_term)
            loss = -bayes_err
            loss.backward()
            optimizer.step()
        prob_dist = (torch.nn.Softmax(dim = 1)(model(D))).flatten()
        # Might be worth to prune the weights?
        valid_inds = (prob_dist > 1e-20)
        self.support = theta_grid[valid_inds].detach().cpu()
        self.probs = prob_dist[valid_inds].detach().cpu()
        
        

    def gen_thetas(self):
        args = self.args
        num_samples = args.batch * args.seqlen * args.dinput
        probs = torch.from_numpy(self.probs).to(args.device)
        indices = torch.multinomial(probs, num_samples = num_samples, replacement = True)
        labels = torch.from_numpy(self.support).float().to(args.device)[indices].reshape(args.batch, args.seqlen, args.dinput)
        return labels

    def gen_bayes_est(self, inputs):
        bayes_est = eval_regfunc(self.support, self.probs, inputs.flatten()).reshape(inputs.shape)
        return bayes_est
        

if __name__ == "__main__":
    # sample use: python3 gen_priors.py --batch 10 --theta_max 50 --seqlen 512 --alpha 1
    parser = argparse.ArgumentParser(description='gen_prior')
    parser.add_argument('--dinput', type=int, default=1, help='dimensionality of inputs and labels');
    parser.add_argument('--batch', type=int, help='number of batches');
    parser.add_argument('--theta_max', type=float, help='limit on the support of the prior');
    parser.add_argument('--seqlen', type=int, help='maximal length of the input');
    parser.add_argument('--alpha', type=float, help='alpha for dirichlet')
    parser.add_argument('--prior', type=str, help='prior we are using')
    parser.add_argument('--m', action='store_true', help='m = c * log DIM for goofy prior')
    parser.add_argument('--n', type=int, default=1000, help='number of true parameters to generate')
    args = parser.parse_args();
    
    if torch.cuda.is_available():
        args.device = 'cuda';
    else:
        args.device = 'cpu';

    args.dtype = torch.float32;

    assert args.prior in ["neural", "dirich"]
    if args.prior == "neural":
        myprior = NeuralPrior(args)
        samples = myprior.gen_thetas()
    else:
        mydirich = DirichletProcess(args)
        samples = mydirich.gen_thetas(seed = 30).cpu().numpy()
    # from IPython import embed; embed()
    # Maybe plot and see...at least the first row.  
    import matplotlib.pyplot as plt
    num_samples = args.seqlen
    alpha = args.alpha
    # Let's do the plots?
    if args.prior == "neural":
        plt.hist(samples.flatten(), bins = np.arange(0, args.theta_max))
        plt.title("Prior: neural; num samples: {}".format(num_samples))
        plt.savefig("neural_labels.png")
    else:
        plt.hist(samples[0], bins = np.arange(0, args.theta_max))
        plt.title("Prior: Dirichlet on uniform; num samples: {}; alpha = {}".format(num_samples, alpha))
        plt.savefig("dirichlet_labels.png")
    plt.clf()
