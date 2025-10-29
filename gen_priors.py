import numpy as np
import torch
import argparse
from algo_helpers import eval_regfunc, eval_regfunc_gaussian, eval_regfunc_multidim
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.dirichlet import Dirichlet
from tqdm import tqdm
import scipy as sp
from typing import Sequence, Optional

# Here, we consider both the neural priors and the dirichlet priors. 
# These are the things we are mainly working on. 
# For neural prior, we are doing random neural nets. 

def gen_random_nn(args, has_negative = False):
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
        if has_negative:
            out = torch.tanh(10*out); # This ensures the output is [-1,1]
        else:
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

# Now, we want to get the Bayes estimator (somehow), but quantize to grids at certain levels. 
# Otherwise it's gonna be too slow. 
def quantize_and_count(X: torch.Tensor, eps, mode: str = "round"):
    """
    Quantize X (n x d) onto a grid with spacing eps in each dimension and
    return the quantized values and their frequencies.

    Args:
        X: float tensor of shape (n, d)
        eps: scalar float or length-d tensor of per-dim spacings
        mode: 'round' (to nearest), 'floor' (left-closed bins), or 'ceil'

    Returns:
        Q: (n, d) float tensor, quantized version of X
        unique_Q: (m, d) float tensor, unique quantized grid points
        counts: (m,) int64 tensor, counts for each unique_Q
        probs: (m,) float tensor, counts / n
        labels: (n,) int64 tensor, label of each row of Q indicating which unique_Q it belongs to
    """
    assert X.dim() == 2, "X must be (n, d)"
    n, d = X.shape
    device, dtype = X.device, X.dtype

    # Prepare spacing (broadcastable to (n, d))
    eps_t = torch.as_tensor(eps, device=device, dtype=dtype)
    if eps_t.ndim == 0:
        eps_t = eps_t.view(1, 1)
    elif eps_t.numel() == d:
        eps_t = eps_t.view(1, d)
    else:
        raise ValueError("eps must be scalar or length-d")

    if mode == "round":
        idx = torch.round(X / eps_t)
    elif mode == "floor":
        idx = torch.floor(X / eps_t)
    elif mode == "ceil":
        idx = torch.ceil(X / eps_t)
    else:
        raise ValueError("mode must be 'round', 'floor', or 'ceil'")

    # Integer grid indices (avoid float-comparison issues)
    idx = idx.to(torch.int64)  # shape (n, d)

    # Quantized values (exact multiples of eps)
    Q = idx.to(dtype) * eps_t  # (n, d)

    # Count unique quantized vectors (unique rows)
    unique_idx, labels, counts = torch.unique(idx, dim=0, return_inverse=True, return_counts=True)
    unique_Q = unique_idx.to(dtype) * eps_t  # (m, d)
    probs = counts.to(torch.float32) / n
    # unique_Q: unique quantized values (m, d)
    # probs: their probabilities 
    return unique_Q, probs

# Below, we write the base prior class. 
class BasePrior():
    def __init__(self):
        pass 

    def gen_thetas(self):
        raise NotImplementedError
    
    def gen_bayes_est(self, inputs, channel = "poisson"):
        if not (channel in ["poisson", "gaussian"]):
            raise NotImplementedError
        # TODO: this is slow bc we gonna sample it repeatedly. 
        # Can we possibly pass in a new "batch" param into gen_thetas?
        num_repeats = 1000 if channel == "poisson" else 50
        labels = torch.concatenate([self.gen_thetas() for _ in range(num_repeats)]) # Tried 20000 but got OOM. 
        mu = labels.reshape(-1, self.dinput) if self.dinput > 1 else labels.flatten()
        lambdas = torch.ones(mu.shape[0]).to(labels.device) / mu.shape[0]
        if channel == "poisson":
            if self.dinput > 1:
                # Welps maybe we just have to rely on the fact that the "quantized" values are not too many. 
                # But if it does get to 4 dimensions, oh well. 
                unique_Q, probs = quantize_and_count(mu, eps=0.01, mode="round") 
                bayes_est = eval_regfunc_multidim(unique_Q, probs, inputs.reshape(-1, self.dinput)).reshape(inputs.shape)
            else:
                bayes_est = eval_regfunc(mu, lambdas, inputs.flatten()).reshape(inputs.shape)
        else:
            unique_Q, probs = quantize_and_count(mu.reshape(-1, self.dinput), eps=0.0001, mode="round")
            if self.dinput > 1:
                bayes_est = eval_regfunc_gaussian(unique_Q, probs, inputs.reshape(-1, self.dinput)).reshape(inputs.shape)
            else:
                bayes_est = eval_regfunc_gaussian(unique_Q.flatten(), probs, inputs.flatten()).reshape(inputs.shape)
            # bayes_est = eval_regfunc_gaussian(mu, lambdas, inputs.flatten()).reshape(inputs.shape)
        return bayes_est
    
    # How one estimates the covariance matrix based on sampling many many inputs. 
    # Motivation: an estimate of how hard it is to outperform "product Bayes". 
    def get_cov_est(self):
        num_repeats = 1000
        labels = torch.concatenate([self.gen_thetas() for _ in range(num_repeats)]) # Tried 20000 but got OOM. 
        mu = labels.reshape(-1, self.dinput)
        return torch.cov(mu.T), torch.corrcoef(mu.T)


# Here, we consolidate everything to make it a neural prior. 
class NeuralPrior(BasePrior):
    def __init__(self, args):
        super(NeuralPrior, self).__init__()
        self.args = args
        self.batch, self.seqlen = args.batch, args.seqlen
        self.theta_max, self.dinput = args.theta_max, args.dinput
        if args.theta_max_israndom:
            mu, std = self.theta_max, 10
            self.theta_max = gen_random_thetamax(mu, std, args.batch).reshape((args.batch, 1, 1))
            self.theta_max = self.theta_max.to(args.device)
        self.nr_mixtures = 4
        if "has_negative" in args:
            self.has_negative = args.has_negative
        else:
            self.has_negative = (args.percent_negative > 0.2) if "percent_negative" in args else False
        self.nns = [gen_random_nn(args, self.has_negative) for _ in range(self.nr_mixtures)]
        if "rotate" in args and args.rotate:
            q, _ = torch.linalg.qr(torch.randn(self.dinput, self.dinput).to(self.args.device))
            self.rot_mat = q

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
        if "rotate" in self.args and self.args.rotate:
            # Random rotation. 
            labels = torch.einsum('bij,jk->bik', labels, self.rot_mat)
            if not self.args.has_negative:
                labels = torch.abs(labels)
        return labels

class DirichletProcess(BasePrior):
    def __init__(self, args):
        super(DirichletProcess, self).__init__()
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
        # Depending on whether we wanna have negative values or not, we might want to do a shift.
        if self.args.has_negative:
            base_unif = 2 * base_unif - theta_max
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
        if "rotate" in self.args and self.args.rotate:
            # Random rotation. 
            q, r = torch.linalg.qr(torch.randn(self.dinput, self.dinput).to(self.args.device))
            samples = torch.einsum('bij,jk->bik', samples, q)
            if not self.args.has_negative:
                samples = torch.abs(samples)
        return samples

    def gen_bayes_est(self, inputs, channel = "poisson"):
        raise NotImplementedError

# Now we consider generating from multinomial distribution. 
# This can either be from a file, or from a probability distribution. 
class Multinomial(BasePrior):
    def __init__(self, args):
        super(Multinomial, self).__init__()
        self.args = args
        self.atoms = args.atoms.to(args.device)
        self.probs = args.probs.to(args.device)

    def gen_thetas(self, seed = None, with_prob = False):
        args = self.args 
        batch, seqlen, dinput = args.batch, args.seqlen, args.dinput  
        self.dinput = args.dinput
        inds = torch.multinomial(self.probs, batch * seqlen, replacement = True).reshape(batch, seqlen)
        thetas = self.atoms[inds]
        if with_prob:
            probs = self.probs[inds]
            return thetas, probs
        else:
            return thetas 

    def gen_theta_with_probs(self, seed = None):
        return self.gen_thetas(self, seed, with_prob = True)

    def gen_bayes_est(self, inputs, channel = "poisson"):
        # TODO: this is better approximated via direct computation instead of sampling a lot of those. 
        if not (channel in ["poisson", "gaussian"]):
            raise NotImplementedError
        mu = self.atoms 
        lambdas = self.probs

        if channel == "poisson":
            if self.dinput > 1:
                bayes_est = eval_regfunc_multidim(mu, lambdas, inputs.reshape(-1, self.dinput)).reshape(inputs.shape)
            else:
                bayes_est = eval_regfunc(mu, lambdas, inputs.flatten()).reshape(inputs.shape)
        else:
            bayes_est = eval_regfunc_gaussian(mu, lambdas, inputs.flatten()).reshape(inputs.shape)
        return bayes_est 

# Before that, we need a mechanism to sample cartesian product without creating the cartesian product itself. 
# This is code written by ChatGPT. 
def sample_cartesian_without_replacement(
    my_list: Sequence[torch.Tensor],
    k: int,
    generator: Optional[torch.Generator] = None,
    return_indices: bool = False,
):
    """
    Uniformly sample k distinct tuples from the Cartesian product of 1D tensors in `my_list`
    WITHOUT materializing the full product.

    Parameters
    ----------
    my_list : sequence of 1D tensors
        Each tensor is the set for one coordinate (length n_i).
    k : int
        Number of tuples to sample (without replacement).
    generator : torch.Generator, optional
        For reproducible randomness (CPU generator).
    return_indices : bool
        If True, also return the tuple of index tensors used for each dimension.

    Returns
    -------
    samples : Tensor, shape (k, d)
        Sampled tuples; d = len(my_list).
    (optional) idx_per_dim : list[Tensor]
        The index tensors per dimension (each shape (k,)).
    """
    if not my_list:
        raise ValueError("my_list must be a non-empty sequence of 1D tensors.")

    if any(t.ndim != 1 for t in my_list):
        raise ValueError("All elements of my_list must be 1D tensors.")

    sizes = [int(t.numel()) for t in my_list]
    if any(s == 0 for s in sizes):
        raise ValueError("All tensors must be non-empty.")

    d = len(sizes)

    # Total number of tuples N = n1 * n2 * ... * nd (computed in Python int)
    N = 1
    for s in sizes:
        N *= s

    if k > N:
        raise ValueError(f"k={k} exceeds total number of tuples N={N}.")

    # ---- Step 1: pick k distinct flat indices in [0, N) without replacement (Floyd's algorithm) ----
    # Works in O(k) time/memory; no big allocations.
    if generator is None:
        generator = torch.Generator(device='cpu')
        #generator.manual_seed(torch.seed())  # non-deterministic default

    chosen = set()
    # Iterate j from N-k to N-1 inclusive
    # For each j, choose t ~ Uniform{0, ..., j}; if t already chosen, add j instead.
    # This yields exactly k unique numbers uniformly distributed over combinations.
    for j in range(N - k, N):
        # draw a single int in [0, j]
        t = int(torch.randint(0, j + 1, (1,), generator=generator, device='cpu').item())
        if t in chosen:
            chosen.add(j)
        else:
            chosen.add(t)
    
    assert(len(chosen) == k)

    # Convert to a torch tensor of flat indices (unsorted; order is random)
    flat = torch.tensor(list(chosen), dtype=torch.long, device='cpu')

    # ---- Step 2: "unrank" flat indices to per-dimension indices (mixed-radix) ----
    # For lexicographic ordering with my_list[0] as the fastest-changing *highest*-radix?
    # We use: idx_i = (flat // stride[i]) % sizes[i], where stride[i] = prod(sizes[i+1:])
    strides = [1] * d
    acc = 1
    for i in range(d - 1, -1, -1):
        strides[i] = acc
        acc *= sizes[i]

    idx_per_dim = []
    for i in range(d):
        stride_i = strides[i]
        # (flat // stride_i) % sizes[i] gives the index for dimension i
        idx_i = (flat // stride_i) % sizes[i]
        idx_per_dim.append(idx_i)

    # ---- Step 3: gather values per dimension and stack ----
    # Index on each tensor, preserving its device/dtype.
    cols = []
    for i, base in enumerate(my_list):
        # Move the index tensor to the same device as base for efficient gather
        cols.append(base.index_select(0, idx_per_dim[i].to(device=base.device)))

    samples = torch.stack(cols, dim=-1)  # shape: (k, d)

    if return_indices:
        return samples, idx_per_dim
    return samples

# Below is Prior on prior for multinomials, where the atoms are fixed but the probability follows dirichlet distribution. 
# EDIT: let's also add support for dinput > 1. 
class RandMultinomial(Multinomial):
    def __init__(self, args):
        # Now let's figure out how do we initialize the atoms. 
        if not "atoms_locations" in args:
            args.atoms_locations = "linspace"
        assert args.atoms_locations in ["linspace", "uniform"], "only linspace or uniform are supported for atoms_locations"
        if args.atoms_locations == "linspace":
            if args.has_negative:
                atom_base = torch.linspace(-args.theta_max, args.theta_max, steps = int(10 * args.theta_max))
            else:
                atom_base = torch.linspace(0.1, args.theta_max, steps = int(10 * args.theta_max))
        else:
            m = args.num_atoms if "num_atoms" in args else int(10 * args.theta_max)
            if args.has_negative:
                atom_base = torch.rand(m) * 2 * args.theta_max - args.theta_max
            else:
                atom_base = torch.rand(m) * args.theta_max
        # Now we need to consider the dinput > 1 case.
        if args.dinput > 1:
            # We could do [M] x [M] x ... x [M] with grid size 0.1, but this is too unwieldy. 
            # Instead we can sample them. 
            atom_base_lst = [atom_base for _ in range (args.dinput)]
            args.atoms = sample_cartesian_without_replacement(atom_base_lst, k = min(int(5 * len(atom_base[0])), int(1e4)), return_indices = False)
        else:
            args.atoms = atom_base.reshape(-1, 1)
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

    def gen_bayes_est(self, inputs, channel = "poisson"):
        if self.args.dinput > 1 or not (channel in ["poisson", "gaussian"]):
            raise NotImplementedError
        if channel == "poisson":
            bayes_est = eval_regfunc(self.support, self.probs, inputs.flatten()).reshape(inputs.shape)
        else:
            bayes_est = eval_regfunc_gaussian(self.support, self.probs, inputs.flatten()).reshape(inputs.shape)
        return bayes_est

# Exponential prior, where we want to randomly sample the lambda parameter. 
class ExponentialPrior(BasePrior):
    def __init__(self, args):
        self.args = args
        # Can probably sample lambda_param from some distribution.
        rand_param = torch.rand(1).item()
        mean_param = args.theta_max * (0.5 + rand_param) # i.e. between theta_max/2 and 3*theta_max/2
        self.lambda_param = 1 / mean_param

    def gen_thetas(self, seed = None):
        args = self.args
        batch, seqlen, dinput = args.batch, args.seqlen, args.dinput
        self.dinput = 1
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        samples = torch.empty((batch, seqlen, dinput)).exponential_(self.lambda_param).to(args.device)
        return samples
    
    def gen_bayes_est(self, inputs, channel = "poisson"):
        if self.args.dinput > 1 or not (channel in ["poisson", "gaussian"]):
            raise NotImplementedError
        
        if channel == "poisson" and self.dinput == 1:
            bayes_est = (inputs + 1) /(self.lambda_param + 1)
        elif channel == "poisson":
            bayes_est = eval_regfunc(self.support, self.probs, inputs.flatten()).reshape(inputs.shape)
        else:
            bayes_est = eval_regfunc_gaussian(self.support, self.probs, inputs.flatten()).reshape(inputs.shape)
        return bayes_est

# (TODO: Cauchy prior but clipped somewhere). 

# We now consider a few priors that can be used in multi-dimensional settings. 
# First, consider how 
class MultidimDirichlet(BasePrior):
    def __init__(self, args):
        super(MultidimDirichlet, self).__init__()
        self.args = args
        self.sample_params = args.sample_params # We'll likely need to modify this.
        assert(args.dinput == self.sample_params.shape[0] - 1)
        self._exp = torch.distributions.Exponential(rate=self._sample_params)

    def gen_thetas(self, seed = None):
        args = self.args
        batch, seqlen, dinput = args.batch, args.seqlen, args.dinput
        self.dinput = dinput
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        samples_raw = self._exp.sample((batch * seqlen, )).reshape(batch, seqlen, dinput).to(args.device)
        S = samples_raw.sum(dim=-1, keepdim=True)
        samples = (self.args.theta_max * samples_raw / S)[..., :dinput]
        return samples
        

if __name__ == "__main__":
    # sample use: python3 gen_priors.py --batch 10 --theta_max 50 --seqlen 512 --alpha 1
    parser = argparse.ArgumentParser(description='gen_prior')
    parser.add_argument('--dinput', type=int, default=1, help='dimensionality of inputs and labels');
    parser.add_argument('--batch', type=int, help='number of batches');
    parser.add_argument('--theta_max', type=float, help='limit on the support of the prior');
    parser.add_argument('--seqlen', type=int, help='maximal length of the input');
    parser.add_argument('--alpha', type=float, help='alpha for dirichlet')
    parser.add_argument('--prior', type=str, help='prior we are using')
    
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
