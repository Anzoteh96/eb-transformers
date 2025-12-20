# This works as a helper file. 
# We describe Robbins, fixed_grid_npmle, and also ERM monotone here. 

import torch
import numpy as np
import scipy as sp 
from tqdm import tqdm

def robbins(inputs):
    """
        Args: 
            inputs: B x N x D (for now only D = 1 is supported)
        Returns:
            result: B x N x D
    """
    device = inputs.device
    B, N, D = inputs.shape
    inputs_long = inputs.long()
    if D > 1:
        eye_matrix = torch.eye(D, device=device).long()
        result = torch.empty_like(inputs).float()
        for i in range(B):
            # Each N x D 
            inputs_row = inputs_long[i]
            max_Xs = (2 + torch.max(inputs_row, dim = 0).values).int()
            uniq, counts = torch.unique(inputs_row, dim = 0, return_counts=True)
            ranges = [torch.arange(n, device = device) for n in max_Xs.tolist()]
            X_range = torch.cartesian_prod(*ranges)
            freq_table = torch.zeros(*max_Xs).long().to(device)
            indices = tuple(uniq.T)  # gives a tuple of D index tensors, each [N]
            freq_table[indices] = counts
            
            freq_this = freq_table[tuple(inputs_row.T.tolist())]
            for d in range(D):
                mult = (inputs[i, :, d] + 1).float()
                freq_next = freq_table[tuple((inputs_row + eye_matrix[d]).T.tolist())]
                result[i, :, d] = mult * freq_next / (freq_this + 1)
                #if result[i, :, d].abs().max() > 1e6:
                #    from IPython import embed; embed()
            #from IPython import embed; embed()
            # Convert to {tuple: count}
            """
            freq = {tuple(row.tolist()): int(c) for row, c in zip(uniq, counts)}
        
            
            for j in range(N):
                row_tuple = tuple(inputs_row[j].tolist())
                count = freq[row_tuple]
                for d in range(D):
                    row_tuple_plus = tuple((inputs_row[j] + eye_matrix[d]).tolist())
                    count_plus = freq.get(row_tuple_plus, 0)
                    result[i, j, d] = (inputs[i, j, d] + 1).float() * count_plus / (count + 1)
            """
            #from IPython import embed; embed()
    else:
        XMax = torch.max(inputs_long).item() + 1
        counts_all = [torch.bincount(inp.flatten()) for inp in inputs_long] # B x XMax
        # Pad them first. 
        counts_all = torch.stack([torch.cat((cnt, torch.zeros(XMax - cnt.shape[0]).to(inputs.device))) for cnt in counts_all])
        shifted_all = torch.roll(counts_all, shifts=-1, dims = 1)
        shifted_all[:, -1] = 0
        mult = shifted_all / (counts_all + 1) # B x XMax
        result = mult[torch.arange(B)[:, None], inputs_long[:,:,0]] * (inputs[:,:,0] + 1)
        result = result[:,:,None]
    return result


def erm_helper(inputs):
    """
        Args:
            inputs: torch tensor, length N
        Returns:
            result: N
    """
    # Part 1: get frequency first. 
    device = 'cpu'
    counts = torch.bincount(inputs.long()).to(device) # Length XMax + 1
    shifted = torch.roll(counts, shifts=-1)
    shifted[-1] = 0
    M = counts.shape[0]
    ref = torch.stack([torch.arange(M).to(device), torch.arange(M).to(device), counts, shifted * torch.arange(1, M + 1).to(device).float()], dim = 1)
    ref = ref[ref[:,2] + ref[:,3] > 0]
    stk_alt = torch.empty((M, 4)).to(device)
    stk_ind = 0
    for r in ref:
        it_now = r.detach().clone()
        # PAVA in isotonic regression?
        while stk_ind > 0 and stk_alt[stk_ind - 1, 3] * it_now[2] >= stk_alt[stk_ind - 1, 2] * it_now[3]:
            stk_ind -= 1
            it_pre = stk_alt[stk_ind]
            it_now[0] = it_pre[0]
            it_now[2] += it_pre[2]
            it_now[3] += it_pre[3]
        stk_alt[stk_ind] = it_now
        stk_ind += 1
    stk_alt = stk_alt[:stk_ind] # Truncate the rest. 
    stk_val = stk_alt[:,3] / stk_alt[:,2]
    dict_0 = torch.zeros(counts.shape).to(device)
    for it, val in zip(stk_alt, stk_val):
        dict_0[it[0].long():(it[1]+1).long()] = val

    answer = (dict_0.to(inputs.device))[inputs.long()]
    return answer

def slice_along_dim(x: torch.Tensor, dim: int, fixed_indices):
    """
    x: k-dim tensor
    dim: dimension along which we slice (0-based)
    fixed_indices: iterable with indices for all other dims (in order)
                   i.e. length = x.ndim - 1
    """
    assert x.ndim >= 1
    assert 0 <= dim < x.ndim
    assert len(fixed_indices) == x.ndim - 1

    idx = []
    j_other = 0
    for d in range(x.ndim):
        if d == dim:
            idx.append(slice(None))      # keep this dimension free
        else:
            idx.append(fixed_indices[j_other])
            j_other += 1

    return x[tuple(idx)]

def erm_helper_multidim(inputs):
    # Here we consider the best monotone function that matches the Robbins. 
    """
        Args:
            inputs: torch tensor, length N x D
        Returns:
            result: N x D
    """
    # Part 1: get frequency first. 
    device = 'cpu'
    D = inputs.shape[1]
    uniq, counts = torch.unique(inputs, dim = 0, return_counts=True)
    result = torch.empty_like(inputs).float()

    max_Xs = (2 + torch.max(inputs, dim = 0).values).int()
    ranges = [torch.arange(n, device = device) for n in max_Xs.tolist()]
    X_range = torch.cartesian_prod(*ranges)
    freq_table = torch.empty(*max_Xs).long().to(device)
    indices = tuple(uniq.T)  # gives a tuple of D index tensors, each [N]
    freq_table[indices] = counts
            
    freq_this = freq_table[tuple(inputs.T.tolist())]
    for d in range(D):
        shapes = torch.delete(max_Xs, d)
        range_sub = torch.cartesian_prod(*shapes)
        for sub in range_sub:
            slice_along_dim(freq_table, d, sub)
        # Here, we group by coordinate except the i-th. 
        submap = {}
        for obs in freq_map:
            ref = tuple(np.delete(np.array(obs), coord, axis=None))
            if not(ref in submap):
                submap[ref] = {}
            submap[ref][obs[coord]] = freq_map[obs]
        # Now operate on each submap. 
        for (ind, ma) in submap.items():
            prefix = ind[:coord]
            suffix = ind[coord:]
            lst = [] # our stack. 
            # Each tuple in the stack is basically (sublist, numerator, denominator)
            # Here, our "reference function" is (x+1)p_n(x+1) / p_n(x)
            Xs = np.sort(np.array(list(ma.keys())))
            Xfs = np.sort(np.unique([Xs, np.maximum(Xs - 1, 0)]))
            # TODO: change this to include -1 too
            ma_sorted = dict(sorted(ma.items(), key=lambda x: x[0], reverse=False))
            #print(Xs)
            #print(Xfs)
            #print(ma_sorted)
            #print()
            """
            for (x, freq) in ma_sorted.items():
                # We first preprocess in case where x - 1 is not in submap
                denom_now = freq
                num_now = (x + 1) * ma[x + 1] if (x + 1 in ma) else 0.00
                intrv_now = [x]
                if x > 0 and (not x - 1 in ma):
                    num_now += x * freq
                    # intrv_now = [x - 1, x]
            """
            for x in Xfs:
                denom_now = ma_sorted[x] if x in ma_sorted else 0
                num_now = (x + 1) * ma[x + 1] if (x + 1 in ma) else 0.00
                intrv_now = [x]
            

                # Now preprocess our thing. 
                while len(lst) > 0:
                    tp = lst[-1]
                    # strict increase here. 
                    if denom_now > 0 and tp[1] / tp[2] < num_now / denom_now: 
                        break
                    lst.pop()
                    num_now += tp[1]
                    denom_now += tp[2]
                    intrv_now = tp[0] + intrv_now
                lst.append((intrv_now, num_now, denom_now))
            for tup in lst:
                intrv, num, den = tup
                for x in intrv:
                    entry = list(prefix) + [x] + list(suffix)
                    if not (tuple(entry) in pred_map):
                        pred_map[tuple(entry)] = np.empty(d)
                    pred_map[tuple(entry)][coord] = num/den
    return pred_map

def erm(inputs):
    """
        Args:
            inputs: B x N x D (for now only D = 1 is supported)
        Returns:
            result: B x N x D
    """
    outputs = []
    device = inputs.device
    assert(inputs.shape[-1] == 1), "only 1D operation is supported"
    for i in range(inputs.shape[0]):
        row = [ int(e) for e in inputs[i].flatten().long() ]
        output = erm_helper(inputs[i].flatten())
        outputs.append(output)
    outputs = torch.stack(outputs)[:, :, None]
    return outputs

def phi_standard(z):
    return torch.exp(-0.5 * z ** 2) / torch.sqrt(2 * torch.tensor([torch.pi])).to(z.device)

# Now we consider the James-Stein estimator for Gaussian mean estimation.
# Here, we assumed that we know the variance sigma^2 = 1.0, thereby giving the formula 
# \theta(x) = (1 - (n - 2) / ||x||^2)x
def james_stein(inputs, sigma_sq = 1.0):
    """
        Args:
            inputs: B x N x D
        Returns:
            result: B x N x D
    """
    B, N, D = inputs.shape
    inputs_reshaped = inputs.reshape(B, N * D)
    norm2 = torch.sum(inputs_reshaped ** 2, dim = 1) # B x N
    shrink = (N * D - 2) * sigma_sq / (norm2 + 1e-8) # B x N
    shrink = torch.clamp(shrink, max = 1.0)
    result = ((1 - shrink[:,None]) * inputs_reshaped).reshape(B, N, D)
    return result

# We might also need to implement the f-modeling equivalent of the normal means estimator. 
def f_geb_dens(inputs, h = None):
    """
        Args:
            inputs: B x N x D
        Returns:
            result: B x N x D
    """
    B, N, D = inputs.shape
    assert (D == 1), "only D = 1 is supported for now"

    if h is None:
        h = torch.sqrt(1.0/torch.log(torch.tensor(max(N, 3)).float()))
    
    diff = (inputs[:, :, None, :] - inputs[:, None, :, :]) / h # B x N x N x D
    K = phi_standard(diff)
    del diff; 

    g_hat = K.mean(axis=2) / h
    g_hatp = ((inputs[:, None, :] - inputs[:, :, None]) * K).mean(axis=2) / (h**3)
    return g_hat, g_hatp 

def f_geb_est(inputs, h = None, rho = 1e-4):
    """
        Args:
            inputs: B x N x D
        Returns:
            result: B x N x D
    """
    B, N, D = inputs.shape
    assert (D == 1), "only D = 1 is supported for now"

    g_hat, g_hatp = f_geb_dens(inputs, h = h) # B x N x D

    denom = torch.clamp(g_hat, min = rho)
    result = inputs + g_hatp / denom
    return result

# Now we consider the following function of computing the Poisson Bayes estimator. 

def eval_regfunc(lambdas, mu, newXs):
    """
        Args: 
            lambdas: numpy/torch, length M (atoms locations)
            mu: numpy/torch, length M (atoms weights)
            newXs: numpy/torch, length N
        Returns:
            ret: numpy/torch, length N
    """
    # First thing is to probably prune those lambdas that are 0.
    # Reason being that we're taking the log of this lambdas later. 
    lambdas_plus = (lambdas > 0)
    lambdas_pruned = lambdas[lambdas_plus] # length M'
    mu_pruned = mu[lambdas_plus] # length M'
    is_torch = isinstance(lambdas, torch.Tensor)
    if is_torch:
        device = lambdas.device
    #newXs = np.append([], newXs.cpu());
    m = len(lambdas_pruned);
    #loglam = torch.zeros(m).to(device) if is_torch else np.zeros(m);
    loglam = torch.log(lambdas_pruned) if is_torch else np.log(lambdas_pruned)

    # Next, we want to encode the mixture density, which is the \int exp(-lambda) * \lambda^x / x! d\pi(\lambda)
    # In discrete setting, this is just sum of exp(-lambda) * \lambda^x * mu[\lambda] / x!
    # For x > 0 we may consider only those where lambda > 0. 
    # However, for x = 0 we have exp(-lambda) * mu[lambda] as sum, so every lambda counts. 
    # Also we want to keep the sum of each of these. 

    max_Xs = int((2 + torch.max(newXs)).item()) if is_torch else 2 + np.max(newXs)
    # So apparently we are not allowed to just input max_Xs into torch empty (not sure why??????), let's extract the list out, lol. 
    log_fdens_lst = torch.empty(max_Xs).to(device) if is_torch else np.empty(max_Xs)
    log_fdens_lst[0] = torch.logsumexp(torch.log(mu) - lambdas, dim = 0) if is_torch else sp.special.logsumexp(np.log(mu) - lambdas)

    for x in range(1, max_Xs):
        if is_torch:
            log_fdens_lst[x] = torch.logsumexp(torch.log(mu_pruned) - lambdas_pruned + x * loglam - torch.lgamma(torch.tensor([x + 1]).to(device)), dim = 0)
        else:
            log_fdens_lst[x] = sp.special.logsumexp(np.log(mu_pruned) - lambdas_pruned + x * loglam - sp.special.gammaln(x + 1))
    
    ret = torch.zeros(len(newXs)).to(device) if is_torch else np.zeros(len(newXs))
    log_x = torch.log(newXs + 1).to(lambdas.device) if is_torch else np.log(newXs + 1)
    if is_torch:
        logret = log_x + log_fdens_lst[(newXs + 1).long()] - log_fdens_lst[newXs.long()]
    else:
        logret = log_x + log_fdens_lst[(newXs + 1).astype(np.uint64)] - log_fdens_lst[newXs.astype(np.uint64)]
    ret = torch.exp(logret) if is_torch else np.exp(logret) 
    return ret;

# Here we expand into multidimensional version. 
def eval_regfunc_multidim(lambdas, mu, newXs):
    """
        Args: 
            lambdas: numpy/torch, length M x d (atoms locations)
            mu: numpy/torch, length M (atoms weights)
            newXs: numpy/torch, length N x d
        Returns:
            ret: numpy/torch, length N x d
    """
    is_torch = isinstance(lambdas, torch.Tensor)
    total_dim = lambdas.shape[1] # This is the dimension. 
    if is_torch:
        device = lambdas.device
    max_Xs = (2 + torch.max(newXs, dim = 0).values).int() if is_torch else 2 + np.max(newXs, axis = 0) # length d
    log_fdens_lst = torch.full(max_Xs.tolist(), torch.nan).to(device) if is_torch else np.empty(max_Xs) #torch.empty(*max_Xs).to(device)
    # Now create a few multi-dimensional object(?)
    # Well for d small (and where the minimax regret is meaningful), it probably is okay to just do all the max_1 x max_2 x ... x max_d products. 

    eye_matrix = torch.eye(total_dim, device=device).long() if is_torch else np.eye(total_dim).astype(np.uint64)
    newX_plus = torch.cat([newXs + eye_matrix[i][None, :] for i in range(total_dim)], dim = 0) # (N * d) x d
    newX_all = torch.cat([newXs, newX_plus], dim = 0) # (N * (d + 1)) x d
    uniq = torch.unique(newX_all.long(), dim = 0)
    ranges = [torch.arange(n, device = device) for n in max_Xs.tolist()]
    X_range = torch.cartesian_prod(*ranges)
    #freq_table = torch.zeros(*max_Xs).long().to(device)
    #indices = tuple(uniq.T)
    #freq_table[indices] = counts

    #freq_this = freq_table[tuple(newXs.T.tolist())]

    # Now we can just precompute every element in this range. 
    # NOTE: this is inefficient for d >= 3. 
    for x in uniq:
        # Density of x is given as the following: 
        # 1/(x1!x2!...xd!) integral e^{-theta_1-theta_2-...-theta_d}theta_1^x_1...theta_d^x_d d\pi
        if is_torch:
            # mu is still the same
            x_loglam = torch.log(lambdas) * x[None, :] # M x d
            x_loglam[:, x == 0] = 0 # Just in case some of the lambdas is 0. 
            log_fdens_lst[tuple(x.tolist())] = torch.logsumexp(torch.log(mu) - lambdas.sum(dim = 1) + x_loglam.sum(dim = 1) - torch.special.gammaln(x + 1).sum(), dim = 0)
            
            
        else:
            x_loglam = np.log(lambdas) * x[np.newaxis, :] # M x d
            x_loglam[:, x == 0] = 0 # Just in case some of the lambdas is 0. 
            log_fdens_lst[tuple(x)] = sp.special.logsumexp(np.log(mu) - lambdas.sum(axis = 1) + x_loglam.sum(axis = 1) - sp.special.gammaln(x + 1).sum(), axis = 0)
    
    ret = torch.empty(newXs.shape).to(device) if is_torch else np.empty(newXs.shape)
    log_x = torch.log(newXs + 1).to(lambdas.device) if is_torch else np.log(newXs + 1)
    

    for d in range(total_dim):
        newXs_int = newXs.long()
        newXs_int_tuple = [newXs_int.select(-1, i) for i in range(total_dim)]
        newXs_next = newXs_int + eye_matrix[d] # One-hot at d. 
        newXs_next_tuple = [newXs_next.select(-1, i) for i in range(total_dim)]
        logret = log_x[:, d] + log_fdens_lst[tuple(newXs_next_tuple)] - log_fdens_lst[tuple(newXs_int_tuple)]
        ret[:, d] = torch.exp(logret) if is_torch else np.exp(logret)
    
    # from IPython import embed; embed()
    
    return ret;

def eval_regfunc_nonint(lambdas, mu, newXs):
    """
        Args: 
            lambdas: numpy/torch, length M (atoms locations)
            mu: numpy/torch, length M (atoms weights)
            newXs: numpy/torch, length N
        Returns:
            ret: numpy/torch, length N
    """
    # This is the same as eval_regfunc, but we do not round newXs to integers.
    # We assume that lambdas and mu are the same as before. 
    lambdas_plus = (lambdas > 0)
    lambdas_pruned = lambdas[lambdas_plus] # length M'
    mu_pruned = mu[lambdas_plus]

    is_torch = isinstance(lambdas, torch.Tensor)
    if is_torch:
        device = lambdas.device
    m = len(lambdas)
    loglam = torch.log(lambdas_pruned) if is_torch else np.log(lambdas_pruned)
    n = newXs.shape[0]

    # Unfortunately might need to do forloop. 
    log_fdens_lst = torch.empty(n).to(device) if is_torch else np.empty(n)
    log_fdens_nxt = torch.empty(n).to(device) if is_torch else np.empty(n)
    for (i, x) in enumerate(newXs):
        if is_torch:
            if x < np.finfo(np.float32).eps:
                log_fdens_lst[i] = torch.logsumexp(torch.log(mu) - lambdas, dim = 0)
            else: 
                log_fdens_lst[i] = torch.logsumexp(torch.log(mu_pruned) - lambdas_pruned + x * loglam - sp.special.gammaln(x + 1), dim = 0)
            log_fdens_nxt[i] = torch.logsumexp(torch.log(mu_pruned) - lambdas_pruned + (x + 1) * loglam - sp.special.gammaln(x + 2), dim = 0)
        else:
            if x < np.finfo(np.float32).eps:
                log_fdens_lst[i] = sp.special.logsumexp(np.log(mu) - lambdas)
            else:
                # This is the same as above, but we use numpy here. 
                log_fdens_lst[i] = sp.special.logsumexp(np.log(mu_pruned) - lambdas_pruned + x * loglam - sp.special.gammaln(x + 1))

            log_fdens_nxt[i] = sp.special.logsumexp(np.log(mu_pruned) - lambdas_pruned + (x + 1) * loglam - sp.special.gammaln(x + 2))

    # ret = torch.zeros(len(newXs)).to(device) if is_torch else np.zeros(len(newXs))
    log_x = torch.log(newXs + 1).to(lambdas.device) if is_torch else np.log(newXs + 1)
    logret = log_x + log_fdens_nxt - log_fdens_lst
    
    ret = torch.exp(logret) if is_torch else np.exp(logret)

    return ret

def eval_regfunc_gaussian(lambdas, mu, newXs):
    """
        Args: 
            lambdas: numpy/torch, length M (atoms locations)
            mu: numpy/torch, length M (atoms weights)
            newXs: numpy/torch, length N
        Returns:
            ret: numpy/torch, length N
    """
    is_torch = isinstance(lambdas, torch.Tensor)
    multidim = len(lambdas.shape) > 1
    if is_torch:
        device = lambdas.device
    m = lambdas.shape[0]
    #loglam = torch.log(lambdas) if is_torch else np.log(lambdas)
    n = newXs.shape[0]
    fdens_lst = torch.empty(n).to(device) if is_torch else np.empty(n)
    fdens_mult = torch.empty(n).to(device) if is_torch else np.empty(n)
    
    for i in tqdm(range(n)):
        x = newXs[i]
        if is_torch:
            if multidim:
                density_exp = torch.exp(-0.5 * torch.sum((x - lambdas) ** 2, dim = 1)).to(device)
                fdens_lst[i] = torch.sum(mu * density_exp)
                fdens_mult[i] = torch.sum(mu[:, None] * density_exp[:, None] * lambdas, dim = 0)
            else:
                density_exp = torch.exp(-(x - lambdas) ** 2 / 2).to(device)
                fdens_lst[i] = torch.sum(mu * density_exp)
                fdens_mult[i] = torch.sum(mu * density_exp * lambdas)
        else:
            if multidim:
                density_exp = np.exp(-0.5 * np.sum((x - lambdas) ** 2, axis = 1))
                fdens_lst[i] = np.sum(mu * density_exp)
                fdens_mult[i] = np.sum(mu[:, None] * density_exp[:, None] * lambdas, axis = 0)
            else:
                density_exp = np.exp(-(x - lambdas) ** 2 / 2)
                fdens_lst[i] = np.sum(mu * density_exp)
                fdens_mult[i] = np.sum(mu * density_exp * lambdas)
    ret = fdens_mult / fdens_lst 
    return ret

def eval_regfunc_general(lambdas, mu, newXs, loglikelihood_func):
    """
        Args: 
            lambdas: numpy/torch, length M (atoms locations)
            mu: numpy/torch, length M (atoms weights)
            newXs: numpy/torch, length N
        Returns:
            ret: numpy/torch, length N
    """
    is_torch = isinstance(lambdas, torch.Tensor)
    multidim = len(lambdas.shape) > 1
    if is_torch:
        device = lambdas.device
    m = lambdas.shape[0]
    #loglam = torch.log(lambdas) if is_torch else np.log(lambdas)
    n = newXs.shape[0]
    dim = lambdas.shape[1]
    fdens_lst = torch.empty(n).to(device) if is_torch else np.empty(n)
    fdens_mult = torch.empty(n, dim).to(device) if is_torch else np.empty(n, dim)
    
    for i in range(n):
        x = newXs[i]
        density_log = loglikelihood_func(x, lambdas).to(device)
        if is_torch:
            fdens_lst[i] = torch.sum(mu * torch.exp(density_log))
            fdens_mult[i] = torch.sum(mu[:, None] * torch.exp(density_log)[:, None] * lambdas, dim = 0)
        else:
            fdens_lst[i] = np.sum(mu * np.exp(density_log))
            fdens_mult[i] = np.sum(mu[:, None] * np.exp(density_log)[:, None] * lambdas, axis = 0)
       #from IPython import embed; embed()
    ret = fdens_mult / fdens_lst.reshape(-1, 1)
    return ret

def npmle(inputs):
    """
        Args:
            inputs: B x T x D (batch, dim_token, dim_inputs/labels)
        Returns:
            outputs: torch tensor, same dimension as inputs
    """
    # NPMLE but varying grid, TODO. 
    raise NotImplementedError

def fixed_grid_npmle(inputs, channel, shrink_inputs):
    #TODO: make this take a batch at a time maybe? but then it might increase memory consumption and probably the parallelism part is enough
    """
        Args:
            inputs: B x T x D (batch, dim_token, dim_inputs/labels)
        Returns:
            outputs: torch tensor, same dimension as inputs
    """
    outputs = torch.zeros_like(inputs)
    for i in tqdm(range(inputs.shape[0])):
        m, n = inputs[i].shape
        row = inputs[i].flatten()
        outputs[i] = eb_fixed_grid_npmle(row, row, channel, weighted = shrink_inputs).reshape(m, n)
    return outputs

def fixed_grid_npmle_torch(
    sample, ngrid, channel, sample_weights = None, iterations=10000, accuracy=3, file_path=None
):
    """
    sample: poison samples (1d)
    ngrid: numbero f paritions of the grod
    return: pi, grid
    """

    eps = 10 ** (-accuracy)
    if sample_weights is None:
        sample_weights = torch.ones_like(sample) / len(sample)

    assert(channel in ["poisson", "gaussian"]), "only Poisson and Gaussian channels are supported for now"

    grid = torch.linspace(torch.min(sample), torch.max(sample), ngrid).to(sample.device)
    pi = torch.ones_like(grid) / len(grid)  # prior is uniform over grid
    if channel == "poisson":
        grid_channel = torch.distributions.poisson.Poisson(rate = grid)
    else:
        # gaussian case here. 
        grid_channel = torch.distributions.normal.Normal(loc = grid, scale = 1.00)
    phi_mat = torch.exp(
        grid_channel.log_prob(sample.reshape(-1, 1))
    )  # Output should be an array of shape (samples x grid), with p[sample| grid atom]
    L = len(sample)
    ones = torch.ones_like(sample)

    for num_it in range(iterations):
        # distribution update logic
        marginals = phi_mat @ pi  # Probability of samples according to pi + grid
        Q = (
            sample_weights[:, None] * phi_mat * pi / marginals[:, None]
        )  # Conditional probablity of each grid location given each sample
        new_pi = (
            Q.T @ ones
        ) # Average the conditional probabilties to get new atom distribution

        # stopping mechanism, maybe use TV as comparison?
        diff = torch.sum(torch.abs(new_pi - pi))
        if diff < eps:
            break

        pi = new_pi

    print("Number of iterations: {}, Error metric: {:.3}".format(num_it, diff.item()))
    return pi, grid


def eb_fixed_grid_npmle(train, test, channel, ngrid=None, weighted = False):
    if ngrid is None:
        ngrid = (1000 * (torch.max(train) - torch.min(train) + 1)).int().item()
        if ngrid > 50000:
            ngrid = 50000
    
    if weighted:
        train_long = train.long()
        counts = torch.bincount(train_long.flatten()).to(train.device)
        ind = counts > 0 
        train_truncate = torch.arange(counts.shape[0]).to(train.device)
        #from IPython import embed; embed()
        pi, grid = fixed_grid_npmle_torch(train_truncate[ind].float(), ngrid, channel, sample_weights = counts[ind] / train.shape[0], accuracy = 3.5)
    else:
        pi, grid = fixed_grid_npmle_torch(train, ngrid, channel)
    # For Poisson, the inputs are all integers. Else we can formulate a function that takes noninteger?
    assert (channel in ["poisson", "gaussian"]), "non poisson/Gaussian channels are not supported"
    if channel == "poisson":
        return eval_regfunc(grid, pi, test)
    return eval_regfunc_gaussian(grid, pi, test)


def eb_npmle_prior(train, channel, ngrid = None, weighted = False):
    if ngrid is None:
        ngrid  = (2000 * (torch.max(train) - torch.min(train) + 1)).int().item()
    if weighted:
        train_long = train.long()
        counts = torch.bincount(train_long.flatten()).to(train.device)
        ind = counts > 0 
        train_truncate = torch.arange(counts.shape[0]).to(train.device)
        #from IPython import embed; embed()
        pi, grid = fixed_grid_npmle_torch(train_truncate[ind].float(), ngrid, channel, sample_weights = counts[ind] / train.shape[0], accuracy = 3.5)
    else:
        pi, grid = fixed_grid_npmle_torch(train, ngrid, channel)

    #pi, grid = fixed_grid_npmle_torch(train, ngrid, channel)
    return pi, grid

