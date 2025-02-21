# This works as a helper file.
# We describe Robbins, fixed_grid_npmle, and also ERM monotone here.

import numpy as np
import scipy as sp
import torch
from tqdm import tqdm


def robbins(inputs):
    """
    Args:
        inputs: B x N x D (for now only D = 1 is supported)
    Returns:
        result: B x N x D
    """
    assert inputs.shape[2] == 1
    B, N, D = inputs.shape
    # Try to vectorize this part?
    inputs_long = inputs.long()
    XMax = torch.max(inputs_long).item() + 1
    counts_all = [torch.bincount(inp.flatten()) for inp in inputs_long]  # B x XMax
    # Pad them first.
    counts_all = torch.stack(
        [
            torch.cat((cnt, torch.zeros(XMax - cnt.shape[0]).to(inputs.device)))
            for cnt in counts_all
        ]
    )
    shifted_all = torch.roll(counts_all, shifts=-1, dims=1)
    shifted_all[:, -1] = 0
    mult = shifted_all / (counts_all + 1)  # B x XMax
    result = mult[torch.arange(B)[:, None], inputs_long[:, :, 0]] * (
        inputs[:, :, 0] + 1
    )

    return result[:, :, None]


def erm_helper(inputs):
    """
    Args:
        inputs: torch tensor, length N
    Returns:
        result: N
    """
    # Part 1: get frequency first.
    device = "cpu"
    counts = torch.bincount(inputs.long()).to(device)  # Length XMax + 1
    shifted = torch.roll(counts, shifts=-1)
    shifted[-1] = 0
    M = counts.shape[0]
    ref = torch.stack(
        [
            torch.arange(M).to(device),
            torch.arange(M).to(device),
            counts,
            shifted * torch.arange(1, M + 1).to(device).float(),
        ],
        dim=1,
    )
    ref = ref[ref[:, 2] + ref[:, 3] > 0]
    stk_alt = torch.empty((M, 4)).to(device)
    stk_ind = 0
    for r in ref:
        it_now = r.detach().clone()
        # PAVA in isotonic regression?
        while (
            stk_ind > 0
            and stk_alt[stk_ind - 1, 3] * it_now[2]
            >= stk_alt[stk_ind - 1, 2] * it_now[3]
        ):
            stk_ind -= 1
            it_pre = stk_alt[stk_ind]
            it_now[0] = it_pre[0]
            it_now[2] += it_pre[2]
            it_now[3] += it_pre[3]
        stk_alt[stk_ind] = it_now
        stk_ind += 1
    stk_alt = stk_alt[:stk_ind]  # Truncate the rest.
    stk_val = stk_alt[:, 3] / stk_alt[:, 2]
    dict_0 = torch.zeros(counts.shape).to(device)
    for it, val in zip(stk_alt, stk_val):
        dict_0[it[0].long() : (it[1] + 1).long()] = val

    answer = (dict_0.to(inputs.device))[inputs.long()]
    return answer


def erm(inputs):
    """
    Args:
        inputs: B x N x D (for now only D = 1 is supported)
    Returns:
        result: B x N x D
    """
    outputs = []
    device = inputs.device
    assert inputs.shape[-1] == 1, "only 1D operation is supported"
    for i in range(inputs.shape[0]):
        row = [int(e) for e in inputs[i].flatten().long()]
        output = erm_helper(inputs[i].flatten())
        outputs.append(output)
    outputs = torch.stack(outputs)[:, :, None]
    return outputs


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
    lambdas_plus = lambdas > 0
    lambdas_pruned = lambdas[lambdas_plus]  # length M'
    mu_pruned = mu[lambdas_plus]  # length M'
    is_torch = isinstance(lambdas, torch.Tensor)
    if is_torch:
        device = lambdas.device
    # newXs = np.append([], newXs.cpu());
    m = len(lambdas_pruned)
    # loglam = torch.zeros(m).to(device) if is_torch else np.zeros(m);
    loglam = torch.log(lambdas_pruned) if is_torch else np.log(lambdas_pruned)

    # Next, we want to encode the mixture density, which is the \int exp(-lambda) * \lambda^x / x! d\pi(\lambda)
    # In discrete setting, this is just sum of exp(-lambda) * \lambda^x * mu[\lambda] / x!
    # For x > 0 we may consider only those where lambda > 0.
    # However, for x = 0 we have exp(-lambda) * mu[lambda] as sum, so every lambda counts.
    # Also we want to keep the sum of each of these.

    max_Xs = int((2 + torch.max(newXs)).item()) if is_torch else 2 + np.max(newXs)
    # So apparently we are not allowed to just input max_Xs into torch empty (not sure why??????), let's extract the list out, lol.
    log_fdens_lst = torch.empty(max_Xs).to(device) if is_torch else np.empty(max_Xs)
    log_fdens_lst[0] = (
        torch.logsumexp(torch.log(mu) - lambdas, dim=0)
        if is_torch
        else sp.special.logsumexp(np.log(mu) - lambdas)
    )

    for x in range(1, max_Xs):
        if is_torch:
            log_fdens_lst[x] = torch.logsumexp(
                torch.log(mu_pruned)
                - lambdas_pruned
                + x * loglam
                - sp.special.gammaln(x + 1),
                dim=0,
            )
        else:
            log_fdens_lst[x] = sp.special.logsumexp(
                np.log(mu_pruned)
                - lambdas_pruned
                + x * loglam
                - sp.special.gammaln(x + 1)
            )

    ret = torch.zeros(len(newXs)).to(device) if is_torch else np.zeros(len(newXs))
    log_x = torch.log(newXs + 1).to(lambdas.device) if is_torch else np.log(newXs + 1)
    if is_torch:
        logret = log_x + log_fdens_lst[(newXs + 1).long()] - log_fdens_lst[newXs.long()]
    else:
        logret = (
            log_x
            + log_fdens_lst[(newXs + 1).astype(np.uint64)]
            - log_fdens_lst[newXs.astype(np.uint64)]
        )
    ret = torch.exp(logret) if is_torch else np.exp(logret)

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


def fixed_grid_npmle(inputs):
    # TODO: make this take a batch at a time maybe? but then it might increase memory consumption and probably the parallelism part is enough
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
        outputs[i] = poison_eb_fixed_grid_npmle(row, row).reshape(m, n)
    return outputs


def fixed_grid_poison_npmle_torch(
    sample, ngrid, iterations=10000, accuracy=3, file_path=None
):
    """
    sample: poison samples (1d)
    ngrid: numbero f paritions of the grod
    return: pi, grid
    """

    eps = 10 ** (-accuracy)

    grid = torch.linspace(torch.min(sample), torch.max(sample), ngrid).to(sample.device)
    pi = torch.ones_like(grid) / len(grid)  # prior is uniform over grid

    grid_poison = torch.distributions.poisson.Poisson(rate=grid)
    phi_mat = torch.exp(
        grid_poison.log_prob(sample.reshape(-1, 1))
    )  # Output should be an array of shape (samples x grid), with p[sample| grid atom]
    L = len(sample)
    ones = torch.ones_like(sample)

    for num_it in range(iterations):
        # distribution update logic
        marginals = phi_mat @ pi  # Probability of samples according to pi + grid
        Q = (
            phi_mat * pi / marginals[:, None]
        )  # Conditional probablity of each grid location given each sample
        new_pi = (
            Q.T @ ones
        ) / L  # Average the conditional probabilties to get new atom distribution

        # stopping mechanism, maybe use TV as comparison?
        diff = torch.sum(torch.abs(new_pi - pi))
        if diff < eps:
            break

        pi = new_pi

    print("Number of iterations: {}, Error metric: {:.3}".format(num_it, diff.item()))
    return pi, grid


def poison_eb_fixed_grid_npmle(train, test, ngrid=None):
    if ngrid is None:
        ngrid = (2000 * (torch.max(train) - torch.min(train) + 1)).int().item()

    pi, grid = fixed_grid_poison_npmle_torch(train, ngrid)
    return eval_regfunc(grid, pi, test)
