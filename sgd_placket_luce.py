from collections import defaultdict
from itertools import chain

import torch
import torch.optim.lr_scheduler as lr_scheduler


def get_log_outcome_probability(weights, argsort_out):
    exp_weights = torch.exp(weights)
    ranked_exp_weights = exp_weights[argsort_out]
    cumsum = torch.cumsum(ranked_exp_weights, dim=1)
    rev_cumsum = (
        torch.sum(ranked_exp_weights, dim=1)[:, None] - cumsum + ranked_exp_weights
    )
    log_probs = torch.log(rev_cumsum + 1e-6)
    return torch.sum(weights) - torch.sum(log_probs, dim=1)


def sgd_placket_luce(
    argsort_out, lr=1e-1, thresh=1e-6, max_iter=100, c=0.01, debug=True, debug_out=""
):
    weights = torch.randn(
        argsort_out.shape[1], requires_grad=True
    )  # might want to look into other initialization methods
    print(weights)
    optimizer = torch.optim.Adam([weights], lr=lr)
    optimizer.zero_grad()
    scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=0.5, end_factor=1e-3, total_iters=max_iter
    )
    for i in range(1, int(max_iter)):
        if weights.grad is not None:
            if torch.norm(weights.grad) < thresh:
                break
            if i % 100 == 0 or i < 10:
                print(torch.norm(weights.grad))
        else:
            if i % 100 == 0 or i < 10:
                print(i, " bad iteration")
        optimizer.zero_grad()
        loss = (
            -torch.mean(get_log_outcome_probability(weights, argsort_out))
            + c * (torch.sum(torch.exp(weights)) - 1) ** 2
        )
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            scheduler.step()
    return weights


def get_adjacency_lists(rankings):
    argsort_out = rankings.argsort(axis=1)
    adjacency_list = defaultdict(lambda: [])
    adjacency_list_rev = defaultdict(lambda: [])

    for i in range(rankings.shape[1]):
        for j in range(rankings.shape[1]):
            if torch.any(argsort_out[:, i] < argsort_out[:, j]):
                adjacency_list[i].append(j)
                adjacency_list_rev[j].append(i)
    return adjacency_list, adjacency_list_rev


def dfs(adjacency_list, l, i, visited):
    if i not in visited:
        visited.add(i)
        for j in adjacency_list[i]:
            dfs(adjacency_list, l, j, visited)
        l.append(i)


def get_strongly_connected_component(adjacency_list, adjacency_list_rev):
    # Source: https://cp-algorithms.com/graph/strongly-connected-components.html
    # STEP 1: make post order
    visited = set()
    post_order = []
    for i in range(len(adjacency_list)):
        dfs(adjacency_list, post_order, i, visited)
    # reverse post order
    post_order.reverse()
    # run dfs on reversed post order to get connected components
    components = []
    visited = set()
    for i in post_order:
        component = []
        dfs(adjacency_list_rev, component, i, visited)
        if len(component) > 0:
            components.append(component)
    return components


def convex_placketluce(
    rankings,
    lr=1e-1,
    thresh=1e-7,
    max_iter=1000000,
    c=0.01,
    gap_factor=2.5,
    debug=True,
    debug_out="",
):
    adjacency_list, adjacency_list_rev = get_adjacency_lists(rankings)

    components = get_strongly_connected_component(adjacency_list, adjacency_list_rev)
    # NOTE: In our case the condensed graph must be a line. In a general setting we might want to return the condensed graph as well to
    # get the full picture
    actual_rankings = rankings.argsort(axis=1)
    weights = []
    for i, component in enumerate(components):
        print(f"handling component {i} with {len(component)} models")
        if len(component) == 1:
            weights.append(torch.tensor([0.0]))
            continue
        relevant_actual_ranking = actual_rankings[:, torch.tensor(component)]
        relevant_ranking = relevant_actual_ranking.argsort(axis=1)
        component_weights = sgd_placket_luce(
            relevant_ranking,
            lr,
            thresh,
            max_iter,
            c,
            debug=debug,
            debug_out=debug_out + f"_{i}",
        )
        weights.append(component_weights)
    temp_concatinated_weights = torch.concat(weights)
    gap = gap_factor * (
        torch.max(temp_concatinated_weights) - torch.min(temp_concatinated_weights)
    )
    output_weights = [weights[0]]
    for comp_weight in weights[1:]:
        upper_bound = torch.min(output_weights[-1]) - gap
        difference = upper_bound - torch.max(comp_weight)
        output_weights.append(difference + comp_weight)
    output_weights = torch.concat(output_weights)
    output_weights[torch.tensor(list(chain.from_iterable(components)))] = (
        output_weights.clone()
    )
    return output_weights


if __name__ == "__main__":
    losses = torch.tensor([[1, 2, 3], [2, 1, 3], [1, 2, 3]])
    ranking = losses.argsort(axis=1)
    print(ranking)
    num_iter = 10000
    weights_convex = convex_placketluce(ranking, max_iter=num_iter)
    weights_none_convex = sgd_placket_luce(ranking, max_iter=num_iter)

    print("final weights")
    print("weights_convex", weights_convex)
    print("weights_none_convex", weights_none_convex)

    # This case deomonstraints that the gradient is much smaller when we break into connected components, which means we are more likely
    # to detect convergence issues

    # After switching to Adam, we can further decrease our threshold to 1e-9, and only the convex one will properly converge.
