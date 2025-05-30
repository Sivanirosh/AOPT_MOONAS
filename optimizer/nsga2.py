# optimizer/nsga2.py

import numpy as np
import torch
import random
from typing import List, Tuple, Any

def non_dominated_sort(losses: np.ndarray) -> np.ndarray:
    """
    Given an (N,2) array of losses, return a boolean mask of the Pareto‐front.
    """
    N = losses.shape[0]
    mask = np.ones(N, dtype=bool)
    for i in range(N):
        # any point strictly dominates losses[i]?
        dominates = np.all(losses < losses[i], axis=1)
        dominates[i] = False
        if np.any(dominates):
            mask[i] = False
    return mask

def crowding_distance(losses: np.ndarray, indices: List[int]) -> np.ndarray:
    """
    Compute crowding distance for the selected front.
    Returns an array `dist` of length len(indices), aligned to `indices`.
    """
    front = losses[indices]
    n = len(indices)
    dist = np.zeros(n)
    for m in range(front.shape[1]):
        order = np.argsort(front[:, m])
        dist[order[0]] = dist[order[-1]] = np.inf
        lo, hi = front[order[0], m], front[order[-1], m]
        denom = hi - lo + 1e-12
        for j in range(1, n-1):
            prev_val = front[order[j-1], m]
            next_val = front[order[j+1], m]
            dist[order[j]] += (next_val - prev_val) / denom
    return dist


def select_by_nsga2(
    losses: List[Tuple[float, float]],
    grads:    List[torch.Tensor],
    K:        int
) -> torch.Tensor:
    """
    Given a list of (loss_cls, loss_size) and their corresponding gradient vectors,
    pick the Pareto‐front, compute crowding distances, select top‐K, and return
    the average gradient over those K individuals.
    """
    costs = np.array(losses)        # shape (N, 2)
    N = costs.shape[0]

    # 1) Pareto‐front mask
    mask = np.ones(N, dtype=bool)
    for i, c in enumerate(costs):
        mask[i] = not np.any((costs[mask] < c).all(axis=1))
    front_idx = np.where(mask)[0]
    front_costs = costs[front_idx]
    M = front_costs.shape[0]

    # 2) Crowding distances
    dist = np.zeros(M)
    for m in range(2):
        order = np.argsort(front_costs[:, m])
        dist[order[0]] = dist[order[-1]] = np.inf
        lo, hi = front_costs[order[0], m], front_costs[order[-1], m]
        denom = hi - lo + 1e-12
        for j in range(1, M - 1):
            prev = front_costs[order[j - 1], m]
            nxt  = front_costs[order[j + 1], m]
            dist[order[j]] += (nxt - prev) / denom

    # 3) Select top‐K by distance (or all if fewer)
    if M > K:
        sel = np.argsort(-dist)[:K]
        chosen = front_idx[sel]
    else:
        chosen = front_idx

    # 4) Average their gradients
    chosen_grads = torch.stack([grads[i] for i in chosen], dim=0)
    return chosen_grads.mean(dim=0)

def evolve_nsga2(
    population: List[torch.Tensor],
    losses: List[Tuple[float, float]],
    pop_size: int,
    crossover_rate: float,
    mutation_rate: float,
    eta: float = 2.0
) -> List[torch.Tensor]:
    """
    1) Non‐dominated sort & crowding select (the 'parents')
    2) SBX crossover + Gaussian mutation to refill to pop_size
    """
    N = len(population)
    loss_arr = np.array(losses)  # shape (N, 2)

    # --- 1) select survivors by Pareto + crowding ---
    front_mask = non_dominated_sort(loss_arr)
    front_idx = np.where(front_mask)[0].tolist()

    # if too many, pick top by crowding distance
    if len(front_idx) > pop_size:
        dist = crowding_distance(loss_arr, front_idx)
        # sort descending and keep top pop_size
        best = np.argsort(-dist)[:pop_size]
        survivors = [population[front_idx[i]] for i in best]
    else:
        # take whole front, and if still < pop_size, fill with next best by crowding
        survivors = [population[i] for i in front_idx]
        if len(survivors) < pop_size:
            # get dominated ones
            rest = [i for i in range(N) if i not in front_idx]
            dist_rest = crowding_distance(loss_arr, rest)
            needed = pop_size - len(survivors)
            best_rest = np.argsort(-dist_rest)[:needed]
            survivors += [population[rest[i]] for i in best_rest]

    # --- 2) variation to refill ---
    new_pop = survivors.copy()
    while len(new_pop) < pop_size:
        # pick two parents
        p1, p2 = random.sample(survivors, 2)
        # SBX crossover
        if random.random() < crossover_rate:
            u = torch.rand_like(p1)
            beta = torch.where(u <= 0.5,
                               (2*u).pow(1.0/(eta+1)),
                               (1/(2*(1-u))).pow(1.0/(eta+1)))
            c1 = 0.5*((1+beta)*p1 + (1-beta)*p2)
            c2 = 0.5*((1-beta)*p1 + (1+beta)*p2)
        else:
            c1, c2 = p1.clone(), p2.clone()

        for child in (c1, c2):
            # mutation: add Gaussian noise
            if random.random() < mutation_rate:
                noise = torch.randn_like(child) * 0.05
                child.add_(noise)
            # project back to simplex
            child.clamp_(min=1e-6)
            child.div_(child.sum())
            new_pop.append(child)
            if len(new_pop) >= pop_size:
                break

    return new_pop[:pop_size]
