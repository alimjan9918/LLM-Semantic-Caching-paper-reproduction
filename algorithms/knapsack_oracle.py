"""Exact additive 0/1 knapsack for real sizes, without storage discretization."""
import numpy as np

from algorithms.base import eligible_candidates


def _subsets(values, sizes):
    count = 1 << len(values)
    weights, rewards = np.zeros(count), np.zeros(count)
    for i, (value, size) in enumerate(zip(values, sizes)):
        block = 1 << i
        weights[block:2 * block] = weights[:block] + size
        rewards[block:2 * block] = rewards[:block] + value
    return weights, rewards


def exact_knapsack(values, storage_sizes, budget, *, candidates=None):
    """Maximize sum(values[q]) under the original floating-point budget.

    Meet in the middle: O(N * 2**(N/2)) time and O(2**(N/2)) space.
    At most 40 feasible candidates; larger instances fail explicitly.
    Equal rewards favor greater occupied memory, then a deterministic subset.
    """
    values = np.asarray(values, dtype=float)
    sizes = np.asarray(storage_sizes, dtype=float)
    if values.ndim != 1 or values.shape != sizes.shape:
        raise ValueError("values and sizes must be equally long vectors")
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("budget must be finite and non-negative")
    entries = eligible_candidates(sizes, budget, range(len(sizes)) if candidates is None else candidates)
    if len(entries) > 40:
        raise ValueError("exact modular knapsack supports at most 40 feasible candidates")
    if not entries:
        return set()
    if np.any(~np.isfinite(values[entries])) or np.any(values[entries] < 0):
        raise ValueError("item values must be finite and non-negative")
    n = len(entries)
    split = n // 2
    weights, rewards = _subsets(values[entries[:split]], sizes[entries[:split]])
    right_weights, right_rewards = _subsets(values[entries[split:]], sizes[entries[split:]])
    order = np.argsort(right_weights, kind="stable")
    right_weights, right_rewards = right_weights[order], right_rewards[order]
    prefix_max = np.maximum.accumulate(right_rewards)
    # Updating on reward ties picks the greatest feasible memory weight.
    prefix_best = np.maximum.accumulate(np.where(right_rewards == prefix_max,
                                                 np.arange(len(order)), 0))
    left_masks = np.flatnonzero(weights <= budget + 1e-12)
    capacity = budget + 1e-12 - weights[left_masks]
    right_limit = np.searchsorted(right_weights, capacity, side="right") - 1
    best_right = prefix_best[right_limit]
    total_reward = rewards[left_masks] + right_rewards[best_right]
    total_weight = weights[left_masks] + right_weights[best_right]
    tied_weight = np.where(total_reward == total_reward.max(), total_weight, -np.inf)
    best = int(np.argmax(tied_weight))
    left_mask, right_mask = int(left_masks[best]), int(order[best_right[best]])
    return ({entries[i] for i in range(split) if left_mask & (1 << i)}
            | {entries[split + i] for i in range(n - split) if right_mask & (1 << i)})
