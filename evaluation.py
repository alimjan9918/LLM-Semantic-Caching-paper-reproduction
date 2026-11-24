"""Exact semantic optimum for small evaluation instances only."""
from __future__ import annotations
import numpy as np
def exact_budgeted_losses(problem, storage_sizes, budgets, *, gamma=1.0):
    """Exact full-Q evaluation benchmark; never supplied to the learner.

    Enumerate all subsets in O(N*2**N) work and O(2**N) memory, sharing the
    enumeration across budgets. Refuse large instances rather than silently
    substituting an approximate optimum.
    """
    n = problem.m
    if n > 22:
        raise ValueError("exact benchmark supports at most 22 queries")
    sizes = np.asarray(storage_sizes, dtype=float)
    budgets = np.asarray(budgets, dtype=float)
    if sizes.shape != (n,) or not np.all(np.isfinite(sizes)) or np.any(sizes <= 0):
        raise ValueError("benchmark requires all positive storage sizes")
    if budgets.ndim != 1 or np.any(budgets < 0):
        raise ValueError("budgets must be a non-negative vector")
    count = 1 << n
    memory = np.zeros(count)
    for i in range(n):
        block = 1 << i
        memory[block:2 * block] = memory[:block] + sizes[i]
    losses = np.zeros(count)
    nearest = np.empty(count)
    for q in range(n):
        nearest[0] = problem.cost[q]
        for i in range(n):
            block = 1 << i
            np.minimum(nearest[:block], gamma * problem.dist[q, i],
                       out=nearest[block:2 * block])
        losses += problem.p[q] * nearest
    return np.array([losses[memory <= b + 1e-12].min() for b in budgets])
