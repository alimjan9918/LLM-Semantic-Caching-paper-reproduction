"""Lazy density greedy with its best-singleton safeguard."""
from __future__ import annotations
import numpy as np
import heapq
from typing import Callable, Iterable
from objective import MemoizedGain
from data import ReferenceProblem
GainOracle = Callable[[Iterable[int]], float]


def lazy_density_greedy(
    f: GainOracle,
    omega: np.ndarray,
    W: float,
    *,
    candidates: Iterable[int] | None = None,
    tol: float = 1e-12,
) -> set[int]:
    """Algorithm 1: lazy density greedy with singleton fallback.

    Items are considered by decreasing stale gain/memory density.  Whenever an
    item reaches the top of the heap, its marginal density is recomputed and
    either accepted, reinserted, or discarded if it can no longer fit.  The best
    feasible singleton is retained as a fallback against greedy path failures.
    """
    omega = np.asarray(omega, dtype=float)
    if omega.ndim != 1 or np.any(omega <= 0):
        raise ValueError("omega must be a one-dimensional positive vector")
    if W <= 0:
        return set()

    n = omega.size
    candidate_indices = (
        sorted({int(q) for q in candidates})
        if candidates is not None
        else list(range(n))
    )
    if any(q < 0 or q >= n for q in candidate_indices):
        raise ValueError("candidate index outside omega")
    best_single: list[int] = []
    best_single_gain = -np.inf
    heap: list[tuple[float, int]] = []

    for q in candidate_indices:
        if 0.0 < omega[q] <= W + tol:
            gain = float(f([q]))
            if gain > best_single_gain + 1e-14:
                best_single, best_single_gain = [q], gain
            heapq.heappush(heap, (-gain / omega[q], q))

    selected: list[int] = []
    in_cache = np.zeros(n, dtype=bool)
    used = 0.0

    while heap:
        neg_density, q = heapq.heappop(heap)
        if in_cache[q]:
            continue
        if used + omega[q] > W + tol:
            # Memory usage only increases, so this item can never fit later.
            continue

        current_gain = float(f(selected))
        marginal_gain = float(f(selected + [q])) - current_gain
        density = marginal_gain / omega[q]
        next_density = -heap[0][0] if heap else -np.inf

        if density + tol >= next_density:
            if density <= tol:
                break
            selected.append(q)
            in_cache[q] = True
            used += float(omega[q])
        else:
            heapq.heappush(heap, (-density, q))

    selected_set = set(selected)
    if best_single and float(f(selected_set)) + 1e-14 < best_single_gain:
        return set(best_single)
    return selected_set


def memory_budget_greedy_oracle(
    problem: ReferenceProblem,
    candidates: Iterable[int],
    p_hat: np.ndarray,
    cost_hat: np.ndarray,
    storage_sizes: np.ndarray,
    budget: float,
    *,
    gamma: float = 1.0,
) -> set[int]:
    """Algorithm 3: lazy greedy oracle over observed candidates V."""
    candidates = [
        int(q)
        for q in sorted(set(candidates))
        if np.isfinite(storage_sizes[q])
        and storage_sizes[q] > 0
        and storage_sizes[q] <= budget + 1e-12
    ]
    if not candidates:
        return set()

    oracle = MemoizedGain(p_hat, cost_hat, problem.dist, gamma=gamma)
    return lazy_density_greedy(
        oracle,
        storage_sizes,
        budget,
        candidates=candidates,
    )
