"""Shared offline/online policy interface; receives estimates, never true p/c."""
from abc import ABC, abstractmethod

import numpy as np

from estimation import SIGMA


def eligible_candidates(storage_sizes, budget, candidates):
    """Filter only the recorded sizes of candidate entries."""
    return sorted(q for q in set(candidates)
                  if np.isfinite(storage_sizes[q]) and 0 < storage_sizes[q] <= budget + 1e-12)


class CachePolicy(ABC):
    """One cache-selection policy usable with a supplied log or online state."""
    estimate_kind = "mean"

    def __init__(self, *, mode="online", seed=0, gamma=1.0):
        if mode not in {"online", "offline"}:
            raise ValueError("mode must be 'online' or 'offline'")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.mode = mode
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)

    def estimate_probabilities(self, arrivals, t, delta, *, horizon=None):
        return np.asarray(arrivals, dtype=float) / t

    def estimate_costs(self, observations, total_cost, horizon, delta):
        mean = np.divide(total_cost, observations, out=np.zeros_like(total_cost, dtype=float),
                         where=observations > 0)
        if self.estimate_kind == "mean":
            return mean
        power = 3 if self.mode == "online" else 1
        radius = SIGMA * np.sqrt(2 * np.log(4 * len(mean) * horizon**power / delta)
                                 / np.maximum(observations, 1))
        if self.estimate_kind == "lcb":
            return np.where(observations > 0, np.maximum(mean - radius, 0), 0.0)
        return np.where(observations > 0, np.minimum(mean + radius, 1), 1.0)

    def fit(self, stream, costs, observed_mask, dist, storage_sizes, budget, *, delta=0.1):
        """Select once from a partial-feedback log, in offline mode only.

        Sizes may be supplied as NaN for queries without an observed answer.
        Even if more sizes are supplied, only observed-answer entries qualify.
        """
        if self.mode != "offline":
            raise ValueError("fit requires mode='offline'")
        stream = np.asarray(stream, dtype=int)
        costs = np.asarray(costs, dtype=float)
        mask = np.asarray(observed_mask, dtype=bool)
        n = len(dist)
        if not len(stream) or stream.shape != costs.shape or mask.shape != stream.shape:
            raise ValueError("offline log must contain equally long nonempty arrays")
        if not 0 < delta < 1 or np.any(stream < 0) or np.any(stream >= n):
            raise ValueError("invalid delta or query indices")
        if np.any(~np.isfinite(costs[mask])):
            raise ValueError("observed costs must be finite")
        arrivals = np.bincount(stream, minlength=n)
        observations = np.bincount(stream, weights=mask, minlength=n)
        totals = np.bincount(stream, weights=np.where(mask, costs, 0.0), minlength=n)
        cost_hat = self.estimate_costs(observations, totals, len(stream), delta)
        p_hat = self.estimate_probabilities(arrivals, len(stream), delta)
        return self.select_cache(p_hat, cost_hat, dist, storage_sizes,
                                 budget, set(np.flatnonzero(observations)))

    def make_agent(self, problem, storage, horizon, delta):
        if self.mode != "online":
            raise ValueError("make_agent requires mode='online'")
        from online import PerRoundAgent
        return PerRoundAgent(problem, storage, horizon, delta, policy=self, gamma=self.gamma)

    @abstractmethod
    def select_cache(self, p_hat, cost_hat, dist, storage_sizes, budget, candidates):
        """Select only from known candidates under the memory budget."""
