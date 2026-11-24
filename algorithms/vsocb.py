"""VSOCB (Yang et al., EMNLP 2026), adapted to mismatch-aware serving.

Source: https://aclanthology.org/2026.emnlp-main.1849.pdf
Retains Eqs. (2)-(4), Algorithms 1-2 and additive knapsack recommendations.
Semantic reuse changes service decisions and the resulting partial feedback;
the paper's exact-hit regret guarantee is not claimed for this adaptation.
"""
import time

import numpy as np

from algorithms.base import CachePolicy
from algorithms.knapsack_oracle import exact_knapsack
from online import OnlineAgent


class VSOCBPolicy(CachePolicy):
    """Offline confidence/knapsack selection or online recommend-and-wait."""
    estimate_kind = "lcb"

    def __init__(self, *, growth=1.0, **kwargs):
        super().__init__(**kwargs)
        if not np.isfinite(growth) or growth <= 0:
            raise ValueError("VSOCB growth must be finite and positive")
        self.growth = float(growth)

    def estimate_costs(self, observations, total_cost, horizon, delta):
        """Equation (2), cost range c2-c1=1 for the common [0,1] model."""
        mean = np.divide(total_cost, observations, out=np.zeros_like(total_cost, dtype=float),
                         where=observations > 0)
        radius = np.sqrt(np.log(8 * horizon * len(mean) / delta)
                         / (2 * np.maximum(observations, 1)))
        return np.where(observations > 0, np.maximum(mean - radius, 0.0), 0.0)

    def estimate_probabilities(self, arrivals, t, delta, *, horizon=None):
        """Equations (3)-(4); probability LCBs need not sum to one."""
        horizon = t if horizon is None else horizon
        mean = np.asarray(arrivals, dtype=float) / t
        variance = np.maximum(mean * (1 - mean), 0.0)
        logterm = np.log(16 * horizon * len(mean) / delta)
        radius = np.sqrt(3 * variance * logterm / t) + 5 * logterm / t
        return np.maximum(mean - radius, 0.0)

    def select_cache(self, p_hat, cost_hat, dist, storage_sizes, budget, candidates):
        # Preserve the original additive surrogate; dist affects serving only.
        return exact_knapsack(np.asarray(p_hat) * cost_hat, storage_sizes, budget,
                              candidates=candidates)

    def make_agent(self, problem, storage, horizon, delta):
        if self.mode != "online":
            raise ValueError("make_agent requires mode='online'")
        return VSOCB(problem, storage, horizon, delta, policy=self, gamma=self.gamma)


class VSOCB(OnlineAgent):
    """Algorithm 1 with a separate recommendation and active cache."""

    def __init__(self, problem, storage, horizon, delta, *, policy, gamma=1.0):
        super().__init__(problem, storage, horizon, delta, gamma=gamma)
        self.policy = policy
        self.recommended_cache = set()
        self.reference_observations = np.zeros(problem.m)
        self.last_oracle_round = 0
        self.current_query = None
        self.current_answer_available = False

    def _on_arrival(self, q):
        self.current_query = q
        # A similar query's answer is not the incoming query's own answer.
        self.current_answer_available = q in self.cache

    def _on_cost(self, q):
        self.current_answer_available = True

    def _oracle_due(self, t, q):
        factor = 1 + self.policy.growth
        return (self.observations[q] >= factor * self.reference_observations[q]
                or t >= factor * self.last_oracle_round)

    def _switch_cache(self, t):
        """Recommend globally, admit locally, and reserve filler memory."""
        q = self.current_query
        self.cost_estimate = self.policy.estimate_costs(
            self.observations, self.total_cost, self.horizon, self.delta)
        p_lcb = self.policy.estimate_probabilities(self.arrivals, t, self.delta,
                                                  horizon=self.horizon)
        start = time.perf_counter()
        previous_cache = set(self.cache)
        selected = set(previous_cache)
        recommended = set(self.recommended_cache)

        # Algorithm 1 lines 16-20: recommend-and-wait, including unused space.
        # A semantic reuse cannot supply a missing answer for admission.
        if self.current_answer_available:
            used = sum(self.storage_sizes[item] for item in recommended)
            if (q in recommended or self.storage_sizes[q] <= self.storage.budget - used + 1e-12):
                selected.add(q)
                recommended.add(q)

        if self._oracle_due(t, q):
            self.reference_observations[q] = self.observations[q]
            self.last_oracle_round = t
            self.oracle_calls += 1
            recommended = self.policy.select_cache(
                p_lcb, self.cost_estimate, self.problem.dist, self.storage_sizes,
                self.storage.budget, self.candidates)
            # Line 23 retains only entries of the preceding active cache.
            # Newly recommended answers wait for a future own-query arrival.
            selected = recommended & previous_cache

        # Retain line 17's filler reservation when no new oracle runs. Without
        # this reservation repeated fillers could exceed the active budget.
        self.recommended_cache = recommended
        if (not selected <= recommended or not selected <= (previous_cache | {q})
                or not selected <= self.candidates or not self.storage.feasible(selected)
                or sum(self.storage_sizes[item] for item in recommended) > self.storage.budget + 1e-9):
            raise AssertionError("VSOCB violated recommendation, availability or memory constraints")
        return selected, time.perf_counter() - start
