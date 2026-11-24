"""Stage-triggered Online Memory-Budgeted Semantic Caching."""
from __future__ import annotations
import numpy as np
import time
from online import OnlineAgent
from estimation import ombsc_confidence_costs
from algorithms.lazy_density_greedy import memory_budget_greedy_oracle
from algorithms.lazy_density_greedy import lazy_density_greedy
from algorithms.base import CachePolicy, eligible_candidates
from objective import MemoizedGain


class OMBSCPolicy(CachePolicy):
    """Offline LCB lazy greedy, or the stage-triggered online OMBSC agent."""
    estimate_kind = "lcb"

    def select_cache(self, p_hat, cost_hat, dist, storage_sizes, budget, candidates):
        entries = eligible_candidates(storage_sizes, budget, candidates)
        return lazy_density_greedy(MemoizedGain(p_hat, cost_hat, dist, gamma=self.gamma),
                                   storage_sizes, budget, candidates=entries)

    def make_agent(self, problem, storage, horizon, delta):
        if self.mode != "online":
            raise ValueError("make_agent requires mode='online'")
        return OMBSC(problem, storage, horizon, delta, gamma=self.gamma)


class OMBSC(OnlineAgent):
    """The stage schedule and optimistic lazy-density cache oracle."""

    def __init__(self, problem, storage, horizon, delta, *, gamma=1.0):
        super().__init__(problem, storage, horizon, delta, gamma=gamma)
        self.stage_cost = np.zeros(problem.m)
        self.history_cost = np.zeros(problem.m)
        self.stage_arrival = 0.0
        self.history_arrival = 0.0

    def _on_arrival(self, q):
        self.stage_arrival += 1.0

    def _on_cost(self, q):
        self.stage_cost[q] += 1.0

    def _update_due(self):
        return self._cost_stage_transition() or self._arrival_stage_transition()

    def _cost_stage_transition(self) -> bool:
        thresholds = 1.0 + np.sqrt(
            self.horizon * self.history_cost / self.problem.m
        )
        eligible = np.flatnonzero(
            (self.observations > 0)
            & (self.stage_cost >= thresholds - 1e-12)
        )
        if eligible.size == 0:
            return False

        # The paper allows any triggered query. Pick the largest overshoot for a
        # deterministic, low-switch implementation.
        q = int(max(eligible, key=lambda i: (self.stage_cost[i] - thresholds[i], -i)))
        self.history_cost[q] += self.stage_cost[q]
        self.stage_cost[q] = 0.0
        return True

    def _arrival_stage_transition(self) -> bool:
        threshold = 1.0 + np.sqrt(self.horizon * self.history_arrival)
        if self.stage_arrival < threshold - 1e-12:
            return False
        self.history_arrival += self.stage_arrival
        self.stage_arrival = 0.0
        return True

    def _switch_cache(self, t: int) -> tuple[set[int], float]:
        """Algorithm 2: refresh estimates and call the memory-budget oracle."""
        p_hat = self.arrivals / t
        self.cost_estimate = ombsc_confidence_costs(
            self.observations,
            self.total_cost,
            self.horizon,
            self.problem.m,
            self.delta,
        )
        start = time.perf_counter()
        self.oracle_calls += 1
        selected = memory_budget_greedy_oracle(
            self.problem,
            self.candidates,
            p_hat,
            self.cost_estimate,
            self.storage_sizes,
            self.storage.budget,
            gamma=self.gamma,
        )
        elapsed = time.perf_counter() - start
        if not self.storage.feasible(selected):
            raise AssertionError("OMBSC oracle returned an infeasible cache")
        return selected, elapsed
