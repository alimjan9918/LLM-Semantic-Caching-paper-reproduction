"""Common online simulator; policies do not receive true costs or sizes."""
from __future__ import annotations
import numpy as np
import random
import time
from dataclasses import dataclass
from data import ReferenceProblem
from storage import OnlineStorageModel
from objective import cache_loss, nearest_distance
from estimation import sample_serving_cost


@dataclass
class OnlineResult:
    losses: np.ndarray
    switching_costs: np.ndarray
    actions: list[str]
    cache_sizes: np.ndarray
    memory_used: np.ndarray
    llm_calls: np.ndarray
    population_entries: np.ndarray
    switches: int
    mean_selection_runtime: float
    total_selection_runtime: float
    revealed_candidates: int
    serving_costs: np.ndarray
    query_stream: np.ndarray
    served_caches: list[list[int]]
    caches: list[list[int]]
    oracle_calls: int

    @property
    def cumulative_objective(self) -> np.ndarray:
        return np.cumsum(self.losses + self.switching_costs)

    @property
    def cumulative_realized_objective(self) -> np.ndarray:
        """Realized action costs plus modeled expected switching penalties."""
        return np.cumsum(self.serving_costs + self.switching_costs)

    def approximation_regret(self, empty_loss, optimal_loss, alpha):
        """Cache-loss alpha-regret including switching; negative values are valid."""
        if not 0 < alpha <= 1:
            raise ValueError("alpha must lie in (0, 1]")
        baseline = (1.0 - alpha) * empty_loss + alpha * optimal_loss
        rounds = np.arange(1, len(self.losses) + 1)
        return self.cumulative_objective - rounds * baseline


class OnlineAgent:
    """Shared serving, revelation and switching-cost accounting."""

    def __init__(
        self,
        problem: ReferenceProblem,
        storage: OnlineStorageModel,
        horizon: int,
        delta: float,
        *,
        gamma: float = 1.0,
    ):
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if not 0 < delta < 1:
            raise ValueError("delta must lie in (0, 1)")
        self.problem = problem
        self.storage = storage
        self.horizon = int(horizon)
        self.delta = float(delta)
        self.gamma = float(gamma)

        n = problem.m
        self.arrivals = np.zeros(n)
        self.observations = np.zeros(n)
        self.total_cost = np.zeros(n)
        self.cost_estimate = np.zeros(n)

        self.cache: set[int] = set()
        self.candidates: set[int] = set()
        self.storage_sizes = np.full(n, np.nan)
        self.oracle_calls = 0

    def step(
        self,
        t: int,
        q: int,
        rng: np.random.RandomState,
    ) -> dict:
        self.arrivals[q] += 1
        self._on_arrival(q)

        # An empty cache offers no answer, including when gamma is zero.
        mismatch = (self.gamma * nearest_distance(q, self.cache, self.problem.dist)
                    if self.cache else float("inf"))
        fresh = (
            self.observations[q] == 0
            or self.cost_estimate[q] < mismatch
        )
        llm_calls = 0

        if fresh:
            realized_cost = sample_serving_cost(float(self.problem.cost[q]), rng)
            self.storage_sizes[q] = self.storage.entry_size(q)
            self.candidates.add(q)
            self.observations[q] += 1
            self.total_cost[q] += realized_cost
            self._on_cost(q)
            llm_calls += 1
            action = "fresh"
        else:
            action = "cache"
            realized_cost = float(mismatch)

        loss = cache_loss(
            self.cache,
            self.problem.p,
            self.problem.cost,
            self.problem.dist,
            gamma=self.gamma,
        )

        switch = self._update_due()

        switching_cost = 0.0
        selection_runtime = 0.0
        cache_changed = False
        population_entries = 0
        if t < self.horizon and switch:
            selected, selection_runtime = self._switch_cache(t)
            newly_added = selected - self.cache
            switching_cost = float(sum(self.problem.cost[q] for q in newly_added))
            # Population has an abstract switching penalty; it supplies no
            # additional serving-cost observations or simulated service calls.
            population_entries = len(newly_added)
            cache_changed = selected != self.cache
            self.cache = selected

        return {
            "loss": float(loss),
            "serving_cost": realized_cost,
            "switching_cost": switching_cost,
            "action": action,
            "cache_size": len(self.cache),
            "memory_used": self.storage.used(self.cache),
            "llm_calls": llm_calls,
            "population_entries": population_entries,
            "cache_changed": cache_changed,
            "selection_runtime": selection_runtime,
        }

    def _on_arrival(self, q):
        pass

    def _on_cost(self, q):
        pass

    def _update_due(self):
        return True

    def _switch_cache(self, t):
        raise NotImplementedError


class PerRoundAgent(OnlineAgent):
    """Recompute a baseline cache after every served query."""

    def __init__(self, problem, storage, horizon, delta, *, policy, gamma=1.0):
        super().__init__(problem, storage, horizon, delta, gamma=gamma)
        self.policy = policy

    def _switch_cache(self, t):
        p_hat = self.arrivals / t
        self.cost_estimate = self.policy.estimate_costs(
            self.observations, self.total_cost, self.horizon, self.delta)
        start = time.perf_counter()
        self.oracle_calls += 1
        selected = self.policy.select_cache(p_hat, self.cost_estimate, self.problem.dist,
                                             self.storage_sizes, self.storage.budget,
                                             self.candidates)
        elapsed = time.perf_counter() - start
        if not selected <= self.candidates or not self.storage.feasible(selected):
            raise AssertionError("policy returned an unknown or infeasible cache")
        return selected, elapsed


def run_online_agent(
    problem: ReferenceProblem,
    storage: OnlineStorageModel,
    horizon: int,
    delta: float,
    seed: int,
    *,
    gamma: float = 1.0,
    agent=None,
    stream=None,
    paired_noise: bool = False,
) -> OnlineResult:
    """Serve one stream with common feedback, noise and switching accounting."""
    py_rng = random.Random(seed)
    rng = np.random.RandomState(seed)
    if stream is None:
        stream = py_rng.choices(range(problem.m), weights=problem.p, k=horizon)
    if len(stream) != horizon:
        raise ValueError('stream length must equal horizon')
    if agent is None:
        from algorithms.ombsc import OMBSC
        agent = OMBSC(problem, storage, horizon, delta, gamma=gamma)
    algo = agent
    query_rngs = [np.random.default_rng(np.random.SeedSequence([seed, q, 7919])) for q in range(problem.m)] if paired_noise else None

    losses = np.zeros(horizon)
    serving_costs = np.zeros(horizon)
    # A cache update at the end of round t is charged when M_{t+1} is used.
    switching_costs = np.zeros(horizon)
    actions: list[str] = []
    cache_sizes = np.zeros(horizon, dtype=int)
    memory_used = np.zeros(horizon)
    llm_calls = np.zeros(horizon, dtype=int)
    population_entries = np.zeros(horizon, dtype=int)
    selection_runtimes = np.zeros(horizon)
    switches = 0

    served_caches, caches = [], []
    for t, q in enumerate(stream, start=1):
        served_caches.append(sorted(algo.cache))
        result = algo.step(t, q, query_rngs[q] if paired_noise else rng)
        caches.append(sorted(algo.cache))
        losses[t - 1] = result["loss"]
        serving_costs[t - 1] = result["serving_cost"]
        if t < horizon:
            switching_costs[t] = result["switching_cost"]
        actions.append(result["action"])
        cache_sizes[t - 1] = result["cache_size"]
        memory_used[t - 1] = result["memory_used"]
        llm_calls[t - 1] = result["llm_calls"]
        population_entries[t - 1] = result["population_entries"]
        selection_runtimes[t - 1] = result["selection_runtime"]
        switches += int(result["cache_changed"])

    return OnlineResult(
        losses=losses,
        switching_costs=switching_costs,
        actions=actions,
        cache_sizes=cache_sizes,
        memory_used=memory_used,
        llm_calls=llm_calls,
        population_entries=population_entries,
        switches=switches,
        mean_selection_runtime=float(selection_runtimes.mean()),
        total_selection_runtime=float(selection_runtimes.sum()),
        revealed_candidates=len(algo.candidates),
        serving_costs=serving_costs,
        query_stream=np.asarray(stream, dtype=int),
        served_caches=served_caches,
        caches=caches,
        oracle_calls=algo.oracle_calls,
    )
