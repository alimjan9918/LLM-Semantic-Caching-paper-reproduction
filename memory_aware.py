"""Memory-budgeted semantic caching for heterogeneous query sizes.

This module adds the knapsack extension to the author reproduction without
changing its original fixed-cardinality experiments.  A cache M must satisfy

    sum_{q in M} omega(q) <= W,

where omega(q) is the positive memory cost of one cached query-response pair.
The objective remains the paper's expected loss

    L(M) = sum_q p(q) * min(c(q), min_{u in M} d(q, u)).

Offline selection uses lazy density greedy with the best feasible singleton.
Online selection follows OMBSC from "Semantic Caching under Memory Budgets for
LLM Serving": answer sizes are revealed only on the first LLM invocation,
cache updates are stage-triggered, and switching cost K_t is charged for newly
added query-response pairs.
"""

from __future__ import annotations

import heapq
import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from caching import SIGMA, Problem, confidence_radius, reverse_greedy

GainOracle = Callable[[Iterable[int]], float]


def _indices(cache: Iterable[int] | np.ndarray | None) -> np.ndarray:
    if cache is None:
        return np.empty(0, dtype=int)
    values = np.asarray(list(cache) if not isinstance(cache, np.ndarray) else cache)
    if values.size == 0:
        return np.empty(0, dtype=int)
    return np.asarray(sorted({int(q) for q in values}), dtype=int)


def cache_loss(
    cache: Iterable[int] | np.ndarray | None,
    p,
    cost,
    dist,
    gamma: float = 1.0,
) -> float:
    """Exact expected loss, including the empty-cache case."""
    idx = _indices(cache)
    if idx.size == 0:
        return float(np.dot(p, cost))
    nearest = dist[:, idx].min(axis=1)
    return float(np.sum(p * np.minimum(gamma * nearest, cost)))


def cache_gain(
    cache: Iterable[int] | np.ndarray | None,
    p,
    cost,
    dist,
    gamma: float = 1.0,
) -> float:
    """Loss reduction relative to an empty cache."""
    return float(
        np.dot(p, cost) - cache_loss(cache, p, cost, dist, gamma=gamma)
    )


class MemoizedGain:
    """Memoised gain oracle used by the lazy greedy algorithm."""

    def __init__(self, p, cost, dist, gamma: float = 1.0):
        self.p = np.asarray(p, dtype=float)
        self.cost = np.asarray(cost, dtype=float)
        self.dist = np.asarray(dist, dtype=float)
        self.gamma = float(gamma)
        self.calls = 0
        self.unique_calls = 0
        self._memo: dict[frozenset[int], float] = {}

    def __call__(self, cache: Iterable[int] | np.ndarray | None) -> float:
        self.calls += 1
        key = frozenset(int(q) for q in _indices(cache))
        value = self._memo.get(key)
        if value is None:
            value = cache_gain(
                key,
                self.p,
                self.cost,
                self.dist,
                gamma=self.gamma,
            )
            self._memo[key] = value
            self.unique_calls += 1
        return value


@dataclass(frozen=True)
class MemoryLayout:
    """Memory size of each cache entry and the total memory budget."""

    omega: np.ndarray
    budget: float
    mode: str
    budget_fraction: float

    def __post_init__(self):
        omega = np.asarray(self.omega, dtype=float)
        if omega.ndim != 1 or np.any(omega <= 0):
            raise ValueError("omega must be a one-dimensional positive vector")
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        object.__setattr__(self, "omega", omega)

    @property
    def total_memory(self) -> float:
        return float(self.omega.sum())

    def used(self, cache: Iterable[int] | np.ndarray | None) -> float:
        return float(self.omega[_indices(cache)].sum())

    def feasible(self, cache: Iterable[int] | np.ndarray | None, tol: float = 1e-9) -> bool:
        return self.used(cache) <= self.budget + tol


def _normalise_mean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x * (1.0 / max(float(x.mean()), 1e-12))


def make_memory_layout(
    problem: Problem,
    budget_fraction: float = 0.25,
    *,
    mode: str = "query",
    memory_noise: float = 0.45,
    seed: int = 0,
) -> MemoryLayout:
    """Build heterogeneous cache-entry sizes from the author query set.

    Modes
    -----
    query
        Store only the GPT-2 token count of the query.
    response
        Use the response-size proxy from ``semcache-knapsack``:
        ``(tokens + 1) ** 1.15 * LogNormal(0, memory_noise)``.
    entry
        Store both prompt and response proxies.

    The sizes are normalised to mean one, so a budget fraction of 0.25 means
    that the cache may occupy 25% of the total memory of all candidate entries.
    """
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must lie in (0, 1]")
    if mode not in {"query", "response", "entry"}:
        raise ValueError("mode must be one of: query, response, entry")

    tokens = np.asarray(problem.tokens, dtype=float)
    if mode == "query":
        raw = tokens
    else:
        rng = np.random.default_rng(seed)
        response = (tokens + 1.0) ** 1.15 * rng.lognormal(
            mean=0.0, sigma=memory_noise, size=tokens.size
        )
        raw = response if mode == "response" else tokens + response

    omega = _normalise_mean(np.maximum(raw, 1e-12))
    return MemoryLayout(
        omega=omega,
        budget=float(budget_fraction * omega.sum()),
        mode=mode,
        budget_fraction=float(budget_fraction),
    )


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


def select_lazy_density(
    p,
    cost,
    dist,
    layout: MemoryLayout,
    *,
    gamma: float = 1.0,
    candidates: Iterable[int] | None = None,
) -> set[int]:
    """Select a cache from known or estimated parameters."""
    oracle = MemoizedGain(p, cost, dist, gamma=gamma)
    return lazy_density_greedy(
        oracle,
        layout.omega,
        layout.budget,
        candidates=candidates,
    )


def memory_blind_reverse_greedy(
    p,
    cost,
    dist,
    layout: MemoryLayout,
) -> set[int]:
    """Paper baseline with k=floor(W/mean(omega)), followed by repair."""
    k = max(0, min(len(p), int(layout.budget / layout.omega.mean())))
    cache = set(reverse_greedy(k, p, cost, dist))
    oracle = MemoizedGain(p, cost, dist)

    while cache and layout.used(cache) > layout.budget + 1e-9:
        base_gain = float(oracle(cache))
        worst = min(
            cache,
            key=lambda q: (
                base_gain - float(oracle(cache - {q}))
            ) / layout.omega[q],
        )
        cache.remove(worst)

    if not cache:
        eligible = [q for q in range(len(p)) if layout.omega[q] <= layout.budget + 1e-12]
        if eligible:
            cache = {max(eligible, key=lambda q: (float(oracle([q])), -q))}
    return cache


def best_singleton(p, cost, dist, layout: MemoryLayout) -> set[int]:
    oracle = MemoizedGain(p, cost, dist)
    eligible = [
        q for q in range(len(p)) if layout.omega[q] <= layout.budget + 1e-12
    ]
    if not eligible:
        return set()
    q = max(eligible, key=lambda i: (float(oracle([i])), -i))
    return {q}


def log_dataset(
    problem: Problem,
    length: int,
    k: int,
    py_rng,
    rng,
):
    """Create an offline log with an explicit observed/missing cost mask.

    The author code encodes a missing cost as ``0.0`` and later tests
    ``cost > 0``.  That loses observations for the shortest query after
    min-max normalisation.  This version keeps the mask separately and always
    uses the problem's global serving cost for a given query.
    """
    m = problem.m
    stream = np.asarray([py_rng.randrange(m) for _ in range(length)], dtype=int)
    cache = set(py_rng.sample(range(m), k))
    observed = np.zeros(length, dtype=bool)
    observed_cost = np.full(length, np.nan, dtype=float)

    for i, q in enumerate(stream):
        if q not in cache:
            observed[i] = True
            observed_cost[i] = np.clip(
                rng.normal(problem.cost[q], SIGMA),
                0.0,
                1.0,
            )
        if q not in cache and py_rng.random() < 0.5:
            cache.discard(py_rng.choice(list(cache)))
            cache.add(int(q))

    return stream, observed_cost, observed


def offline_estimates_masked(
    stream,
    observed_cost,
    observed_mask,
    m: int,
    delta: float,
):
    """Empirical probabilities and confidence bounds with partial feedback."""
    stream = np.asarray(stream, dtype=int)
    observed_cost = np.asarray(observed_cost, dtype=float)
    observed_mask = np.asarray(observed_mask, dtype=bool)
    n = len(stream)
    counts = np.bincount(stream, minlength=m)
    observations = np.bincount(
        stream,
        weights=observed_mask.astype(float),
        minlength=m,
    )
    total = np.bincount(
        stream,
        weights=np.where(observed_mask, observed_cost, 0.0),
        minlength=m,
    )
    seen = observations > 0
    c_hat = np.divide(total, observations, out=np.zeros(m), where=seen)
    radius = confidence_radius(np.maximum(observations, 1), m, n, delta)
    lcb = np.where(seen, np.maximum(c_hat - radius, 0.0), 0.0)
    ucb = np.where(seen, np.minimum(c_hat + radius, 1.0), 1.0)
    return counts / n, c_hat, lcb, ucb, counts, seen


def offline_memory_caches(
    problem: Problem,
    layout: MemoryLayout,
    stream,
    observed_cost,
    observed_mask,
    delta: float,
) -> dict[str, set[int]]:
    """Offline size-aware variants built from one partial-feedback log."""
    p_hat, c_hat, lcb, ucb, _, _ = offline_estimates_masked(
        stream,
        observed_cost,
        observed_mask,
        problem.m,
        delta,
    )
    return {
        "Lazy-Density-CLCB": select_lazy_density(p_hat, lcb, problem.dist, layout),
        "Lazy-Density-CUCB": select_lazy_density(p_hat, ucb, problem.dist, layout),
        "Lazy-Density-Mean": select_lazy_density(p_hat, c_hat, problem.dist, layout),
        "Memory-Blind-RG": memory_blind_reverse_greedy(
            p_hat, c_hat, problem.dist, layout
        ),
        "Best-Singleton": best_singleton(p_hat, c_hat, problem.dist, layout),
    }


def nearest_distance(q: int, cache: Iterable[int], dist) -> float:
    idx = _indices(cache)
    return float("inf") if idx.size == 0 else float(dist[q, idx].min())


@dataclass(frozen=True)
class OnlineStorageModel:
    """Hidden query/answer sizes used to reveal ``S(q)=L(q)+A(q)`` online."""

    query_sizes: np.ndarray
    answer_sizes: np.ndarray
    budget: float
    budget_fraction: float
    memory_noise: float

    def __post_init__(self):
        query_sizes = np.asarray(self.query_sizes, dtype=float)
        answer_sizes = np.asarray(self.answer_sizes, dtype=float)
        if query_sizes.ndim != 1 or answer_sizes.shape != query_sizes.shape:
            raise ValueError("query_sizes and answer_sizes must be equally long vectors")
        if np.any(query_sizes <= 0) or np.any(answer_sizes <= 0):
            raise ValueError("query and answer sizes must be positive")
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        object.__setattr__(self, "query_sizes", query_sizes)
        object.__setattr__(self, "answer_sizes", answer_sizes)

    @property
    def total_memory(self) -> float:
        return float((self.query_sizes + self.answer_sizes).sum())

    def entry_size(self, q: int) -> float:
        return float(self.query_sizes[q] + self.answer_sizes[q])

    def used(self, cache: Iterable[int] | np.ndarray | None) -> float:
        return float(sum(self.entry_size(q) for q in _indices(cache)))

    def feasible(
        self,
        cache: Iterable[int] | np.ndarray | None,
        tol: float = 1e-9,
    ) -> bool:
        return self.used(cache) <= self.budget + tol


def make_online_storage_model(
    problem: Problem,
    budget_fraction: float = 0.25,
    *,
    memory_noise: float = 0.45,
    seed: int = 0,
) -> OnlineStorageModel:
    """Create the paper's hidden storage model.

    ``L(q)`` is the GPT-2 query length. ``A(q)`` is a latent response-size proxy;
    the online algorithm does not observe it until it invokes the LLM for q.
    """
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must lie in (0, 1]")
    if memory_noise < 0:
        raise ValueError("memory_noise must be non-negative")

    tokens = np.asarray(problem.tokens, dtype=float)
    rng = np.random.default_rng(seed)
    answer_raw = (tokens + 1.0) ** 1.15 * rng.lognormal(
        mean=0.0,
        sigma=memory_noise,
        size=tokens.size,
    )
    raw_total = tokens + answer_raw
    scale = 1.0 / max(float(raw_total.mean()), 1e-12)
    query_sizes = np.maximum(tokens * scale, 1e-12)
    answer_sizes = np.maximum(answer_raw * scale, 1e-12)
    budget = float(budget_fraction * (query_sizes + answer_sizes).sum())
    return OnlineStorageModel(
        query_sizes=query_sizes,
        answer_sizes=answer_sizes,
        budget=budget,
        budget_fraction=float(budget_fraction),
        memory_noise=float(memory_noise),
    )


def ombsc_confidence_costs(
    observations: np.ndarray,
    total_cost: np.ndarray,
    horizon: int,
    n_queries: int,
    delta: float,
) -> np.ndarray:
    """Equation (4): optimistic cost in the online OMBSC algorithm."""
    observations = np.asarray(observations, dtype=float)
    total_cost = np.asarray(total_cost, dtype=float)
    seen = observations > 0
    c_hat = np.divide(
        total_cost,
        observations,
        out=np.zeros_like(total_cost, dtype=float),
        where=seen,
    )
    radius = SIGMA * np.sqrt(
        2.0
        * np.log(4.0 * n_queries * horizon**3 / delta)
        / np.maximum(observations, 1.0)
    )
    return np.where(seen, np.maximum(c_hat - radius, 0.0), 0.0)


def memory_budget_greedy_oracle(
    problem: Problem,
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


@dataclass
class OMBSCResult:
    losses: np.ndarray
    switching_costs: np.ndarray
    actions: list[str]
    cache_sizes: np.ndarray
    memory_used: np.ndarray
    llm_calls: np.ndarray
    switches: int
    mean_selection_runtime: float
    total_selection_runtime: float
    revealed_candidates: int

    @property
    def cumulative_objective(self) -> np.ndarray:
        return np.cumsum(self.losses + self.switching_costs)


class OMBSC:
    """Online Memory-Budgeted Semantic Caching, Algorithms 1-3."""

    def __init__(
        self,
        problem: Problem,
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

        self.stage_cost = np.zeros(n)
        self.history_cost = np.zeros(n)
        self.stage_arrival = 0.0
        self.history_arrival = 0.0

        self.cache: set[int] = set()
        self.candidates: set[int] = set()
        self.storage_sizes = np.full(n, np.nan)

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

    def step(
        self,
        t: int,
        q: int,
        rng: np.random.RandomState,
    ) -> dict:
        self.arrivals[q] += 1
        self.stage_arrival += 1.0

        mismatch = self.gamma * nearest_distance(q, self.cache, self.problem.dist)
        fresh = (
            self.observations[q] == 0
            or self.cost_estimate[q] < mismatch
        )
        llm_calls = 0

        if fresh:
            realized_cost = float(
                np.clip(
                    rng.normal(self.problem.cost[q], SIGMA),
                    0.0,
                    1.0,
                )
            )
            self.storage_sizes[q] = self.storage.entry_size(q)
            self.candidates.add(q)
            self.observations[q] += 1
            self.total_cost[q] += realized_cost
            self.stage_cost[q] += 1.0
            llm_calls += 1
            action = "fresh"
        else:
            action = "cache"

        loss = cache_loss(
            self.cache,
            self.problem.p,
            self.problem.cost,
            self.problem.dist,
            gamma=self.gamma,
        )

        switch = self._cost_stage_transition()
        if not switch:
            switch = self._arrival_stage_transition()

        switching_cost = 0.0
        selection_runtime = 0.0
        cache_changed = False
        if t < self.horizon and switch:
            selected, selection_runtime = self._switch_cache(t)
            newly_added = selected - self.cache
            switching_cost = float(sum(self.problem.cost[q] for q in newly_added))
            llm_calls += len(newly_added)
            cache_changed = selected != self.cache
            self.cache = selected

        return {
            "loss": float(loss),
            "switching_cost": switching_cost,
            "action": action,
            "cache_size": len(self.cache),
            "memory_used": self.storage.used(self.cache),
            "llm_calls": llm_calls,
            "cache_changed": cache_changed,
            "selection_runtime": selection_runtime,
        }


def run_ombsc(
    problem: Problem,
    storage: OnlineStorageModel,
    horizon: int,
    delta: float,
    seed: int,
    *,
    gamma: float = 1.0,
) -> OMBSCResult:
    """Run Algorithms 1-3 for one online query stream."""
    py_rng = random.Random(seed)
    rng = np.random.RandomState(seed)
    stream = [py_rng.randrange(problem.m) for _ in range(horizon)]
    algo = OMBSC(problem, storage, horizon, delta, gamma=gamma)

    losses = np.zeros(horizon)
    # A cache update at the end of round t is charged when M_{t+1} is used.
    switching_costs = np.zeros(horizon)
    actions: list[str] = []
    cache_sizes = np.zeros(horizon, dtype=int)
    memory_used = np.zeros(horizon)
    llm_calls = np.zeros(horizon, dtype=int)
    selection_runtimes = np.zeros(horizon)
    switches = 0

    for t, q in enumerate(stream, start=1):
        result = algo.step(t, q, rng)
        losses[t - 1] = result["loss"]
        if t < horizon:
            switching_costs[t] = result["switching_cost"]
        actions.append(result["action"])
        cache_sizes[t - 1] = result["cache_size"]
        memory_used[t - 1] = result["memory_used"]
        llm_calls[t - 1] = result["llm_calls"]
        selection_runtimes[t - 1] = result["selection_runtime"]
        switches += int(result["cache_changed"])

    return OMBSCResult(
        losses=losses,
        switching_costs=switching_costs,
        actions=actions,
        cache_sizes=cache_sizes,
        memory_used=memory_used,
        llm_calls=llm_calls,
        switches=switches,
        mean_selection_runtime=float(selection_runtimes.mean()),
        total_selection_runtime=float(selection_runtimes.sum()),
        revealed_candidates=len(algo.candidates),
    )


def brute_force_budget_loss(problem: Problem, layout: MemoryLayout) -> float:
    """Exact optimum for small test instances."""
    best = float(np.dot(problem.p, problem.cost))
    n = problem.m
    for mask in range(1 << n):
        cache = [q for q in range(n) if mask & (1 << q)]
        if layout.used(cache) <= layout.budget + 1e-12:
            best = min(best, cache_loss(cache, problem.p, problem.cost, problem.dist))
    return best
