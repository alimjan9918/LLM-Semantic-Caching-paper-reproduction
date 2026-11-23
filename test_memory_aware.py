from dataclasses import dataclass

import numpy as np

from memory_aware import (
    MemoryLayout,
    best_singleton,
    brute_force_budget_loss,
    cache_gain,
    cache_loss,
    lazy_density_greedy,
    make_memory_layout,
    MemoizedGain,
    OMBSC,
    make_online_storage_model,
    ombsc_confidence_costs,
    offline_estimates_masked,
    run_ombsc,
    select_lazy_density,
)


@dataclass
class TinyProblem:
    p: np.ndarray
    cost: np.ndarray
    dist: np.ndarray
    tokens: np.ndarray

    @property
    def m(self):
        return len(self.p)


def make_problem(n=10, seed=7):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n, 4))
    dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    dist /= dist.max()
    np.fill_diagonal(dist, 0.0)
    tokens = rng.integers(2, 20, size=n).astype(float)
    return TinyProblem(
        p=np.full(n, 1.0 / n),
        cost=np.clip((tokens - tokens.min()) / np.ptp(tokens), 0.02, 1.0),
        dist=dist,
        tokens=tokens,
    )


def test_memory_layout_is_positive_and_respects_budget_fraction():
    prob = make_problem()
    layout = make_memory_layout(prob, 0.25, mode="query")
    assert np.all(layout.omega > 0)
    assert np.isclose(layout.budget, 0.25 * layout.total_memory)


def test_lazy_density_is_feasible_and_at_least_best_singleton():
    prob = make_problem()
    layout = make_memory_layout(prob, 0.20, mode="query")
    cache = select_lazy_density(prob.p, prob.cost, prob.dist, layout)
    singleton = best_singleton(prob.p, prob.cost, prob.dist, layout)
    assert layout.feasible(cache)
    assert cache_gain(cache, prob.p, prob.cost, prob.dist) >= (
        cache_gain(singleton, prob.p, prob.cost, prob.dist) - 1e-12
    )


def test_lazy_density_returns_empty_when_nothing_fits():
    prob = make_problem()
    omega = np.full(prob.m, 2.0)
    layout = MemoryLayout(omega, 1.5, "unit", 0.0)
    oracle = MemoizedGain(prob.p, prob.cost, prob.dist)
    assert lazy_density_greedy(oracle, omega, layout.budget) == set()


def test_lazy_density_near_exact_optimum_on_small_instance():
    prob = make_problem(n=10, seed=11)
    layout = make_memory_layout(prob, 0.30, mode="query")
    cache = select_lazy_density(prob.p, prob.cost, prob.dist, layout)
    optimum = brute_force_budget_loss(prob, layout)
    ratio = cache_loss(cache, prob.p, prob.cost, prob.dist) / optimum
    # A useful sanity threshold, not an approximation guarantee.
    assert ratio < 1.15


def test_masked_offline_estimates_keep_zero_cost_observation():
    stream = np.array([0, 0, 1, 1])
    observed_cost = np.array([0.0, 0.0, 0.4, 0.6])
    observed_mask = np.ones(4, dtype=bool)
    p_hat, c_hat, lcb, ucb, counts, seen = offline_estimates_masked(
        stream,
        observed_cost,
        observed_mask,
        m=2,
        delta=0.8,
    )
    assert np.allclose(p_hat, [0.5, 0.5])
    assert np.allclose(counts, [2, 2])
    assert seen[0]
    assert c_hat[0] == 0.0
    assert lcb[0] == 0.0
    assert ucb[0] > 0.0


def test_ombsc_storage_reveals_query_plus_answer_size():
    prob = make_problem()
    storage = make_online_storage_model(prob, 0.25, seed=3)
    assert np.allclose(
        storage.query_sizes + storage.answer_sizes,
        [storage.entry_size(q) for q in range(prob.m)],
    )
    assert np.isclose(storage.budget, 0.25 * storage.total_memory)


def test_ombsc_confidence_uses_cubic_horizon():
    observations = np.array([2.0, 0.0])
    total_cost = np.array([0.8, 0.0])
    horizon = 20
    n_queries = 2
    delta = 0.8

    result = ombsc_confidence_costs(
        observations,
        total_cost,
        horizon,
        n_queries,
        delta,
    )
    expected_radius = 0.05 * np.sqrt(
        2.0
        * np.log(4.0 * n_queries * horizon**3 / delta)
        / observations[0]
    )
    assert np.isclose(result[0], max(0.4 - expected_radius, 0.0))
    assert result[1] == 0.0


def test_ombsc_first_query_is_fresh_and_reveals_only_that_size():
    prob = make_problem()
    storage = make_online_storage_model(prob, 0.25, seed=3)
    algo = OMBSC(prob, storage, horizon=20, delta=0.8)
    rng = np.random.RandomState(5)
    result = algo.step(1, 0, rng)
    assert result["action"] == "fresh"
    assert 0 in algo.candidates
    assert np.isfinite(algo.storage_sizes[0])
    assert np.isnan(algo.storage_sizes[1])
    assert storage.feasible(algo.cache)


def test_ombsc_is_deterministic_and_respects_memory_budget():
    prob = make_problem()
    storage = make_online_storage_model(prob, 0.30, seed=3)
    first = run_ombsc(prob, storage, 30, 0.8, seed=5)
    second = run_ombsc(prob, storage, 30, 0.8, seed=5)
    assert np.array_equal(first.losses, second.losses)
    assert np.array_equal(first.switching_costs, second.switching_costs)
    assert first.actions == second.actions
    assert first.switches == second.switches
    assert np.all(first.memory_used <= storage.budget + 1e-9)
    assert np.all(np.isfinite(first.cumulative_objective))
