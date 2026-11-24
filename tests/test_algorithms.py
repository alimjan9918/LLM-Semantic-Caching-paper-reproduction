"""Current selectors, storage and frozen online behavior, without legacy code."""
import json
from pathlib import Path
import unittest

import numpy as np

from algorithms.lazy_density_greedy import lazy_density_greedy
from algorithms.ombsc import OMBSC
from algorithms.registry import create_policy
from estimation import ombsc_confidence_costs
from evaluation import exact_budgeted_losses
from objective import MemoizedGain, cache_gain, cache_loss
from online import run_online_agent
from storage import make_online_storage_model
from tests.helpers import make_problem

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class AlgorithmTests(unittest.TestCase):
    def test_storage_includes_query_and_answer(self):
        prob = make_problem()
        storage = make_online_storage_model(prob, 0.25, seed=3)
        self.assertTrue(np.all(storage.query_sizes > 0))
        self.assertTrue(np.all(storage.answer_sizes > 0))
        np.testing.assert_allclose(storage.query_sizes + storage.answer_sizes,
                                   [storage.entry_size(q) for q in range(prob.m)])
        self.assertAlmostEqual(storage.budget, 0.25 * storage.total_memory)

    def test_greedy_is_feasible_and_at_least_best_singleton(self):
        prob = make_problem()
        storage = make_online_storage_model(prob, 0.2)
        sizes = storage.query_sizes + storage.answer_sizes
        gain = MemoizedGain(prob.p, prob.cost, prob.dist)
        cache = lazy_density_greedy(gain, sizes, storage.budget)
        singleton_gain = max([0.] + [gain({q}) for q in range(prob.m) if sizes[q] <= storage.budget])
        self.assertTrue(storage.feasible(cache))
        self.assertGreaterEqual(gain(cache) + 1e-12, singleton_gain)

    def test_greedy_returns_empty_when_nothing_fits(self):
        prob = make_problem()
        gain = MemoizedGain(prob.p, prob.cost, prob.dist)
        self.assertEqual(lazy_density_greedy(gain, np.full(prob.m, 2.), 1.5), set())

    def test_greedy_gain_respects_the_conservative_ratio_on_small_instances(self):
        for seed in (2, 7, 11):
            prob = make_problem(n=8, seed=seed)
            storage = make_online_storage_model(prob, 0.3, seed=seed)
            sizes = storage.query_sizes + storage.answer_sizes
            cache = lazy_density_greedy(MemoizedGain(prob.p, prob.cost, prob.dist), sizes, storage.budget)
            optimal_loss = exact_budgeted_losses(prob, sizes, [storage.budget])[0]
            optimal_gain = cache_loss(set(), prob.p, prob.cost, prob.dist) - optimal_loss
            self.assertGreaterEqual(cache_gain(cache, prob.p, prob.cost, prob.dist) + 1e-12,
                                    (1 - np.exp(-1)) / 2 * optimal_gain)

    def test_ombsc_confidence_keeps_the_cubic_horizon_term(self):
        result = ombsc_confidence_costs(np.array([2., 0.]), np.array([0.8, 0.]), 20, 2, 0.8)
        radius = 0.05 * np.sqrt(2 * np.log(4 * 2 * 20**3 / 0.8) / 2)
        np.testing.assert_allclose(result, [max(0.4 - radius, 0.), 0.])

    def test_first_invocation_reveals_only_its_own_size(self):
        prob = make_problem()
        storage = make_online_storage_model(prob, 0.25, seed=3)
        agent = OMBSC(prob, storage, horizon=20, delta=0.8)
        result = agent.step(1, 0, np.random.default_rng(5))
        self.assertEqual(result["action"], "fresh")
        self.assertEqual(agent.candidates, {0})
        self.assertTrue(np.isfinite(agent.storage_sizes[0]))
        self.assertTrue(np.all(np.isnan(agent.storage_sizes[1:])))
        self.assertTrue(storage.feasible(agent.cache))

    def test_ombsc_matches_frozen_reference(self):
        expected = json.loads((FIXTURES / "ombsc_reference.json").read_text())
        prob = make_problem(n=6)
        storage = make_online_storage_model(prob, 0.3, seed=3)
        result = run_online_agent(prob, storage, 40, 0.025, seed=5)
        for key in ("losses", "switching_costs", "serving_costs", "query_stream", "memory_used", "llm_calls"):
            np.testing.assert_allclose(getattr(result, key), expected[key], atol=1e-13)
        self.assertEqual(result.actions, expected["actions"])
        cache = lazy_density_greedy(MemoizedGain(prob.p, prob.cost, prob.dist),
                                   storage.query_sizes + storage.answer_sizes, storage.budget)
        self.assertEqual(sorted(cache), expected["greedy"])

    def test_all_five_online_algorithms_match_the_current_reference(self):
        from experiments.e1 import run_experiment
        expected = json.loads((FIXTURES / "e1_reference.json").read_text())
        actual = run_experiment(horizon=100, runs=1)["runs"]
        for before, after in zip(expected, actual):
            self.assertEqual(before["algorithm"], after["algorithm"])
            for key in before:
                if key in ("algorithm", "actions", "caches", "oracle_calls"):
                    self.assertEqual(before[key], after[key])
                else:
                    np.testing.assert_allclose(before[key], after[key], atol=1e-13)

    def test_offline_zero_cost_observation_is_not_treated_as_missing(self):
        prob = make_problem(n=3)
        cache = create_policy("LFU", mode="offline").fit(
            [0, 0, 1], [0., 0., np.nan], [True, True, False],
            prob.dist, np.array([1., np.nan, np.nan]), 1.)
        self.assertEqual(cache, {0})
