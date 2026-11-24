"""Paper-specific behavior, not just a generic budgeted cache selector."""
import importlib
from itertools import combinations
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from tests.helpers import make_problem
from storage import OnlineStorageModel

ROOT = Path(__file__).resolve().parents[1]
NAME = "VSOCB-Semantic"


class VSOCBTests(unittest.TestCase):
    def module(self):
        self.assertTrue((ROOT / "algorithms/vsocb.py").is_file(), "VSOCB implementation missing")
        return importlib.import_module("algorithms.vsocb")

    def test_exact_knapsack_handles_fractional_memory_without_rounding(self):
        self.module()
        from algorithms.knapsack_oracle import exact_knapsack
        rng = np.random.default_rng(2)
        for trial in range(12):
            sizes = rng.uniform(0.1, 2.1, 8)
            values = rng.uniform(0, 1, 8)
            budget = 2.7
            selected = exact_knapsack(values, sizes, budget)
            optimum = max(sum(values[list(s)]) for k in range(9)
                          for s in combinations(range(8), k)
                          if sum(sizes[list(s)]) <= budget + 1e-12)
            self.assertLessEqual(sum(sizes[list(selected)]), budget + 1e-12)
            self.assertAlmostEqual(sum(values[list(selected)]), optimum)
        selected = exact_knapsack(np.array([1., 1.]), np.array([0.50000001, 0.5]), 1.0)
        self.assertEqual(len(selected), 1, "rounding fractional sizes would violate capacity")
        self.assertEqual(exact_knapsack(np.zeros(3), np.array([0.4, 0.6, 0.8]), 1.0), {0, 1})
        with self.assertRaises(ValueError):
            exact_knapsack(np.ones(41), np.ones(41), 20)

    def test_paper_cost_and_probability_lower_bounds(self):
        module = self.module()
        policy = module.VSOCBPolicy(mode="online")
        observations = np.array([100., 0., 400.])
        totals = np.array([80., 0., 200.])
        horizon, delta, t = 1000, 0.001, 900
        expected_c = np.maximum(0., np.array([0.8, 0., 0.5])
                                - np.sqrt(np.log(8 * horizon * 3 / delta) / (2 * np.maximum(observations, 1))))
        np.testing.assert_allclose(policy.estimate_costs(observations, totals, horizon, delta), expected_c)
        counts = np.array([700., 150., 50.])
        p = counts / t
        logterm = np.log(16 * horizon * 3 / delta)
        expected_p = np.maximum(0., p - np.sqrt(3 * p * (1 - p) * logterm / t) - 5 * logterm / t)
        np.testing.assert_allclose(policy.estimate_probabilities(counts, t, delta, horizon=horizon), expected_p)

    def test_oracle_uses_additive_values_independent_of_semantic_distance(self):
        module = self.module()
        policy = module.VSOCBPolicy()
        p, c = np.array([0.6, 0.3, 0.1]), np.array([0.3, 1.0, 0.9])
        sizes = np.array([1., 1., np.nan])
        for dist in (np.eye(3), np.zeros((3, 3))):
            self.assertEqual(policy.select_cache(p, c, dist, sizes, 1.0, {0, 1}), {1})

    def test_trigger_uses_multiplicative_cost_counts_or_time(self):
        module = self.module()
        prob = make_problem(n=3)
        storage = OnlineStorageModel(np.ones(3) / 2, np.ones(3) / 2, 1., 1/3, 0.)
        agent = module.VSOCBPolicy(growth=1.).make_agent(prob, storage, 100, 0.01)
        agent.last_oracle_round = 10
        agent.reference_observations[0] = 2
        agent.observations[0] = 3
        self.assertFalse(agent._oracle_due(19, 0))
        agent.observations[0] = 4
        self.assertTrue(agent._oracle_due(19, 0))
        agent.observations[0] = 3
        self.assertTrue(agent._oracle_due(20, 0))

    def test_recommended_answer_is_not_populated_until_own_arrival(self):
        module = self.module()
        prob = make_problem(n=3)
        storage = OnlineStorageModel(np.ones(3) / 2, np.ones(3) / 2, 1., 1/3, 0.)
        policy = module.VSOCBPolicy(growth=1.)
        agent = policy.make_agent(prob, storage, 100, 0.01)
        agent.cache = {0}
        agent.recommended_cache = {0}
        agent.candidates = {0, 1}
        agent.storage_sizes[:2] = 1.
        agent.observations[:2] = [1., 2.]
        agent.reference_observations[:2] = [1., 2.]
        agent.total_cost[:2] = [0.5, 1.]
        agent.arrivals[:2] = [1., 2.]
        agent.last_oracle_round = 2
        with patch.object(policy, "select_cache", return_value={1}):
            agent.step(4, 0, np.random.default_rng(1))
        self.assertEqual(agent.recommended_cache, {1})
        self.assertEqual(agent.cache, set(), "recorded size is not an available answer")
        self.assertEqual(agent.oracle_calls, 1)
        agent.step(5, 1, np.random.default_rng(1))
        self.assertEqual(agent.cache, {1})
        self.assertEqual(agent.oracle_calls, 1)

    def test_semantic_reuse_has_positive_cost_and_no_false_answer_admission(self):
        module = self.module()
        prob = make_problem(n=3)
        prob.dist[1, 0] = prob.dist[0, 1] = 0.2
        storage = OnlineStorageModel(np.ones(3) / 2, np.ones(3) / 2, 2., 2/3, 0.)
        agent = module.VSOCBPolicy().make_agent(prob, storage, 100, 0.01)
        agent.cache = {0}
        agent.recommended_cache = {0, 1}
        agent.candidates = {0, 1}
        agent.storage_sizes[:2] = 1.
        agent.observations[:2] = 2.
        agent.reference_observations[:2] = 2.
        agent.cost_estimate[1] = 0.8
        agent.total_cost[:2] = 1.6
        agent.last_oracle_round = 4
        result = agent.step(5, 1, np.random.default_rng(1))
        self.assertEqual(result["action"], "cache")
        self.assertAlmostEqual(result["serving_cost"], 0.2)
        self.assertEqual(agent.observations[1], 2.)
        self.assertEqual(agent.cache, {0})

    def test_filler_admissions_reserve_memory_between_oracle_calls(self):
        module = self.module()
        prob = make_problem(n=3)
        storage = OnlineStorageModel(np.ones(3) / 2, np.ones(3) / 2, 1.5, 0.5, 0.)
        agent = module.VSOCBPolicy().make_agent(prob, storage, 100, 0.01)
        with patch.object(agent, "_oracle_due", return_value=False):
            agent.step(1, 0, np.random.default_rng(1))
            self.assertEqual(agent.recommended_cache, {0})
            agent.step(2, 1, np.random.default_rng(1))
        self.assertEqual(agent.cache, {0})
        self.assertEqual(agent.recommended_cache, {0})
        self.assertTrue(storage.feasible(agent.cache))

    def test_e1_includes_vsocb_by_default_and_growth_is_separate_from_alpha(self):
        self.module()
        from algorithms.registry import DEFAULT_ALGORITHMS
        from experiments.e1 import run_experiment
        self.assertIn(NAME, DEFAULT_ALGORITHMS)
        with self.assertRaises(ValueError):
            run_experiment(vsocb_growth=0)


if __name__ == "__main__":
    unittest.main()
