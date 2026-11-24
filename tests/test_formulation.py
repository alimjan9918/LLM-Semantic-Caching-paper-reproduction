"""Regressions for the agreed research formulation; uses unittest only."""
import itertools
import unittest
from unittest.mock import patch

import numpy as np

from algorithms.ombsc import OMBSC
from online import run_online_agent
from storage import make_online_storage_model
from evaluation import exact_budgeted_losses
from objective import cache_loss
from tests.helpers import make_problem


class SignedRng:
    def __init__(self, sign):
        self.sign = sign

    def choice(self, values):
        return values[0 if self.sign < 0 else -1]


class FormulationTests(unittest.TestCase):
    def test_online_cost_observations_preserve_the_declared_mean(self):
        for mean in (0.02, 0.98, 1.0):
            observed = []
            for sign in (-1, 1):
                prob = make_problem(n=2)
                prob.cost[0] = mean
                storage = make_online_storage_model(prob)
                agent = OMBSC(prob, storage, horizon=1, delta=0.5)
                agent.step(1, 0, SignedRng(sign))
                observed.append(agent.total_cost[0])
            self.assertAlmostEqual(np.mean(observed), mean, places=14)
            self.assertTrue(all(0 <= x <= 1 for x in observed))

    def test_online_queries_follow_problem_p(self):
        prob = make_problem(n=2)
        prob.p = np.array([0.99, 0.01])
        storage = make_online_storage_model(prob)
        seen = []
        original = OMBSC.step

        def record(agent, t, q, rng):
            seen.append(q)
            return original(agent, t, q, rng)

        with patch.object(OMBSC, "step", record):
            run_online_agent(prob, storage, 1000, 0.5, seed=3)
        self.assertGreater(seen.count(0), 900)



    def test_modeled_cache_insertions_are_separate_from_service_calls(self):
        prob = make_problem(n=2)
        storage = make_online_storage_model(prob, budget_fraction=1.0)
        agent = OMBSC(prob, storage, horizon=3, delta=0.5)
        with patch.object(agent, "_switch_cache", return_value=({0}, 0.0)):
            result = agent.step(1, 0, SignedRng(1))
        self.assertEqual(result["llm_calls"], 1)
        self.assertEqual(result["population_entries"], 1)
        self.assertAlmostEqual(result["switching_cost"], prob.cost[0])
        self.assertEqual(agent.observations.sum(), 1)

    def test_regret_includes_switches_uses_affine_baseline_and_keeps_negative_values(self):
        prob = make_problem(n=2)
        result = run_online_agent(
            prob, make_online_storage_model(prob), 3, 0.5, seed=2
        )
        method = getattr(result, "approximation_regret", None)
        self.assertTrue(callable(method), "missing alpha-regret evaluation")
        result.losses = np.array([0.1, 0.2, 0.1])
        result.switching_costs = np.array([0.0, 0.3, 0.0])
        np.testing.assert_allclose(method(1.0, 0.2, 0.5), [-0.5, -0.6, -1.1])
        np.testing.assert_allclose(method(1.0, 0.2, 1.0), [-0.1, 0.2, 0.1])
        with self.assertRaises(ValueError):
            method(1.0, 0.2, 0.0)

    def test_exact_benchmark_matches_exhaustive_small_instance(self):
        evaluator = exact_budgeted_losses
        self.assertTrue(callable(evaluator), "missing exact budgeted benchmark")
        prob = make_problem(n=6)
        sizes = np.arange(1, 7, dtype=float)
        budgets = [0.5, 3.0, 9.0, 21.0]
        expected = []
        for budget in budgets:
            feasible = (
                set(subset)
                for k in range(prob.m + 1)
                for subset in itertools.combinations(range(prob.m), k)
                if sizes[list(subset)].sum() <= budget
            )
            expected.append(min(
                cache_loss(x, prob.p, prob.cost, prob.dist, gamma=0.7)
                for x in feasible
            ))
        np.testing.assert_allclose(
            evaluator(prob, sizes, budgets, gamma=0.7), expected, atol=1e-12
        )



if __name__ == "__main__":
    unittest.main()
