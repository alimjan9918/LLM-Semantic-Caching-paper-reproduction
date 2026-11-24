"""Offline/online modes, prefix causality and switching chronology."""
import importlib
from pathlib import Path
import unittest
import numpy as np
from tests.helpers import make_problem
ROOT = Path(__file__).resolve().parents[1]


class PolicyTests(unittest.TestCase):
    def registry(self):
        self.assertTrue((ROOT / "algorithms/registry.py").is_file(), "mode-aware policy registry missing")
        return importlib.import_module("algorithms.registry")

    def test_comparison_algorithms_support_offline_and_online_modes(self):
        registry = self.registry()
        prob = make_problem(n=3)
        sizes = np.ones(3)
        for name in registry.ALGORITHMS:
            with self.subTest(name=name):
                policy = registry.create_policy(name, mode="offline", seed=3)
                selected = policy.fit(
                    np.array([0, 1, 1, 2]), np.array([0.3, np.nan, np.nan, np.nan]),
                    np.array([True, False, False, False]), prob.dist, sizes, 2.0,
                )
                self.assertTrue(selected <= {0}, "unobserved answers cannot enter the cache")
                with self.assertRaises(ValueError):
                    policy.make_agent(prob, None, 10, 0.1)
                online = registry.create_policy(name, mode="online", seed=3)
                with self.assertRaises(ValueError):
                    online.fit([], [], [], prob.dist, sizes, 2.0)
        with self.assertRaises(ValueError):
            registry.create_policy("LFU", mode="other")

    def test_online_algorithms_share_stream_reveal_sizes_and_charge_next_round(self):
        registry = self.registry()
        from storage import make_online_storage_model
        from online import run_online_agent
        prob = make_problem(n=5)
        storage = make_online_storage_model(prob, 0.35, seed=4)
        stream = np.array([0, 1, 2, 1, 0, 4, 3, 2, 4, 0, 1, 0])
        for name in registry.ALGORITHMS:
            with self.subTest(name=name):
                policy = registry.create_policy(name, mode="online", seed=3)
                agent = policy.make_agent(prob, storage, len(stream), 0.1)
                self.assertTrue(np.all(np.isnan(agent.storage_sizes)))
                result = run_online_agent(prob, storage, len(stream), 0.1, seed=3,
                                          agent=agent, stream=stream, paired_noise=True)
                np.testing.assert_array_equal(result.query_stream, stream)
                self.assertEqual(result.served_caches[0], [])
                self.assertEqual(result.switching_costs[0], 0)
                seen = set()
                for t, q in enumerate(stream):
                    if q not in seen:
                        self.assertEqual(result.actions[t], "fresh")
                    if result.actions[t] == "fresh":
                        seen.add(int(q))
                    self.assertTrue(set(result.caches[t]) <= seen)
                    self.assertTrue(storage.feasible(result.caches[t]))
                    if t + 1 < len(stream):
                        self.assertEqual(result.served_caches[t + 1], result.caches[t])
                        added = set(result.caches[t]) - set(result.served_caches[t])
                        self.assertAlmostEqual(result.switching_costs[t + 1], sum(prob.cost[q] for q in added))
                self.assertEqual(result.caches[-1], result.served_caches[-1])

    def test_online_lfu_depends_only_on_prefix_and_skips_entries_that_do_not_fit(self):
        registry = self.registry()
        from online import run_online_agent
        from storage import OnlineStorageModel
        prob = make_problem(n=3)
        storage = OnlineStorageModel(np.ones(3), np.ones(3), 2.0, 1 / 3, 0.0)
        runs = []
        for stream in ([0, 1, 0, 2, 2, 2], [0, 1, 0, 1, 1, 1]):
            policy = registry.create_policy("LFU", mode="online")
            agent = policy.make_agent(prob, storage, len(stream), 0.1)
            runs.append(run_online_agent(prob, storage, len(stream), 0.1, 1,
                                         agent=agent, stream=stream, paired_noise=True))
        self.assertEqual(runs[0].caches[:3], runs[1].caches[:3])
        self.assertEqual(runs[0].caches[2], [0])
        policy = registry.create_policy("LFU", mode="online")
        selected = policy.select_cache(np.array([0.6, 0.3, 0.1]), np.ones(3), prob.dist,
                                       np.array([4.0, 1.0, 1.0]), 2.0, {0, 1, 2})
        self.assertEqual(selected, {1, 2})

    def test_empty_feasible_candidate_set_is_supported(self):
        registry = self.registry()
        prob = make_problem(n=3)
        for name in registry.ALGORITHMS:
            policy = registry.create_policy(name, mode="online", epsilon=1.0)
            self.assertEqual(policy.select_cache(prob.p, prob.cost, prob.dist,
                             np.array([5.0, np.nan, 4.0]), 1.0, {0, 2}), set())

    def test_zero_mismatch_scale_with_empty_cache_still_invokes_llm(self):
        registry = self.registry()
        from storage import make_online_storage_model
        from online import run_online_agent
        prob = make_problem(n=3)
        storage = make_online_storage_model(prob, 0.001)
        for name in registry.ALGORITHMS:
            with self.subTest(name=name):
                policy = registry.create_policy(name, gamma=0, mode="online")
                agent = policy.make_agent(prob, storage, 3, 0.1)
                result = run_online_agent(prob, storage, 3, 0.1, 2,
                                          agent=agent, stream=[0, 0, 0])
                self.assertEqual(result.actions, ["fresh", "fresh", "fresh"])
                self.assertTrue(np.all(np.isfinite(result.serving_costs)))
