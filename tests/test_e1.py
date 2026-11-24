"""Fixed-budget experiment accounting and reproducibility."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.helpers import make_problem


class E1Tests(unittest.TestCase):
    def module(self):
        self.assertTrue((Path(__file__).resolve().parents[1] / "experiments/e1.py").is_file(), "e1 entry missing")
        return importlib.import_module("experiments.e1")

    def test_fixed_budget_shared_inputs_and_regret_accounting(self):
        e1 = self.module()
        prob = make_problem(n=5)
        prob.p = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
        with patch.object(e1, "load_problem", return_value=prob):
            result = e1.run_experiment(n_queries=5, runs=1, horizon=20, budget=1.8, alpha=0.5)
        self.assertEqual(result["config"]["experiment"], "e1")
        self.assertEqual(result["config"]["mode"], "online")
        rows = result["runs"]
        self.assertEqual({r["algorithm"] for r in rows}, {"OMBSC", "CLCB-SC", "Epsilon-Greedy", "LFU", "VSOCB-Semantic"})
        for row in rows:
            self.assertEqual(row["query_stream"], rows[0]["query_stream"])
            self.assertEqual(row["entry_sizes"], rows[0]["entry_sizes"])
            self.assertAlmostEqual(row["budget"], 1.8)
            self.assertTrue(np.all(np.asarray(row["memory_used"]) <= 1.8 + 1e-10))
            objective = np.asarray(row["cache_losses"]) + row["switching_costs"]
            baseline = 0.5 * row["empty_loss"] + 0.5 * row["optimal_loss"]
            np.testing.assert_allclose(row["cumulative_alpha_regret"],
                                       objective.cumsum() - np.arange(1, 21) * baseline)
            np.testing.assert_allclose(row["cumulative_exact_regret"],
                                       objective.cumsum() - np.arange(1, 21) * row["optimal_loss"])
            self.assertEqual(row["service_llm_calls"][0], 1)
        # The first observation of the shared first query is paired across algorithms.
        self.assertEqual(len({r["realized_serving_costs"][0] for r in rows}), 1)

    def test_reproducibility_outputs_and_algorithm_subset(self):
        e1 = self.module()
        prob = make_problem(n=4)
        kwargs = dict(n_queries=4, runs=1, horizon=8, algorithms=["LFU", "Epsilon-Greedy"], seed=11)
        with patch.object(e1, "load_problem", return_value=prob):
            first = e1.run_experiment(**kwargs)
            second = e1.run_experiment(**kwargs)
        for a, b in zip(first["runs"], second["runs"]):
            for key in ("cache_losses", "switching_costs", "realized_serving_costs", "caches"):
                self.assertEqual(a[key], b[key])
        with tempfile.TemporaryDirectory() as tmp:
            e1.save_results(first, Path(tmp))
            for name in ("e1.json", "summary.csv", "curves.csv", "e1.png"):
                self.assertGreater((Path(tmp) / name).stat().st_size, 100)
            loaded = json.loads((Path(tmp) / "e1.json").read_text())
            self.assertEqual(loaded["runs"][0]["query_stream"], first["runs"][0]["query_stream"])

    def test_invalid_experiment_configuration_is_rejected(self):
        e1 = self.module()
        for kwargs in (dict(horizon=1), dict(runs=0), dict(alpha=0), dict(budget=-1),
                       dict(algorithms=["LFU", "LFU"]), dict(algorithms=[]), dict(epsilon=2)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                e1.run_experiment(**kwargs)


if __name__ == "__main__":
    unittest.main()
