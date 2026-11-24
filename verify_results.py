"""Verify the current e1 outputs without any old snapshots or experiment code."""
import argparse
import json
from pathlib import Path

import numpy as np

from data import load_problem
from objective import cache_loss


def verify_payload(data):
    config = data["config"]
    prob = load_problem(config["n_queries"], config["arrival"])
    paired, environments, seen_runs = {}, {}, set()
    assert len(data["runs"]) == config["runs"] * len(config["algorithms"])
    for row in data["runs"]:
        seed = row["seed"]
        run_key = (row["algorithm"], seed)
        assert run_key not in seen_runs and row["algorithm"] in config["algorithms"]
        seen_runs.add(run_key)
        environment = (row["query_stream"], row["entry_sizes"])
        if seed in environments:
            assert environment == environments[seed]
        environments[seed] = environment
        np.testing.assert_allclose(row["budget"], data["runs"][0]["budget"])
        sizes = np.asarray(row["entry_sizes"])
        loss, switches = np.asarray(row["cache_losses"]), np.asarray(row["switching_costs"])
        assert len(loss) == config["horizon"]
        calls = np.zeros(prob.m, dtype=int)
        assert switches[0] == 0 and row["served_caches"][0] == []
        for t, query in enumerate(row["query_stream"]):
            cache, served = set(row["caches"][t]), set(row["served_caches"][t])
            np.testing.assert_allclose(row["memory_used"][t], sizes[list(cache)].sum())
            assert row["memory_used"][t] <= row["budget"] + 1e-9
            np.testing.assert_allclose(loss[t], cache_loss(served, prob.p, prob.cost, prob.dist))
            if t + 1 < config["horizon"]:
                assert row["served_caches"][t + 1] == row["caches"][t]
                np.testing.assert_allclose(switches[t + 1], sum(prob.cost[q] for q in cache - served))
            else:
                assert cache == served
            if row["actions"][t] == "fresh":
                key = (seed, query, calls[query])
                value = row["realized_serving_costs"][t]
                if key in paired:
                    assert value == paired[key]
                paired[key] = value
                calls[query] += 1
            else:
                assert served
                np.testing.assert_allclose(row["realized_serving_costs"][t], prob.dist[query, list(served)].min())
            assert cache <= set(np.flatnonzero(calls))
            if row["algorithm"] == "VSOCB-Semantic":
                assert cache <= served | {query}
                if row["actions"][t] != "fresh" and query not in served:
                    assert query not in cache
        rounds = np.arange(1, config["horizon"] + 1)
        objective = np.cumsum(loss + switches)
        baseline = (1 - row["alpha"]) * row["empty_loss"] + row["alpha"] * row["optimal_loss"]
        np.testing.assert_allclose(row["alpha_baseline"], baseline)
        np.testing.assert_allclose(row["cumulative_alpha_regret"], objective - rounds * baseline)
        np.testing.assert_allclose(row["cumulative_exact_regret"], objective - rounds * row["optimal_loss"])
        assert row["service_llm_call_count"] == int(calls.sum())
        assert 0 <= row["oracle_calls"] <= config["horizon"] - 1
    return len(data["runs"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="e1.json files or result directories")
    args = parser.parse_args()
    paths = args.paths or sorted((Path(__file__).resolve().parent / "results").glob("*/e1.json"))
    if not paths:
        parser.error("no e1 outputs found; run e1.py first")
    for path in paths:
        if path.is_dir():
            path = path / "e1.json"
        count = verify_payload(json.loads(path.read_text(encoding="utf-8")))
        print(f"Verified {path.parent.name}: {count} runs; budgets, paired noise, cache timing and both regrets.")


if __name__ == "__main__":
    main()
