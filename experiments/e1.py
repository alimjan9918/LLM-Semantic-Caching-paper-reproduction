"""e1: online algorithms under one fixed memory budget.

python e1.py --quick
python e1.py --horizon 2000 --runs 10 --budget-fraction 0.25 --arrival zipf
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algorithms.registry import ALGORITHMS, DEFAULT_ALGORITHMS, create_policy
from data import load_problem
from estimation import SIGMA
from evaluation import exact_budgeted_losses
from objective import cache_loss
from online import run_online_agent
from storage import make_online_storage_model

DEFAULT_ALPHA = (1 - np.exp(-1)) / 2


def run_experiment(*, n_queries=20, horizon=200, runs=10, budget_fraction=0.25,
                   budget=None, algorithms=DEFAULT_ALGORITHMS, arrival="uniform",
                   alpha=DEFAULT_ALPHA, epsilon=0.15, seed=0, vsocb_growth=1.0):
    """Pair environment inputs across policies; use truth only for evaluation."""
    if horizon < 2 or runs < 1 or not 2 <= n_queries <= 20 or seed < 0:
        raise ValueError("require horizon >= 2, runs >= 1, 2 <= queries <= 20, seed >= 0")
    if not 0 < alpha <= 1 or not 0 <= epsilon <= 1:
        raise ValueError("require alpha in (0,1] and epsilon in [0,1]")
    if not np.isfinite(vsocb_growth) or vsocb_growth <= 0:
        raise ValueError("VSOCB growth must be finite and positive")
    if budget is not None and (not np.isfinite(budget) or budget <= 0):
        raise ValueError("budget must be finite and positive")
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget fraction must lie in (0,1]")
    if arrival not in {"uniform", "zipf"}:
        raise ValueError("arrival must be uniform or zipf")
    names = list(algorithms)
    if not names or len(set(names)) != len(names) or set(names) - set(ALGORITHMS):
        raise ValueError("algorithms must be a nonempty list of distinct known names")

    prob = load_problem(n_queries, arrival)
    delta = 1.0 / horizon
    empty_loss = cache_loss(set(), prob.p, prob.cost, prob.dist)
    rows = []
    for trial in range(seed, seed + runs):
        storage = make_online_storage_model(prob, budget_fraction, seed=trial)
        if budget is not None:
            storage = replace(storage, budget=float(budget),
                              budget_fraction=float(budget / storage.total_memory))
        sizes = storage.query_sizes + storage.answer_sizes
        optimum = float(exact_budgeted_losses(prob, sizes, [storage.budget])[0])
        stream = random.Random(trial).choices(range(prob.m), weights=prob.p, k=horizon)
        for name in names:
            policy = create_policy(name, mode="online", seed=trial, epsilon=epsilon,
                                   vsocb_growth=vsocb_growth)
            agent = policy.make_agent(prob, storage, horizon, delta)
            result = run_online_agent(prob, storage, horizon, delta, trial, agent=agent,
                                      stream=stream, paired_noise=True)
            regret = result.approximation_regret(empty_loss, optimum, alpha)
            exact_regret = result.approximation_regret(empty_loss, optimum, 1.0)
            row = {
                "algorithm": name, "mode": "online", "seed": trial,
                "budget": storage.budget, "budget_fraction": storage.budget_fraction,
                "alpha": float(alpha), "delta": delta, "empty_loss": empty_loss,
                "optimal_loss": optimum,
                "alpha_baseline": float((1 - alpha) * empty_loss + alpha * optimum),
                "final_alpha_regret": float(regret[-1]),
                "final_exact_regret": float(exact_regret[-1]),
                "average_loss": float(result.losses.mean()),
                "average_objective": float(result.cumulative_objective[-1] / horizon),
                "average_realized_serving_cost": float(result.serving_costs.mean()),
                "total_switching_cost": float(result.switching_costs.sum()),
                "switches": result.switches,
                "service_llm_call_count": int(result.llm_calls.sum()),
                "population_entry_count": int(result.population_entries.sum()),
                "total_selection_runtime_s": result.total_selection_runtime,
                "oracle_calls": result.oracle_calls,
                "revealed_candidates": result.revealed_candidates,
                "entry_sizes": sizes.tolist(), "query_stream": result.query_stream.tolist(),
                "actions": result.actions,
                "cache_losses": result.losses.tolist(),
                "switching_costs": result.switching_costs.tolist(),
                "realized_serving_costs": result.serving_costs.tolist(),
                "service_llm_calls": result.llm_calls.tolist(),
                "population_entries": result.population_entries.tolist(),
                "memory_used": result.memory_used.tolist(),
                "cache_sizes": result.cache_sizes.tolist(),
                "served_caches": result.served_caches, "caches": result.caches,
                "cumulative_alpha_regret": regret.tolist(),
                "cumulative_exact_regret": exact_regret.tolist(),
            }
            rows.append(row)
        print(f"[e1] seed {trial}: {len(names)} algorithms finished", flush=True)
    return {
        "config": {
            "experiment": "e1", "mode": "online", "n_queries": prob.m,
            "horizon": horizon, "runs": runs, "first_seed": seed,
            "algorithms": names, "budget": budget, "budget_fraction": budget_fraction if budget is None else None,
            "arrival": arrival, "arrival_probabilities": prob.p.tolist(),
            "alpha": float(alpha), "delta": delta, "epsilon": epsilon,
            "vsocb_growth": float(vsocb_growth),
            "vsocb_source": "https://aclanthology.org/2026.emnlp-main.1849.pdf",
            "vsocb_adaptation": "paper LCBs and additive knapsack; recommend-and-wait; common semantic serving/loss/switching",
            "cost_noise_proxy": SIGMA,
            "noise_pairing": "independent per-query streams, paired by query and service-call index",
            "query_source": "bundled reference_data; normalized sizes and synthetic answers",
            "benchmark": "exact full-Q budgeted optimum, evaluation only",
            "loss": "expected ideal cache loss, distinct from realized serving cost",
            "switching": "K[t+1] = sum c(q) for entries inserted after round t; no extra observations",
            "cache_timestamps": "served_caches[t] before serving; caches[t] after update",
            "policies": {
                "OMBSC": "LCB + lazy density greedy/best singleton, stage-triggered updates",
                "CLCB-SC": "LCB + memory-budget reverse greedy, per-round updates",
                "Epsilon-Greedy": "mean cost + epsilon-random reverse-greedy deletion, per-round updates",
                "LFU": "frequency-first feasible packing, empirical mean for serving, per-round updates",
                "CUCB-SC": "UCB + memory-budget reverse greedy, per-round updates",
                "VSOCB-Semantic": "paper cost/probability LCBs + exact additive knapsack, multiplicative triggers, recommend-and-wait",
            },
        },
        "runs": rows,
    }


def save_results(payload, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "e1.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    scalar_keys = [key for key, value in payload["runs"][0].items() if not isinstance(value, list)]
    with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in scalar_keys} for row in payload["runs"])
    curve_keys = ("cache_losses", "switching_costs", "realized_serving_costs", "service_llm_calls",
                  "population_entries", "memory_used", "cache_sizes", "cumulative_alpha_regret", "cumulative_exact_regret")
    with (out / "curves.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["algorithm", "seed", "round", *curve_keys])
        writer.writeheader()
        for row in payload["runs"]:
            for t in range(payload["config"]["horizon"]):
                writer.writerow(dict(algorithm=row["algorithm"], seed=row["seed"], round=t + 1,
                                     **{key: row[key][t] for key in curve_keys}))
    plot_results(payload, out / "e1.png")


def plot_results(payload, path):
    """Across-seed mean and one standard deviation under a common budget."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    t = np.arange(1, payload["config"]["horizon"] + 1)
    for name in payload["config"]["algorithms"]:
        rows = [row for row in payload["runs"] if row["algorithm"] == name]
        loss = np.array([row["cache_losses"] for row in rows])
        switches = np.array([row["switching_costs"] for row in rows])
        curves = [np.array([row["cumulative_alpha_regret"] for row in rows]),
                  np.cumsum(loss + switches, axis=1) / t,
                  np.cumsum(switches, axis=1), np.cumsum(loss, axis=1) / t]
        for ax, curve in zip(axes.flat, curves):
            mean, std = curve.mean(axis=0), curve.std(axis=0)
            ax.plot(t, mean, label=name)
            ax.fill_between(t, mean - std, mean + std, alpha=0.12)
    labels = ["Cumulative alpha-regret", "Average objective (loss + switching)",
              "Cumulative switching cost", "Average expected cache loss"]
    for ax, label in zip(axes.flat, labels):
        ax.set(xlabel="Online round", ylabel=label)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0, 0].axhline(0, color="gray", linewidth=0.6)
    budgets = [row["budget"] for row in payload["runs"]]
    fig.suptitle(f"e1: fixed budget B={budgets[0]:.3g}, {payload['config']['arrival']} arrivals")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="2 seeds, 100 rounds")
    budget_group = parser.add_mutually_exclusive_group()
    budget_group.add_argument("--budget", type=float, help="fixed budget in normalized entry-memory units")
    budget_group.add_argument("--budget-fraction", type=float, default=0.25)
    parser.add_argument("--arrival", choices=("uniform", "zipf"), default="uniform")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--vsocb-growth", type=float, default=1.0,
                        help="VSOCB multiplicative trigger growth; distinct from regret alpha")
    parser.add_argument("--algorithms", nargs="+", choices=tuple(ALGORITHMS), default=list(DEFAULT_ALGORITHMS))
    parser.add_argument("--out", type=Path, default=Path("results/e1"))
    args = parser.parse_args()
    try:
        payload = run_experiment(n_queries=args.queries, horizon=100 if args.quick else args.horizon,
                                 runs=2 if args.quick else args.runs, budget=args.budget,
                                 budget_fraction=args.budget_fraction, algorithms=args.algorithms,
                                 arrival=args.arrival, alpha=args.alpha, epsilon=args.epsilon, seed=args.seed,
                                 vsocb_growth=args.vsocb_growth)
    except ValueError as exc:
        parser.error(str(exc))
    save_results(payload, args.out)
    print("Algorithm          alpha-regret  exact-regret  switch-cost  service-calls")
    for name in args.algorithms:
        rows = [row for row in payload["runs"] if row["algorithm"] == name]
        means = [np.mean([row[key] for row in rows]) for key in
                 ("final_alpha_regret", "final_exact_regret", "total_switching_cost", "service_llm_call_count")]
        print(f"{name:18s}" + " ".join(f"{value:12.4f}" for value in means))
    print(f"Saved {args.out.resolve()}")


if __name__ == "__main__":
    main()
