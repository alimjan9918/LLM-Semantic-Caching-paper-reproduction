"""Run the heterogeneous-query-size extension on the author's query set.

Examples:

    python run_memory_budgeted.py --quick
    python run_memory_budgeted.py --runs 10 --memory-mode query
    python run_memory_budgeted.py --memory-mode response

The original ``run.py`` experiments remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from caching import Problem, min_max
from memory_aware import (
    cache_loss,
    log_dataset,
    make_memory_layout,
    make_online_storage_model,
    offline_memory_caches,
    run_ombsc,
    select_lazy_density,
)

DELTA = 0.8
M = 20
K = 5
T = 200
GAMMA = 1.0
OFFLINE_LENGTHS = (300, 450, 600)
BUDGET_FRACTIONS = (0.10, 0.20, 0.30, 0.40)
OFFLINE_ALGORITHMS = (
    "Lazy-Density-Known-Params",
    "Lazy-Density-CLCB",
    "Lazy-Density-CUCB",
    "Lazy-Density-Mean",
    "Memory-Blind-RG",
    "Best-Singleton",
)
ONLINE_ALGORITHMS = ("OMBSC",)

COLORS = {
    "Lazy-Density-Known-Params": "black",
    "Lazy-Density-CLCB": "tab:blue",
    "Lazy-Density-CUCB": "tab:orange",
    "Lazy-Density-Mean": "tab:green",
    "OMBSC": "tab:blue",
    "Memory-Blind-RG": "tab:red",
    "Best-Singleton": "tab:gray",
}
MARKERS = {
    "Lazy-Density-Known-Params": "x",
    "Lazy-Density-CLCB": "o",
    "Lazy-Density-CUCB": "v",
    "Lazy-Density-Mean": "s",
    "OMBSC": "o",
    "Memory-Blind-RG": "^",
    "Best-Singleton": ".",
}


@dataclass
class ReferenceProblem:
    """The 20-query instance used by the semcache-knapsack reference runs."""

    p: np.ndarray
    cost: np.ndarray
    dist: np.ndarray
    tokens: np.ndarray

    @property
    def m(self) -> int:
        return len(self.p)


@lru_cache(maxsize=None)
def problem(m: int, use_author_queries: bool = False):
    if use_author_queries:
        return Problem(m)

    root = Path(__file__).resolve().parent / "reference_data"
    queries = [
        line.strip()
        for line in (root / "queries20.txt").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    tokens = np.asarray(
        json.loads((root / "token_counts20.json").read_text(encoding="utf-8")),
        dtype=float,
    )
    dist = np.loadtxt(root / "distance20.csv", delimiter=",")
    if not (len(queries) == tokens.size == dist.shape[0] == dist.shape[1]):
        raise ValueError("reference query, token and distance data have inconsistent sizes")
    if m > len(queries):
        raise ValueError(f"reference data only contains {len(queries)} queries")

    idx = np.arange(m)
    local_dist = dist[np.ix_(idx, idx)].copy()
    local_dist = min_max(local_dist)
    np.fill_diagonal(local_dist, 0.0)
    local_cost = np.clip(min_max(tokens[idx]), 0.02, 1.0)
    return ReferenceProblem(
        p=np.full(m, 1.0 / m),
        cost=local_cost,
        dist=local_dist,
        tokens=tokens[idx],
    )


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std())


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_offline(
    runs: int,
    budget_fractions,
    memory_mode: str,
    use_author_queries: bool,
) -> list[dict]:
    prob = problem(M, use_author_queries)
    rows: list[dict] = []

    for seed in range(runs):
        datasets = {}
        for length in OFFLINE_LENGTHS:
            py_rng = random.Random(seed)
            rng = np.random.RandomState(seed)
            datasets[length] = log_dataset(
                prob,
                length,
                K,
                py_rng,
                rng,
            )

        for fraction in budget_fractions:
            layout = make_memory_layout(
                prob,
                fraction,
                mode=memory_mode,
                seed=seed,
            )
            oracle_cache = select_lazy_density(
                prob.p,
                prob.cost,
                prob.dist,
                layout,
            )

            for length in OFFLINE_LENGTHS:
                stream, observed_cost, observed_mask = datasets[length]
                caches = offline_memory_caches(
                    prob,
                    layout,
                    stream,
                    observed_cost,
                    observed_mask,
                    DELTA,
                )
                caches["Lazy-Density-Known-Params"] = oracle_cache

                for name in OFFLINE_ALGORITHMS:
                    cache = caches[name]
                    rows.append(
                        {
                            "seed": seed,
                            "budget_fraction": fraction,
                            "stream_length": length,
                            "algorithm": name,
                            "loss": cache_loss(
                                cache,
                                prob.p,
                                prob.cost,
                                prob.dist,
                            ),
                            "memory_used": layout.used(cache),
                            "budget": layout.budget,
                            "memory_fraction_used": layout.used(cache) / layout.budget,
                            "cache_size": len(cache),
                        }
                    )
        print(f"[offline] seed {seed + 1}/{runs} done", flush=True)
    return rows


def run_online(
    runs: int,
    budget_fractions,
    horizon: int,
    use_author_queries: bool,
    memory_noise: float,
) -> list[dict]:
    prob = problem(M, use_author_queries)
    rows: list[dict] = []

    for seed in range(runs):
        for fraction in budget_fractions:
            storage = make_online_storage_model(
                prob,
                fraction,
                memory_noise=memory_noise,
                seed=seed,
            )
            for name in ONLINE_ALGORITHMS:
                result = run_ombsc(
                    prob,
                    storage,
                    horizon,
                    DELTA,
                    seed,
                    gamma=GAMMA,
                )
                tail = slice(-max(1, min(25, len(result.losses) // 4)), None)
                objective = result.losses + result.switching_costs
                rows.append(
                    {
                        "seed": seed,
                        "budget_fraction": fraction,
                        "algorithm": name,
                        "average_loss": float(np.mean(result.losses)),
                        "final_window_loss": float(np.mean(result.losses[tail])),
                        "average_objective": float(np.mean(objective)),
                        "final_window_objective": float(np.mean(objective[tail])),
                        "total_switching_cost": float(result.switching_costs.sum()),
                        "switches": int(result.switches),
                        "runtime_s": float(result.mean_selection_runtime),
                        "total_llm_calls": int(result.llm_calls.sum()),
                        "revealed_candidates": int(result.revealed_candidates),
                        "final_cache_size": int(result.cache_sizes[-1]),
                        "final_memory_fraction_used": float(
                            result.memory_used[-1] / storage.budget
                        ),
                    }
                )
        print(f"[online] seed {seed + 1}/{runs} done", flush=True)
    return rows


def plot_results(
    offline_rows: list[dict],
    online_rows: list[dict],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    offline = defaultdict(list)
    for row in offline_rows:
        offline[(row["budget_fraction"], row["algorithm"])].append(row["loss"])
    for name in OFFLINE_ALGORITHMS:
        xs, means, stds = [], [], []
        for fraction in sorted({row["budget_fraction"] for row in offline_rows}):
            values = offline.get((fraction, name), [])
            if not values:
                continue
            mean, std = _mean_std(values)
            xs.append(fraction)
            means.append(mean)
            stds.append(std)
        axes[0].plot(
            xs,
            means,
            marker=MARKERS[name],
            color=COLORS[name],
            label=name,
        )
        axes[0].fill_between(
            xs,
            np.asarray(means) - np.asarray(stds),
            np.asarray(means) + np.asarray(stds),
            color=COLORS[name],
            alpha=0.10,
            linewidth=0,
        )

    online = defaultdict(list)
    for row in online_rows:
        online[(row["budget_fraction"], row["algorithm"])].append(
            row["final_window_loss"]
        )
    for name in ONLINE_ALGORITHMS:
        xs, means, stds = [], [], []
        for fraction in sorted({row["budget_fraction"] for row in online_rows}):
            values = online.get((fraction, name), [])
            if not values:
                continue
            mean, std = _mean_std(values)
            xs.append(fraction)
            means.append(mean)
            stds.append(std)
        axes[1].plot(
            xs,
            means,
            marker=MARKERS[name],
            color=COLORS[name],
            label=name,
        )
        axes[1].fill_between(
            xs,
            np.asarray(means) - np.asarray(stds),
            np.asarray(means) + np.asarray(stds),
            color=COLORS[name],
            alpha=0.10,
            linewidth=0,
        )

    axes[0].set(
        xlabel="Memory budget W / total entry memory",
        ylabel="Expected loss",
        title="(a) Offline partial feedback",
    )
    axes[1].set(
        xlabel="Memory budget W / total entry memory",
        ylabel="Final-window expected loss",
        title="(b) Online adaptation",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=7.2, ncol=2, frameon=False)
    axes[1].legend(fontsize=7.2, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def summarize(offline_rows: list[dict], online_rows: list[dict]) -> None:
    print("\nOffline mean loss:")
    for name in OFFLINE_ALGORITHMS:
        values = [row["loss"] for row in offline_rows if row["algorithm"] == name]
        print(f"  {name:22s} {np.mean(values):.6f}")

    print("\nOnline OMBSC final-window mean loss:")
    for name in ONLINE_ALGORITHMS:
        values = [
            row["final_window_loss"]
            for row in online_rows
            if row["algorithm"] == name
        ]
        switches = [
            row["switches"] for row in online_rows if row["algorithm"] == name
        ]
        switch_cost = [
            row["total_switching_cost"]
            for row in online_rows
            if row["algorithm"] == name
        ]
        runtime = [
            row["runtime_s"] for row in online_rows if row["algorithm"] == name
        ]
        print(
            f"  {name:22s} loss={np.mean(values):.6f} "
            f"switches={np.mean(switches):6.1f} "
            f"switch_cost={np.mean(switch_cost):.4f} "
            f"select_time={np.mean(runtime):.5f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--memory-mode",
        choices=("query", "response", "entry"),
        default="query",
    )
    parser.add_argument(
        "--author-queries",
        action="store_true",
        help="use the author's first M queries; downloads embedding/tokenizer models",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/budgeted"),
    )
    args = parser.parse_args()

    runs = 2 if args.quick else args.runs
    horizon = 100 if args.quick else T
    memory_noise = 0.45
    args.out.mkdir(parents=True, exist_ok=True)

    offline_rows = run_offline(
        runs,
        BUDGET_FRACTIONS,
        args.memory_mode,
        args.author_queries,
    )
    online_rows = run_online(
        runs,
        BUDGET_FRACTIONS,
        horizon,
        args.author_queries,
        memory_noise,
    )

    payload = {
        "config": {
            "runs": runs,
            "m": M,
            "k_for_logging": K,
            "offline_lengths": list(OFFLINE_LENGTHS),
            "online_horizon": horizon,
            "budget_fractions": list(BUDGET_FRACTIONS),
            "memory_mode": args.memory_mode,
            "online_storage": "S(q)=L(q)+A(q), A revealed on first LLM call",
            "online_memory_noise": memory_noise,
            "gamma": GAMMA,
            "query_source": "author" if args.author_queries else "reference_data",
            "delta": DELTA,
        },
        "offline": offline_rows,
        "online": online_rows,
    }
    (args.out / "results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    _write_csv(args.out / "offline.csv", offline_rows)
    _write_csv(args.out / "online.csv", online_rows)
    plot_results(offline_rows, online_rows, args.out / "memory_budgeted.png")
    summarize(offline_rows, online_rows)
    print(f"\nSaved results to {args.out.resolve()}")


if __name__ == "__main__":
    main()
