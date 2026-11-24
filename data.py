"""Bundled query data and fixed arrival distributions."""
from __future__ import annotations
import numpy as np
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def min_max(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min())


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


def arrival_probabilities(m: int, arrival: str = "uniform"):
    if arrival == "uniform":
        return np.full(m, 1.0 / m)
    if arrival == "zipf":
        weights = np.arange(1, m + 1, dtype=float) ** -1.1
        return weights / weights.sum()
    raise ValueError("arrival must be uniform or zipf")


@lru_cache(maxsize=None)
def load_problem(m: int, arrival: str = "uniform"):
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
        p=arrival_probabilities(m, arrival),
        cost=local_cost,
        dist=local_dist,
        tokens=tokens[idx],
    )
