"""Semantic cache loss and loss reduction, independent of algorithms."""
from __future__ import annotations
import numpy as np
from typing import Iterable


def _indices(cache: Iterable[int] | np.ndarray | None) -> np.ndarray:
    if cache is None:
        return np.empty(0, dtype=int)
    values = np.asarray(list(cache) if not isinstance(cache, np.ndarray) else cache)
    if values.size == 0:
        return np.empty(0, dtype=int)
    return np.asarray(sorted({int(q) for q in values}), dtype=int)


def cache_loss(
    cache: Iterable[int] | np.ndarray | None,
    p,
    cost,
    dist,
    gamma: float = 1.0,
) -> float:
    """Exact expected loss, including the empty-cache case."""
    idx = _indices(cache)
    if idx.size == 0:
        return float(np.dot(p, cost))
    nearest = dist[:, idx].min(axis=1)
    return float(np.sum(p * np.minimum(gamma * nearest, cost)))


def cache_gain(
    cache: Iterable[int] | np.ndarray | None,
    p,
    cost,
    dist,
    gamma: float = 1.0,
) -> float:
    """Loss reduction relative to an empty cache."""
    return float(
        np.dot(p, cost) - cache_loss(cache, p, cost, dist, gamma=gamma)
    )


class MemoizedGain:
    """Memoised gain oracle used by the lazy greedy algorithm."""

    def __init__(self, p, cost, dist, gamma: float = 1.0):
        self.p = np.asarray(p, dtype=float)
        self.cost = np.asarray(cost, dtype=float)
        self.dist = np.asarray(dist, dtype=float)
        self.gamma = float(gamma)
        self.calls = 0
        self.unique_calls = 0
        self._memo: dict[frozenset[int], float] = {}

    def __call__(self, cache: Iterable[int] | np.ndarray | None) -> float:
        self.calls += 1
        key = frozenset(int(q) for q in _indices(cache))
        value = self._memo.get(key)
        if value is None:
            value = cache_gain(
                key,
                self.p,
                self.cost,
                self.dist,
                gamma=self.gamma,
            )
            self._memo[key] = value
            self.unique_calls += 1
        return value


def nearest_distance(q: int, cache: Iterable[int], dist) -> float:
    idx = _indices(cache)
    return float("inf") if idx.size == 0 else float(dist[q, idx].min())
