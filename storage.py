"""Query-plus-answer storage and revealed-size memory model."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Iterable
from data import ReferenceProblem
from objective import _indices


@dataclass(frozen=True)
class RevealedStorage:
    """Policy view: the budget and sizes already observed through service."""

    budget: float
    recorded_sizes: np.ndarray

    def used(self, cache):
        return float(sum(self.recorded_sizes[q] for q in _indices(cache)))

    def feasible(self, cache, tol=1e-9):
        used = self.used(cache)
        return bool(np.isfinite(used) and used <= self.budget + tol)


@dataclass(frozen=True)
class OnlineStorageModel:
    """Hidden query/answer sizes used to reveal ``S(q)=L(q)+A(q)`` online."""

    query_sizes: np.ndarray
    answer_sizes: np.ndarray
    budget: float
    budget_fraction: float
    memory_noise: float

    def __post_init__(self):
        query_sizes = np.asarray(self.query_sizes, dtype=float)
        answer_sizes = np.asarray(self.answer_sizes, dtype=float)
        if query_sizes.ndim != 1 or answer_sizes.shape != query_sizes.shape:
            raise ValueError("query_sizes and answer_sizes must be equally long vectors")
        if np.any(query_sizes <= 0) or np.any(answer_sizes <= 0):
            raise ValueError("query and answer sizes must be positive")
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        object.__setattr__(self, "query_sizes", query_sizes)
        object.__setattr__(self, "answer_sizes", answer_sizes)

    @property
    def total_memory(self) -> float:
        return float((self.query_sizes + self.answer_sizes).sum())

    def entry_size(self, q: int) -> float:
        return float(self.query_sizes[q] + self.answer_sizes[q])

    def used(self, cache: Iterable[int] | np.ndarray | None) -> float:
        return float(sum(self.entry_size(q) for q in _indices(cache)))

    def feasible(
        self,
        cache: Iterable[int] | np.ndarray | None,
        tol: float = 1e-9,
    ) -> bool:
        return self.used(cache) <= self.budget + tol


def make_online_storage_model(
    problem: ReferenceProblem,
    budget_fraction: float = 0.25,
    *,
    memory_noise: float = 0.45,
    seed: int = 0,
) -> OnlineStorageModel:
    """Create the paper's hidden storage model.

    ``L(q)`` is the GPT-2 query length. ``A(q)`` is a latent response-size proxy;
    the online algorithm does not observe it until it invokes the LLM for q.
    """
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must lie in (0, 1]")
    if memory_noise < 0:
        raise ValueError("memory_noise must be non-negative")

    tokens = np.asarray(problem.tokens, dtype=float)
    rng = np.random.default_rng(seed)
    answer_raw = (tokens + 1.0) ** 1.15 * rng.lognormal(
        mean=0.0,
        sigma=memory_noise,
        size=tokens.size,
    )
    raw_total = tokens + answer_raw
    scale = 1.0 / max(float(raw_total.mean()), 1e-12)
    query_sizes = np.maximum(tokens * scale, 1e-12)
    answer_sizes = np.maximum(answer_raw * scale, 1e-12)
    budget = float(budget_fraction * (query_sizes + answer_sizes).sum())
    return OnlineStorageModel(
        query_sizes=query_sizes,
        answer_sizes=answer_sizes,
        budget=budget,
        budget_fraction=float(budget_fraction),
        memory_noise=float(memory_noise),
    )
