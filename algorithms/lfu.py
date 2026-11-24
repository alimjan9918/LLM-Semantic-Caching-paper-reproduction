"""Frequency-first memory-feasible policy with offline/online modes."""
from algorithms.base import CachePolicy, eligible_candidates


class LFU(CachePolicy):
    """Frequency-first memory-feasible cache, using only the observed prefix."""

    def select_cache(self, p_hat, cost_hat, dist, storage_sizes, budget, candidates):
        entries = eligible_candidates(storage_sizes, budget, candidates)
        entries.sort(key=lambda q: (-p_hat[q], q))
        selected, used = set(), 0.0
        for q in entries:
            if used + storage_sizes[q] <= budget + 1e-12:
                selected.add(q)
                used += storage_sizes[q]
        return selected
