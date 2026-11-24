"""Memory-budget reverse greedy used by the comparison policies."""
from objective import cache_loss


def reverse_greedy_budgeted(p, cost, dist, storage_sizes, budget, *, candidates,
                            gamma=1.0, explore=0.0, rng=None):
    """Budget adaptation: delete least loss per released memory until feasible.

    This heuristic is separate from the loss-reduction approximation guarantee
    of lazy_density_greedy. Epsilon exploration randomizes a deletion.
    """
    from algorithms.base import eligible_candidates
    if not 0 <= explore <= 1:
        raise ValueError("explore must lie in [0, 1]")
    cache = set(eligible_candidates(storage_sizes, budget, candidates))
    used = sum(storage_sizes[q] for q in cache)
    while cache and used > budget + 1e-12:
        entries = sorted(cache)
        if explore and rng.random() < explore:
            removed = int(rng.choice(entries))
        else:
            current = cache_loss(cache, p, cost, dist, gamma=gamma)
            removed = min(entries, key=lambda q: (
                round((cache_loss(cache - {q}, p, cost, dist, gamma=gamma) - current)
                      / storage_sizes[q], 12), q))
        cache.remove(removed)
        used -= storage_sizes[removed]
    return cache
