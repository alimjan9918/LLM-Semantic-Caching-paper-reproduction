"""EpsilonGreedyPolicy: mode-aware memory-budget comparison policy."""
from algorithms.base import CachePolicy
from algorithms.reverse_greedy import reverse_greedy_budgeted


class EpsilonGreedyPolicy(CachePolicy):
    """Mean-cost reverse greedy with epsilon-random deletions, in either mode."""

    def __init__(self, *, epsilon=0.15, **kwargs):
        super().__init__(**kwargs)
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must lie in [0, 1]")
        self.epsilon = float(epsilon)

    def select_cache(self, p_hat, cost_hat, dist, storage_sizes, budget, candidates):
        return reverse_greedy_budgeted(p_hat, cost_hat, dist, storage_sizes, budget,
                                      candidates=candidates, gamma=self.gamma,
                                      explore=self.epsilon, rng=self.rng)
