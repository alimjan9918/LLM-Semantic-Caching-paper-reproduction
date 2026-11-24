"""CLCBSC: mode-aware memory-budget comparison policy."""
from algorithms.base import CachePolicy
from algorithms.reverse_greedy import reverse_greedy_budgeted


class CLCBSC(CachePolicy):
    """Cost LCB + budget-adapted reverse greedy, offline or per-round online."""
    estimate_kind = "lcb"

    def select_cache(self, p_hat, cost_hat, dist, storage_sizes, budget, candidates):
        return reverse_greedy_budgeted(p_hat, cost_hat, dist, storage_sizes, budget,
                                      candidates=candidates, gamma=self.gamma)
