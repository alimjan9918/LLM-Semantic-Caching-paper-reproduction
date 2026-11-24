"""Cost-UCB memory-budget comparison policy."""
from algorithms.clcb_sc import CLCBSC


class CUCBSC(CLCBSC):
    """Cost UCB with the same budgeted selector and explicit mode interface."""
    estimate_kind = "ucb"
