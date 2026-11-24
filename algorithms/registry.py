"""Mode-aware comparison algorithms; selectors themselves live in separate files."""
from algorithms.clcb_sc import CLCBSC
from algorithms.cucb_sc import CUCBSC
from algorithms.epsilon_greedy import EpsilonGreedyPolicy
from algorithms.lfu import LFU
from algorithms.ombsc import OMBSCPolicy
from algorithms.vsocb import VSOCBPolicy

ALGORITHMS = {
    "OMBSC": OMBSCPolicy,
    "CLCB-SC": CLCBSC,
    "Epsilon-Greedy": EpsilonGreedyPolicy,
    "LFU": LFU,
    "CUCB-SC": CUCBSC,
    "VSOCB-Semantic": VSOCBPolicy,
}
DEFAULT_ALGORITHMS = ("OMBSC", "CLCB-SC", "Epsilon-Greedy", "LFU", "VSOCB-Semantic")


def create_policy(name, *, mode="online", seed=0, gamma=1.0, epsilon=0.15, vsocb_growth=1.0):
    if name not in ALGORITHMS:
        raise ValueError(f"unknown algorithm: {name}")
    kwargs = dict(mode=mode, seed=seed, gamma=gamma)
    if name == "Epsilon-Greedy":
        kwargs["epsilon"] = epsilon
    if name == "VSOCB-Semantic":
        kwargs["growth"] = vsocb_growth
    return ALGORITHMS[name](**kwargs)
