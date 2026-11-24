"""Small deterministic query instances used by the checks."""
from dataclasses import dataclass
import numpy as np


@dataclass
class TinyProblem:
    p: np.ndarray
    cost: np.ndarray
    dist: np.ndarray
    tokens: np.ndarray

    @property
    def m(self):
        return len(self.p)


def make_problem(n=10, seed=7):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n, 4))
    dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    dist /= dist.max()
    np.fill_diagonal(dist, 0.0)
    tokens = rng.integers(2, 20, size=n).astype(float)
    return TinyProblem(
        p=np.full(n, 1.0 / n),
        cost=np.clip((tokens - tokens.min()) / np.ptp(tokens), 0.02, 1.0),
        dist=dist,
        tokens=tokens,
    )
