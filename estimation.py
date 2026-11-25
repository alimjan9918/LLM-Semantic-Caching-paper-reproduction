"""Mean-preserving cost noise and OMBSC confidence estimates."""
from __future__ import annotations
import numpy as np
SIGMA = 0.05


def sample_serving_cost(mean: float, rng, sigma: float = SIGMA) -> float:
    """Symmetric bounded noise: E[C]=mean, with sub-Gaussian proxy sigma.

    Unlike clipping a Gaussian, a fair random sign preserves the mean.
    The amplitude shrinks near the boundaries, so sigma bounds the noise
    rather than specifying its standard deviation for every query.
    """
    if not 0 <= mean <= 1 or not np.isfinite(sigma) or sigma < 0:
        raise ValueError("cost mean must be in [0, 1] and sigma non-negative")
    amplitude = min(sigma, mean, 1.0 - mean)
    return float(mean + amplitude * rng.choice([-1.0, 1.0]))


def ombsc_confidence_costs(
    observations: np.ndarray,
    total_cost: np.ndarray,
    horizon: int,
    n_queries: int,
    delta: float,
    *,
    sigma: float = SIGMA,
) -> np.ndarray:
    """Equation (4): optimistic cost in the online OMBSC algorithm."""
    observations = np.asarray(observations, dtype=float)
    total_cost = np.asarray(total_cost, dtype=float)
    seen = observations > 0
    c_hat = np.divide(
        total_cost,
        observations,
        out=np.zeros_like(total_cost, dtype=float),
        where=seen,
    )
    radius = sigma * np.sqrt(
        2.0
        * np.log(4.0 * n_queries * horizon**3 / delta)
        / np.maximum(observations, 1.0)
    )
    return np.where(seen, np.maximum(c_hat - radius, 0.0), 0.0)
