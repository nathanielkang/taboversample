"""
relevance.py - Relevance function φ(y) for imbalanced regression.

Implements the relevance framework from:
  - Ribeiro (2011): Utility-based Regression
  - Torgo et al. (2013): SMOTE for Regression

The relevance function φ: Y → [0, 1] assigns higher relevance to
rarer / more extreme target values.  Two methods are provided:
  1. boxplot  - based on IQR thresholds (default)
  2. density  - based on inverse kernel density estimation
"""

import numpy as np
from scipy.stats import gaussian_kde


# ---------------------------------------------------------------------------
# Core relevance functions
# ---------------------------------------------------------------------------

def relevance_function(y: np.ndarray, method: str = "boxplot") -> np.ndarray:
    """
    Compute relevance φ(y) ∈ [0, 1] for each target value.

    Higher relevance  →  rarer / more extreme value.
    Lower  relevance  →  common / near-median value.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Target values.
    method : str
        'boxplot'  – Torgo-style IQR-based relevance (default).
        'density'  – Inverse kernel density estimation.

    Returns
    -------
    phi : np.ndarray, shape (n,)
        Relevance scores in [0, 1].
    """
    y = np.asarray(y, dtype=np.float64)

    if method == "boxplot":
        return _relevance_boxplot(y)
    elif method == "density":
        return _relevance_density(y)
    else:
        raise ValueError(f"Unknown relevance method '{method}'. "
                         f"Choose 'boxplot' or 'density'.")


def _relevance_boxplot(y: np.ndarray) -> np.ndarray:
    """
    Box-plot-based relevance (Ribeiro 2011, Torgo et al. 2013).

    The idea:
        - Compute Q1, Q3, IQR of y.
        - Values beyond the whiskers  (< Q1 - 1.5*IQR  or  > Q3 + 1.5*IQR)
          receive relevance = 1.
        - Values at the median receive relevance = 0.
        - In between: piecewise-linear interpolation so that relevance
          increases as values move away from the median toward the whiskers.

    This produces a U-shaped relevance curve that peaks at both tails.
    """
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1
    median = np.median(y)

    low_fence = q1 - 1.5 * iqr
    high_fence = q3 + 1.5 * iqr

    phi = np.zeros_like(y, dtype=np.float64)

    # Below median: interpolate from median (φ=0) to low_fence (φ=1)
    below = y < median
    if median != low_fence:
        phi[below] = np.clip((median - y[below]) / (median - low_fence), 0.0, 1.0)

    # Above median: interpolate from median (φ=0) to high_fence (φ=1)
    above = y >= median
    if high_fence != median:
        phi[above] = np.clip((y[above] - median) / (high_fence - median), 0.0, 1.0)

    # Anything beyond the fences is clamped to 1
    phi = np.clip(phi, 0.0, 1.0)

    return phi


def _relevance_density(y: np.ndarray) -> np.ndarray:
    """
    Density-based relevance: φ(y) = 1 - normalized_density(y).

    Uses Gaussian KDE to estimate p(y), then defines relevance as the
    complement of the normalised density.  Rare values (low density)
    get high relevance.
    """
    kde = gaussian_kde(y, bw_method="scott")
    density = kde(y)

    # Normalise to [0, 1]
    d_min, d_max = density.min(), density.max()
    if d_max - d_min < 1e-12:
        return np.ones_like(y, dtype=np.float64)

    norm_density = (density - d_min) / (d_max - d_min)
    phi = 1.0 - norm_density
    return phi.astype(np.float64)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_sampling_weights(y: np.ndarray, relevance: np.ndarray) -> np.ndarray:
    """
    Convert relevance scores to sampling probabilities.

    Samples with higher relevance are more likely to be picked.
    A small floor (0.01) ensures that even common samples can be chosen.

    Parameters
    ----------
    y : np.ndarray          Target values (unused, kept for API symmetry).
    relevance : np.ndarray  Relevance scores φ(y) ∈ [0, 1].

    Returns
    -------
    weights : np.ndarray    Sampling probabilities summing to 1.
    """
    w = relevance + 0.01  # floor to avoid zero-prob samples
    return w / w.sum()


def get_balanced_target_distribution(
    y: np.ndarray,
    relevance: np.ndarray,
    n_bins: int = 50,
) -> np.ndarray:
    """
    Create a rebalanced target distribution for synthetic-sample generation.

    Strategy:
        1. Bin y into *n_bins* equal-width bins.
        2. Compute the mean relevance per bin.
        3. Set the desired count per bin proportional to its mean relevance
           (so that rarer regions receive more synthetic samples).
        4. Sample target values from a distribution that favours rare bins.

    Parameters
    ----------
    y : np.ndarray          Training target values.
    relevance : np.ndarray  Corresponding relevance scores.
    n_bins : int            Number of histogram bins.

    Returns
    -------
    target_distribution : np.ndarray, shape (n_bins, 3)
        Each row: [bin_centre, bin_width, sampling_weight].
    """
    y_min, y_max = y.min(), y.max()
    bin_edges = np.linspace(y_min - 1e-8, y_max + 1e-8, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    bin_weights = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        mask = (y >= bin_edges[i]) & (y < bin_edges[i + 1])
        if mask.any():
            bin_weights[i] = relevance[mask].mean()

    # Normalise to a probability distribution
    total = bin_weights.sum()
    if total < 1e-12:
        bin_weights = np.ones(n_bins) / n_bins
    else:
        bin_weights /= total

    dist = np.column_stack([bin_centres, np.full(n_bins, bin_width), bin_weights])
    return dist


def sample_from_balanced_distribution(
    target_distribution: np.ndarray,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Draw *n_samples* target values from the balanced distribution.

    Each sample is drawn by:
        1. Picking a bin according to the bin weights.
        2. Sampling uniformly within that bin.

    Parameters
    ----------
    target_distribution : np.ndarray  Output of get_balanced_target_distribution.
    n_samples : int                   How many target values to draw.
    rng : np.random.Generator         Optional RNG for reproducibility.

    Returns
    -------
    y_sampled : np.ndarray, shape (n_samples,)
    """
    if rng is None:
        rng = np.random.default_rng()

    centres = target_distribution[:, 0]
    widths = target_distribution[:, 1]
    weights = target_distribution[:, 2]

    # Choose bins
    bin_idx = rng.choice(len(centres), size=n_samples, p=weights)

    # Uniform within chosen bins
    offsets = rng.uniform(-0.5, 0.5, size=n_samples)
    y_sampled = centres[bin_idx] + offsets * widths[bin_idx]

    return y_sampled.astype(np.float32)


def identify_rare_regions(
    y: np.ndarray,
    relevance: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Boolean mask indicating rare / extreme samples.

    Parameters
    ----------
    y : np.ndarray          Target values (unused, kept for API symmetry).
    relevance : np.ndarray  Relevance scores φ(y) ∈ [0, 1].
    threshold : float       Samples with φ(y) > threshold are considered rare.

    Returns
    -------
    mask : np.ndarray[bool], shape (n,)
    """
    return relevance > threshold


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # Synthetic skewed target
    y = np.concatenate([rng.normal(5, 1, size=900),
                        rng.normal(15, 0.5, size=100)])
    rng.shuffle(y)

    phi_bp = relevance_function(y, method="boxplot")
    phi_dn = relevance_function(y, method="density")

    print(f"Boxplot relevance  -  min={phi_bp.min():.3f}  "
          f"max={phi_bp.max():.3f}  mean={phi_bp.mean():.3f}")
    print(f"Density relevance  -  min={phi_dn.min():.3f}  "
          f"max={phi_dn.max():.3f}  mean={phi_dn.mean():.3f}")

    rare_mask = identify_rare_regions(y, phi_bp, threshold=0.5)
    print(f"Rare samples (boxplot, thr=0.5): {rare_mask.sum()} / {len(y)}")

    weights = get_sampling_weights(y, phi_bp)
    print(f"Sampling weights sum={weights.sum():.4f}  "
          f"max={weights.max():.6f}  min={weights.min():.6f}")

    dist = get_balanced_target_distribution(y, phi_bp)
    y_new = sample_from_balanced_distribution(dist, 200, rng=rng)
    print(f"Balanced distribution sampled: n={len(y_new)}, "
          f"range=[{y_new.min():.2f}, {y_new.max():.2f}]")
