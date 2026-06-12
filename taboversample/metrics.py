"""
metrics.py - Evaluation metrics for imbalanced regression.

Standard metrics:
    - RMSE              Root Mean Squared Error (overall)
    - MAE               Mean Absolute Error (overall)

Rare-region metrics:
    - RMSE_rare         RMSE on rare samples only  (φ > threshold)
    - MAE_rare          MAE  on rare samples only

Relevance-aware metric:
    - SERA              Squared Error-Relevance Area (Ribeiro 2011)
                        ∑ φ(y_i) · (y_i − ŷ_i)²   (higher relevance counts more)

All metric functions follow the signature:
    metric(y_true, y_pred, [relevance], [threshold])
"""

import numpy as np


# ---------------------------------------------------------------------------
# Standard metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
# Rare-region metrics
# ---------------------------------------------------------------------------

def rmse_rare(y_true: np.ndarray, y_pred: np.ndarray,
              relevance: np.ndarray, threshold: float = 0.5) -> float:
    """
    RMSE on rare samples only (where relevance > threshold).

    Returns np.nan if no samples exceed the threshold.
    """
    mask = relevance > threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae_rare(y_true: np.ndarray, y_pred: np.ndarray,
             relevance: np.ndarray, threshold: float = 0.5) -> float:
    """
    MAE on rare samples only (where relevance > threshold).

    Returns np.nan if no samples exceed the threshold.
    """
    mask = relevance > threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


# ---------------------------------------------------------------------------
# SERA — Squared Error-Relevance Area
# ---------------------------------------------------------------------------

def sera(y_true: np.ndarray, y_pred: np.ndarray,
         relevance: np.ndarray) -> float:
    """
    Squared Error-Relevance Area (Ribeiro 2011).

    SERA = (1/n) · ∑_i  φ(y_i) · (y_i − ŷ_i)²

    This is essentially a relevance-weighted MSE: rare samples contribute
    proportionally more to the error.

    Parameters
    ----------
    y_true : np.ndarray     True target values.
    y_pred : np.ndarray     Predicted target values.
    relevance : np.ndarray  Relevance scores φ(y) ∈ [0, 1].

    Returns
    -------
    float   The SERA score (lower is better).
    """
    weighted_se = relevance * (y_true - y_pred) ** 2
    return float(np.mean(weighted_se))


# ---------------------------------------------------------------------------
# Convenience: evaluate all metrics at once
# ---------------------------------------------------------------------------

def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray,
                 relevance: np.ndarray,
                 threshold: float = 0.5) -> dict:
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray     True targets.
    y_pred : np.ndarray     Predictions.
    relevance : np.ndarray  Relevance scores for y_true.
    threshold : float       Threshold for "rare" classification.

    Returns
    -------
    dict  Metric name → value.
    """
    n_rare = int((relevance > threshold).sum())
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "RMSE_rare": rmse_rare(y_true, y_pred, relevance, threshold),
        "MAE_rare": mae_rare(y_true, y_pred, relevance, threshold),
        "SERA": sera(y_true, y_pred, relevance),
        "n_test": len(y_true),
        "n_rare": n_rare,
    }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    y_true = rng.normal(10, 3, size=200).astype(np.float32)
    y_pred = y_true + rng.normal(0, 1, size=200).astype(np.float32)

    # Fake relevance: extreme values are rare
    from relevance import relevance_function
    phi = relevance_function(y_true, method="boxplot")

    results = evaluate_all(y_true, y_pred, phi)
    for k, v in results.items():
        print(f"  {k:12s}: {v:.4f}" if isinstance(v, float) else
              f"  {k:12s}: {v}")
