"""
baselines.py - Baseline oversampling methods for imbalanced regression.

Every class exposes the same interface:

    .oversample(X, y, relevance_fn, n_synthetic=None) → (X_aug, y_aug)

Methods
-------
1. NoOversampling       — identity (returns original data)
2. RandomOversampler    — duplicate rare samples
3. SMOTEROversampler    — SMOTE for Regression (Torgo et al. 2013)
4. SMOGNOversampler     — SMOGN: SMOTE + Gaussian noise for regression
5. VanillaTabDDPM       — unconditional TabDDPM (no relevance weighting)
6. CTGANOversampler     — CTGAN-based oversampling (via sdv)
"""

import warnings
import numpy as np
from sklearn.neighbors import NearestNeighbors

from relevance import (
    relevance_function,
    identify_rare_regions,
)


# ---------------------------------------------------------------------------
# 1. No Oversampling
# ---------------------------------------------------------------------------

class NoOversampling:
    """Baseline: no oversampling at all."""

    name = "None"

    def oversample(self, X, y, relevance_fn=None, n_synthetic=None):
        return X.copy(), y.copy()


# ---------------------------------------------------------------------------
# 2. Random Oversampling
# ---------------------------------------------------------------------------

class RandomOversampler:
    """
    Duplicate random samples drawn preferentially from rare regions.

    Strategy: sample with replacement from rare samples (φ > threshold).
    """

    name = "RandomOS"

    def __init__(self, threshold: float = 0.5, seed: int = 42):
        self.threshold = threshold
        self.seed = seed

    def oversample(self, X, y, relevance_fn=None, n_synthetic=None):
        phi = relevance_fn(y) if relevance_fn else relevance_function(y)
        rare_mask = identify_rare_regions(y, phi, threshold=self.threshold)

        if n_synthetic is None:
            n_synthetic = int(rare_mask.sum())
        if n_synthetic == 0:
            return X.copy(), y.copy()

        rng = np.random.default_rng(self.seed)

        # Draw from rare samples with replacement
        rare_idx = np.where(rare_mask)[0]
        if len(rare_idx) == 0:
            # Fall back to all samples weighted by relevance
            probs = (phi + 0.01) / (phi + 0.01).sum()
            chosen = rng.choice(len(y), size=n_synthetic, replace=True, p=probs)
        else:
            chosen = rng.choice(rare_idx, size=n_synthetic, replace=True)

        X_aug = np.concatenate([X, X[chosen]], axis=0)
        y_aug = np.concatenate([y, y[chosen]], axis=0)
        return X_aug, y_aug


# ---------------------------------------------------------------------------
# 3. SMOTER  (SMOTE for Regression)
# ---------------------------------------------------------------------------

class SMOTEROversampler:
    """
    SMOTE for Regression (Torgo et al. 2013).

    For each rare sample, find k nearest neighbours among *all* samples,
    pick one at random, and interpolate both features and target.
    """

    name = "SMOTER"

    def __init__(self, k: int = 5, threshold: float = 0.5, seed: int = 42):
        self.k = k
        self.threshold = threshold
        self.seed = seed

    def oversample(self, X, y, relevance_fn=None, n_synthetic=None):
        phi = relevance_fn(y) if relevance_fn else relevance_function(y)
        rare_mask = identify_rare_regions(y, phi, threshold=self.threshold)
        rare_idx = np.where(rare_mask)[0]

        if n_synthetic is None:
            n_synthetic = len(rare_idx)
        if n_synthetic == 0 or len(rare_idx) == 0:
            return X.copy(), y.copy()

        rng = np.random.default_rng(self.seed)
        k = min(self.k, len(X) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)

        X_new, y_new = [], []
        for _ in range(n_synthetic):
            idx = rng.choice(rare_idx)
            dists, nbrs = nn.kneighbors(X[idx].reshape(1, -1))
            nbrs = nbrs[0, 1:]  # exclude self
            nb = rng.choice(nbrs)

            lam = rng.uniform(0, 1)
            x_syn = X[idx] + lam * (X[nb] - X[idx])
            y_syn = y[idx] + lam * (y[nb] - y[idx])
            X_new.append(x_syn)
            y_new.append(y_syn)

        X_aug = np.concatenate([X, np.array(X_new)], axis=0)
        y_aug = np.concatenate([y, np.array(y_new)], axis=0)
        return X_aug, y_aug


# ---------------------------------------------------------------------------
# 4. SMOGN  (SMOTE + Gaussian Noise for Regression)
# ---------------------------------------------------------------------------

class SMOGNOversampler:
    """
    SMOGN (Branco et al. 2017).

    Combines SMOTE interpolation with additive Gaussian noise.
    - If the chosen neighbour is "close" (distance < median dist), use
      SMOTE interpolation.
    - If the neighbour is "far", add Gaussian noise scaled by the feature
      standard deviation.
    """

    name = "SMOGN"

    def __init__(self, k: int = 5, threshold: float = 0.5, seed: int = 42):
        self.k = k
        self.threshold = threshold
        self.seed = seed

    def oversample(self, X, y, relevance_fn=None, n_synthetic=None):
        phi = relevance_fn(y) if relevance_fn else relevance_function(y)
        rare_mask = identify_rare_regions(y, phi, threshold=self.threshold)
        rare_idx = np.where(rare_mask)[0]

        if n_synthetic is None:
            n_synthetic = len(rare_idx)
        if n_synthetic == 0 or len(rare_idx) == 0:
            return X.copy(), y.copy()

        rng = np.random.default_rng(self.seed)
        k = min(self.k, len(X) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)

        # Median NN distance across all rare samples (for close/far decision)
        dists_all, _ = nn.kneighbors(X[rare_idx])
        median_dist = np.median(dists_all[:, 1:])

        feat_std = X.std(axis=0) + 1e-8

        X_new, y_new = [], []
        for _ in range(n_synthetic):
            idx = rng.choice(rare_idx)
            dists, nbrs = nn.kneighbors(X[idx].reshape(1, -1))
            nbrs = nbrs[0, 1:]
            dists_nb = dists[0, 1:]
            nb_sel = rng.choice(len(nbrs))
            nb = nbrs[nb_sel]
            d = dists_nb[nb_sel]

            lam = rng.uniform(0, 1)
            if d < median_dist:
                # SMOTE interpolation
                x_syn = X[idx] + lam * (X[nb] - X[idx])
                y_syn = y[idx] + lam * (y[nb] - y[idx])
            else:
                # Gaussian noise
                noise = rng.normal(0, feat_std * 0.1)
                x_syn = X[idx] + noise
                y_syn = y[idx] + rng.normal(0, abs(y[idx]) * 0.01 + 1e-4)

            X_new.append(x_syn)
            y_new.append(y_syn)

        X_aug = np.concatenate([X, np.array(X_new)], axis=0)
        y_aug = np.concatenate([y, np.array(y_new)], axis=0)
        return X_aug, y_aug


# ---------------------------------------------------------------------------
# 5. Vanilla TabDDPM  (unconditional)
# ---------------------------------------------------------------------------

class VanillaTabDDPM:
    """
    Standard unconditional TabDDPM without relevance weighting.

    Trains the base TabularDiffusion to model p(x, y) jointly
    (features *and* target are generated together), then filters
    generated samples to keep those in rare target regions.
    """

    name = "TabDDPM"

    def __init__(self, n_timesteps: int = 1000, hidden_dim: int = 256,
                 n_layers: int = 3, epochs: int = 100,
                 batch_size: int = 256, lr: float = 1e-3,
                 seed: int = 42, verbose: bool = True):
        self.n_timesteps = n_timesteps
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.verbose = verbose

    def oversample(self, X, y, relevance_fn=None, n_synthetic=None):
        from diffusion import TabularDiffusion
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        phi = relevance_fn(y) if relevance_fn else relevance_function(y)
        rare_mask = identify_rare_regions(y, phi, threshold=0.5)

        if n_synthetic is None:
            n_synthetic = int(rare_mask.sum())
        if n_synthetic == 0:
            n_synthetic = max(1, len(y) // 5)

        # Train on (X, y) jointly
        Xy = np.column_stack([X, y.reshape(-1, 1)])
        model = TabularDiffusion(
            input_dim=Xy.shape[1],
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            n_timesteps=self.n_timesteps,
        )
        model.train_model(Xy, epochs=self.epochs,
                          batch_size=self.batch_size, lr=self.lr,
                          verbose=self.verbose)

        # Over-generate and keep samples whose y falls in rare region
        # Generate extra to increase chance of getting rare samples
        n_generate = n_synthetic * 5
        Xy_syn = model.sample(n_generate, verbose=self.verbose)
        X_syn = Xy_syn[:, :-1]
        y_syn = Xy_syn[:, -1]

        # Keep samples in rare regions
        phi_syn = relevance_fn(y_syn) if relevance_fn else relevance_function(
            np.concatenate([y, y_syn])
        )[len(y):]
        rare_syn = phi_syn > 0.3  # looser threshold to get enough samples

        if rare_syn.sum() >= n_synthetic:
            sel = np.where(rare_syn)[0][:n_synthetic]
        else:
            # Not enough rare - take what we got + random fill
            sel = np.arange(min(n_synthetic, len(y_syn)))

        X_aug = np.concatenate([X, X_syn[sel].astype(X.dtype)], axis=0)
        y_aug = np.concatenate([y, y_syn[sel].astype(y.dtype)], axis=0)
        return X_aug, y_aug


# ---------------------------------------------------------------------------
# 6. CTGAN Oversampler
# ---------------------------------------------------------------------------

class CTGANOversampler:
    """
    CTGAN-based oversampling (Xu et al. 2019).

    Uses the sdv/ctgan library.  Falls back to SMOTER if sdv is not installed.
    """

    name = "CTGAN"

    def __init__(self, epochs: int = 100, batch_size: int = 256,
                 threshold: float = 0.5, seed: int = 42,
                 verbose: bool = True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
        self.seed = seed
        self.verbose = verbose

    def oversample(self, X, y, relevance_fn=None, n_synthetic=None):
        phi = relevance_fn(y) if relevance_fn else relevance_function(y)
        rare_mask = identify_rare_regions(y, phi, threshold=self.threshold)

        if n_synthetic is None:
            n_synthetic = int(rare_mask.sum())
        if n_synthetic == 0:
            n_synthetic = max(1, len(y) // 5)

        try:
            import pandas as pd
            from ctgan import CTGAN

            # Build a DataFrame with features + target
            cols = [f"f{i}" for i in range(X.shape[1])] + ["target"]
            df = pd.DataFrame(
                np.column_stack([X, y.reshape(-1, 1)]),
                columns=cols,
            )

            ctgan = CTGAN(
                epochs=self.epochs,
                batch_size=min(self.batch_size, len(df)),
                verbose=self.verbose,
            )
            ctgan.fit(df)

            # Over-generate and filter for rare-region samples
            n_gen = n_synthetic * 5
            syn_df = ctgan.sample(n_gen)
            X_syn = syn_df.iloc[:, :-1].values.astype(X.dtype)
            y_syn = syn_df["target"].values.astype(y.dtype)

            phi_syn = relevance_fn(y_syn) if relevance_fn else relevance_function(
                np.concatenate([y, y_syn])
            )[len(y):]
            rare_syn = phi_syn > 0.3

            if rare_syn.sum() >= n_synthetic:
                sel = np.where(rare_syn)[0][:n_synthetic]
            else:
                sel = np.arange(min(n_synthetic, len(y_syn)))

            X_aug = np.concatenate([X, X_syn[sel]], axis=0)
            y_aug = np.concatenate([y, y_syn[sel]], axis=0)
            return X_aug, y_aug

        except ImportError:
            warnings.warn(
                "ctgan/sdv not installed — falling back to SMOTER. "
                "Install via: pip install ctgan",
                stacklevel=2,
            )
            fallback = SMOTEROversampler(seed=self.seed)
            return fallback.oversample(X, y, relevance_fn, n_synthetic)


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------

def get_all_baselines(epochs: int = 100, verbose: bool = True) -> list:
    """Return instances of all baseline methods."""
    return [
        NoOversampling(),
        RandomOversampler(),
        SMOTEROversampler(),
        SMOGNOversampler(),
        VanillaTabDDPM(epochs=epochs, verbose=verbose),
        CTGANOversampler(epochs=epochs, verbose=verbose),
    ]


def get_baseline(name: str, **kwargs):
    """Get a single baseline by name."""
    registry = {
        "none": NoOversampling,
        "random": RandomOversampler,
        "smoter": SMOTEROversampler,
        "smogn": SMOGNOversampler,
        "tabddpm": VanillaTabDDPM,
        "ctgan": CTGANOversampler,
    }
    key = name.lower().replace("_", "").replace(" ", "")
    if key not in registry:
        raise ValueError(f"Unknown baseline '{name}'. "
                         f"Choose from: {list(registry.keys())}")
    return registry[key](**kwargs)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    n, d = 300, 5
    X = np.random.randn(n, d).astype(np.float32)
    y = np.concatenate([
        np.random.normal(5, 1, size=250),
        np.random.normal(15, 0.5, size=50),
    ]).astype(np.float32)

    for cls in [NoOversampling, RandomOversampler, SMOTEROversampler,
                SMOGNOversampler]:
        method = cls() if cls == NoOversampling else cls()
        Xa, ya = method.oversample(X, y)
        print(f"{method.name:12s}  {X.shape[0]} -> {Xa.shape[0]}  "
              f"(+{Xa.shape[0] - X.shape[0]})")
