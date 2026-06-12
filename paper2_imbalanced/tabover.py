"""
tabover.py - TabOversample: Conditional Diffusion Model for
             Imbalanced Tabular Regression.

Extends the base TabularDiffusion with:
    1. **Target-conditioned denoising** — the network receives both the
       noisy features x_t *and* a sinusoidal encoding of the target value y
       so it can learn p(x | y).
    2. **Relevance-weighted training** — the MSE loss for each sample is
       weighted by its relevance φ(y_i), making the model pay more attention
       to rare / extreme target values.

During oversampling:
    1. Compute relevance φ(y) for the training targets.
    2. Build a balanced target distribution that up-weights rare regions.
    3. Sample new target values from that distribution.
    4. Generate synthetic features conditioned on each sampled target.
    5. Merge synthetic data with the original training set.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from relevance import (
    relevance_function,
    get_balanced_target_distribution,
    sample_from_balanced_distribution,
    identify_rare_regions,
)
from diffusion import (
    linear_beta_schedule,
    cosine_beta_schedule,
    SinusoidalEmbedding,
)


# ---------------------------------------------------------------------------
# Conditional denoising MLP
# ---------------------------------------------------------------------------

class ConditionalDenoisingMLP(nn.Module):
    """
    MLP that predicts noise  ε_θ(x_t, t, y).

    Input = [ x_t ‖ time_emb ‖ y_emb ]
    where time_emb and y_emb are independent sinusoidal embeddings.
    """

    def __init__(self, input_dim: int, time_dim: int, y_dim: int,
                 hidden_dim: int = 256, n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(time_dim)
        self.y_embed = SinusoidalEmbedding(y_dim)

        layers = []
        in_features = input_dim + time_dim + y_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_features = hidden_dim
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                y_cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        y_emb = self.y_embed(y_cond)
        inp = torch.cat([x_t, t_emb, y_emb], dim=-1)
        return self.net(inp)


# ---------------------------------------------------------------------------
# TabOversample
# ---------------------------------------------------------------------------

class TabOversample(nn.Module):
    """
    Conditional diffusion model for imbalanced tabular regression.

    Parameters
    ----------
    input_dim : int
        Number of features per row (excluding the target).
    hidden_dim : int
        Width of hidden layers (default 256).
    n_layers : int
        Number of hidden layers (default 3).
    n_timesteps : int
        Diffusion timesteps T (default 1000).
    schedule : str
        'linear' or 'cosine' beta schedule.
    relevance_method : str
        Method for computing φ(y):  'boxplot' or 'density'.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 n_layers: int = 3, n_timesteps: int = 1000,
                 schedule: str = "linear",
                 relevance_method: str = "boxplot"):
        super().__init__()
        self.input_dim = input_dim
        self.n_timesteps = n_timesteps
        self.relevance_method = relevance_method

        # ---- Noise schedule ----
        if schedule == "linear":
            betas = linear_beta_schedule(n_timesteps)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(n_timesteps)
        else:
            raise ValueError(f"Unknown schedule '{schedule}'")

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar",
                             torch.sqrt(1.0 - alpha_bar))

        # ---- Conditional denoising network ----
        time_dim = min(128, hidden_dim)
        y_dim = min(64, hidden_dim)
        self.denoiser = ConditionalDenoisingMLP(
            input_dim=input_dim,
            time_dim=time_dim,
            y_dim=y_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

        # Will be set during training for y normalisation
        self._y_mean = 0.0
        self._y_std = 1.0

    # ----- Forward diffusion -----

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        x_t = sqrt_ab * x_0 + sqrt_omab * noise
        return x_t, noise

    # ----- Predict noise (conditioned on y) -----

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                y_cond: torch.Tensor) -> torch.Tensor:
        """Predict noise ε_θ(x_t, t, y_cond)."""
        return self.denoiser(x, t, y_cond)

    # ----- Training with relevance weighting -----

    def compute_loss(self, x_0: torch.Tensor, y: torch.Tensor,
                     relevance_weights: torch.Tensor) -> torch.Tensor:
        """
        Relevance-weighted DDPM loss:

            L = E_{t, ε} [ φ(y_i) · ||ε - ε_θ(x_t, t, y_i)||² ]
        """
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,),
                          device=x_0.device)

        x_t, noise = self.q_sample(x_0, t)

        # Normalise y for stable sinusoidal encoding
        y_norm = (y - self._y_mean) / (self._y_std + 1e-8)
        predicted_noise = self.forward(x_t, t, y_norm)

        loss_per_sample = (noise - predicted_noise).pow(2).mean(dim=-1)

        # Weight by relevance
        weighted_loss = (loss_per_sample * relevance_weights).mean()
        return weighted_loss

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    relevance_fn=None,
                    epochs: int = 100,
                    batch_size: int = 256,
                    lr: float = 1e-3,
                    relevance_method: str | None = None,
                    verbose: bool = True) -> list:
        """
        Train the conditional diffusion model with relevance weighting.

        Parameters
        ----------
        X_train : np.ndarray, shape (n, d)
        y_train : np.ndarray, shape (n,)
        relevance_fn : callable or None
            If None, uses relevance_function(y_train, method=...).
        epochs, batch_size, lr : training hyperparameters.
        relevance_method : str or None
            Override self.relevance_method if given.
        verbose : bool

        Returns
        -------
        losses : list[float]
        """
        self.train()
        method = relevance_method or self.relevance_method

        # Compute relevance weights
        if relevance_fn is not None:
            phi = relevance_fn(y_train)
        else:
            phi = relevance_function(y_train, method=method)

        # Normalise relevance weights: floor + scale so mean ≈ 1
        phi_w = phi + 0.1  # floor so common samples are not completely ignored
        phi_w = phi_w / phi_w.mean()

        # Store y statistics for normalisation
        self._y_mean = float(y_train.mean())
        self._y_std = float(y_train.std())

        # Tensors
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        w_t = torch.tensor(phi_w, dtype=torch.float32)

        dataset = TensorDataset(X_t, y_t, w_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=False)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []

        epoch_iter = tqdm(range(epochs), desc="TabOversample training",
                          disable=not verbose)
        for epoch in epoch_iter:
            epoch_loss = 0.0
            n_batches = 0
            for x_b, y_b, w_b in loader:
                optimizer.zero_grad()
                loss = self.compute_loss(x_b, y_b, w_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg = epoch_loss / max(n_batches, 1)
            losses.append(avg)
            epoch_iter.set_postfix(loss=f"{avg:.4f}")

        return losses

    # ----- Conditional generation -----

    @torch.no_grad()
    def generate_for_target(self, y_target: np.ndarray,
                            n_samples: int | None = None,
                            verbose: bool = False) -> np.ndarray:
        """
        Generate feature vectors conditioned on specific target values.

        Parameters
        ----------
        y_target : np.ndarray, shape (m,)
            Target values to condition on.
        n_samples : int or None
            If None, generate one row per entry in y_target.

        Returns
        -------
        X_synthetic : np.ndarray, shape (m, input_dim)
        """
        self.eval()
        if n_samples is not None and n_samples != len(y_target):
            # Resample y_target to match n_samples
            rng = np.random.default_rng()
            idx = rng.choice(len(y_target), size=n_samples, replace=True)
            y_target = y_target[idx]

        m = len(y_target)
        y_norm = torch.tensor(
            (y_target - self._y_mean) / (self._y_std + 1e-8),
            dtype=torch.float32,
        )

        x = torch.randn(m, self.input_dim)

        timesteps = list(range(self.n_timesteps - 1, -1, -1))
        step_iter = tqdm(timesteps, desc="Conditional sampling",
                         disable=not verbose)

        for t_val in step_iter:
            t = torch.full((m,), t_val, dtype=torch.long)
            predicted_noise = self.forward(x, t, y_norm)

            alpha = self.alphas[t_val]
            alpha_b = self.alpha_bar[t_val]
            beta = self.betas[t_val]

            coef1 = 1.0 / torch.sqrt(alpha)
            coef2 = beta / torch.sqrt(1.0 - alpha_b)
            mean = coef1 * (x - coef2 * predicted_noise)

            if t_val > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta)
                x = mean + sigma * noise
            else:
                x = mean

        return x.numpy()

    # ----- Main oversampling API -----

    def oversample(self, X_train: np.ndarray, y_train: np.ndarray,
                   relevance_fn=None,
                   n_synthetic: int | None = None,
                   epochs: int = 100,
                   batch_size: int = 256,
                   lr: float = 1e-3,
                   verbose: bool = True) -> tuple:
        """
        Full oversampling pipeline:

        1. Compute relevance φ(y) for the training set.
        2. Train the conditional diffusion model (relevance-weighted).
        3. Build a balanced target distribution that up-weights rare bins.
        4. Sample target values from the balanced distribution.
        5. Generate synthetic feature vectors conditioned on those targets.
        6. Return the augmented dataset (X_aug, y_aug).

        Parameters
        ----------
        X_train : np.ndarray, shape (n, d)
        y_train : np.ndarray, shape (n,)
        relevance_fn : callable or None
        n_synthetic : int or None
            Number of synthetic samples.  If None, defaults to the number
            of rare samples (φ > 0.5).
        epochs, batch_size, lr : training hyper-parameters.
        verbose : bool

        Returns
        -------
        X_aug : np.ndarray, shape (n + n_synthetic, d)
        y_aug : np.ndarray, shape (n + n_synthetic,)
        """
        # 1. Relevance
        method = self.relevance_method
        if relevance_fn is not None:
            phi = relevance_fn(y_train)
        else:
            phi = relevance_function(y_train, method=method)

        # Default n_synthetic = number of rare samples
        if n_synthetic is None:
            rare_mask = identify_rare_regions(y_train, phi, threshold=0.5)
            n_synthetic = int(rare_mask.sum())
            if n_synthetic == 0:
                n_synthetic = max(1, len(y_train) // 5)

        if verbose:
            print(f"[TabOversample] Generating {n_synthetic} synthetic samples "
                  f"({n_synthetic / len(y_train) * 100:.1f}% of train)...")

        # 2. Train
        self.train_model(X_train, y_train, relevance_fn=relevance_fn,
                         epochs=epochs, batch_size=batch_size, lr=lr,
                         verbose=verbose)

        # 3. Balanced target distribution
        dist = get_balanced_target_distribution(y_train, phi)

        # 4. Sample target values
        rng = np.random.default_rng(42)
        y_sampled = sample_from_balanced_distribution(dist, n_synthetic, rng=rng)

        # 5. Generate features conditioned on sampled targets
        X_synthetic = self.generate_for_target(y_sampled, verbose=verbose)

        # 6. Merge
        X_aug = np.concatenate([X_train, X_synthetic], axis=0)
        y_aug = np.concatenate([y_train, y_sampled], axis=0)

        if verbose:
            print(f"[TabOversample] Augmented: {X_train.shape[0]} -> "
                  f"{X_aug.shape[0]} samples")

        return X_aug, y_aug


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    n, d = 500, 8
    X = np.random.randn(n, d).astype(np.float32)
    y = np.concatenate([
        np.random.normal(5, 1, size=400),
        np.random.normal(15, 0.5, size=100),
    ]).astype(np.float32)

    model = TabOversample(input_dim=d, hidden_dim=128, n_layers=2,
                          n_timesteps=50)
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    X_aug, y_aug = model.oversample(
        X, y, epochs=5, batch_size=64, lr=1e-3, verbose=True
    )
    print(f"\nOriginal: X={X.shape}, y={y.shape}")
    print(f"Augmented: X={X_aug.shape}, y={y_aug.shape}")
