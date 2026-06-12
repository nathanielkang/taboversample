"""
diffusion.py - Simplified TabDDPM (Denoising Diffusion Probabilistic Model)
for tabular data.

Implements:
    - Forward (noising) process with linear or cosine beta schedule
    - Sinusoidal timestep embeddings
    - MLP-based denoising network  eps_theta(x_t, t)
    - Standard DDPM training loss  L = E[||eps - eps_theta(x_t, t)||^2]
    - Reverse-process sampling to generate new tabular rows

Designed to run on CPU (32 GB RAM).
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------

def linear_beta_schedule(n_timesteps: int,
                         beta_start: float = 1e-4,
                         beta_end: float = 0.02) -> torch.Tensor:
    """Linearly-spaced beta schedule (Ho et al., 2020)."""
    return torch.linspace(beta_start, beta_end, n_timesteps, dtype=torch.float32)


def cosine_beta_schedule(n_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule (Nichol & Dhariwal, 2021).

    Produces a smoother noise schedule that preserves signal longer.
    """
    steps = n_timesteps + 1
    x = torch.linspace(0, n_timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / n_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clip(betas, 1e-6, 0.999).float()


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """
    Maps scalar timestep t → vector of dimension *dim* via sinusoidal
    positional encoding (Vaswani et al., 2017).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : torch.Tensor, shape (B,) or (B,1)
            Integer timesteps.

        Returns
        -------
        emb : torch.Tensor, shape (B, dim)
        """
        t = t.float().view(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:  # handle odd dim
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


# ---------------------------------------------------------------------------
# Denoising MLP
# ---------------------------------------------------------------------------

class DenoisingMLP(nn.Module):
    """
    Simple MLP that predicts noise  eps_theta(x_t, t).

    Architecture
    ------------
    [x_t || time_emb] → Linear → ReLU → Dropout
                      → Linear → ReLU → Dropout
                      → Linear → ReLU → Dropout
                      → Linear → output (same dim as x_t)
    """

    def __init__(self, input_dim: int, time_dim: int,
                 hidden_dim: int = 256, n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(time_dim)

        layers = []
        in_features = input_dim + time_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_features = hidden_dim
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        inp = torch.cat([x_t, t_emb], dim=-1)
        return self.net(inp)


# ---------------------------------------------------------------------------
# TabularDiffusion (unconditional DDPM)
# ---------------------------------------------------------------------------

class TabularDiffusion(nn.Module):
    """
    Base tabular diffusion model — simplified TabDDPM.

    This is an *unconditional* model: it learns p(x) and can generate
    new rows, but it does **not** condition on the target y.

    Parameters
    ----------
    input_dim : int
        Number of features per sample.
    hidden_dim : int
        Width of hidden layers in the denoising MLP (default 256).
    n_layers : int
        Number of hidden layers (default 3).
    n_timesteps : int
        Diffusion timesteps T (default 1000).
    schedule : str
        'linear' or 'cosine' beta schedule.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 n_layers: int = 3, n_timesteps: int = 1000,
                 schedule: str = "linear"):
        super().__init__()
        self.input_dim = input_dim
        self.n_timesteps = n_timesteps

        # ---- Noise schedule ----
        if schedule == "linear":
            betas = linear_beta_schedule(n_timesteps)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(n_timesteps)
        else:
            raise ValueError(f"Unknown schedule '{schedule}'")

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        # Register as buffers (not parameters) so they move with .to(device)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar",
                             torch.sqrt(1.0 - alpha_bar))

        # ---- Denoising network ----
        time_dim = min(128, hidden_dim)
        self.denoiser = DenoisingMLP(
            input_dim=input_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

    # ----- Forward diffusion (noising) -----

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> tuple:
        """
        Sample x_t from q(x_t | x_0) = N(sqrt(ᾱ_t) x_0, (1-ᾱ_t) I).

        Returns (x_t, noise).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        x_t = sqrt_ab * x_0 + sqrt_omab * noise
        return x_t, noise

    # ----- Predict noise -----

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise ε_θ(x_t, t)."""
        return self.denoiser(x, t)

    # ----- Training -----

    def compute_loss(self, x_0: torch.Tensor,
                     sample_weights: torch.Tensor | None = None) -> torch.Tensor:
        """
        DDPM training loss:  L = E_t,ε [ w_i ||ε - ε_θ(x_t, t)||^2 ]
        """
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_0.device)

        x_t, noise = self.q_sample(x_0, t)
        predicted_noise = self.forward(x_t, t)

        loss_per_sample = (noise - predicted_noise).pow(2).mean(dim=-1)

        if sample_weights is not None:
            loss = (loss_per_sample * sample_weights).mean()
        else:
            loss = loss_per_sample.mean()

        return loss

    def train_model(self, X_train: np.ndarray,
                    epochs: int = 100,
                    batch_size: int = 256,
                    lr: float = 1e-3,
                    verbose: bool = True) -> list:
        """
        Train the diffusion model on tabular features.

        Parameters
        ----------
        X_train : np.ndarray, shape (n, d)
        epochs : int
        batch_size : int
        lr : float
        verbose : bool   Show tqdm progress bar.

        Returns
        -------
        losses : list[float]   Per-epoch average loss.
        """
        self.train()
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=False)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []

        epoch_iter = tqdm(range(epochs), desc="TabDDPM training",
                          disable=not verbose)
        for epoch in epoch_iter:
            epoch_loss = 0.0
            n_batches = 0
            for (x_batch,) in loader:
                optimizer.zero_grad()
                loss = self.compute_loss(x_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            epoch_iter.set_postfix(loss=f"{avg_loss:.4f}")

        return losses

    # ----- Sampling (reverse process) -----

    @torch.no_grad()
    def sample(self, n_samples: int, verbose: bool = False) -> np.ndarray:
        """
        Generate *n_samples* synthetic rows via the DDPM reverse process.

        Returns
        -------
        X_synthetic : np.ndarray, shape (n_samples, input_dim)
        """
        self.eval()
        x = torch.randn(n_samples, self.input_dim)

        timesteps = list(range(self.n_timesteps - 1, -1, -1))
        step_iter = tqdm(timesteps, desc="Sampling", disable=not verbose)

        for t_val in step_iter:
            t = torch.full((n_samples,), t_val, dtype=torch.long)
            predicted_noise = self.forward(x, t)

            alpha = self.alphas[t_val]
            alpha_b = self.alpha_bar[t_val]
            beta = self.betas[t_val]

            # Mean of p(x_{t-1} | x_t)
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


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    X_dummy = np.random.randn(500, 8).astype(np.float32)

    model = TabularDiffusion(input_dim=8, hidden_dim=128, n_layers=2,
                             n_timesteps=100)
    print(model)
    print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")

    losses = model.train_model(X_dummy, epochs=5, batch_size=64, lr=1e-3)
    print(f"Final loss: {losses[-1]:.4f}")

    samples = model.sample(10)
    print(f"Generated samples shape: {samples.shape}")
    print(samples[:3])
