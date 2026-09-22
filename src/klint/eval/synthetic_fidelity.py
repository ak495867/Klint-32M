"""Synthetic market generation fidelity: LSTM discriminator classifier, MMD, and Wasserstein distance."""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class LSTMDiscriminator(nn.Module):
    """Post-hoc LSTM classifier trained to distinguish Real vs Synthetic market trajectories."""
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1, :]).squeeze(-1)
        return logits


@dataclass
class SyntheticFidelityResult:
    """Quantitative fidelity scorecard for generative synthetic market trajectories."""
    discriminator_accuracy: float      # Accuracy of discriminator (50% = optimal indistinguishability)
    discriminative_score: float        # |Accuracy - 0.5| (0.0 = perfect realism, 0.5 = easily distinguished)
    mmd_score: float                   # Maximum Mean Discrepancy (lower is closer)
    wasserstein_distance: float        # 2-Wasserstein distance on returns
    synthetic_precision: float         # Manifold precision (% synthetic lying in real support)
    synthetic_recall: float            # Manifold recall (% real support covered by synthetic)
    real_sample_count: int
    synthetic_sample_count: int


def compute_rbf_mmd(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    """Computes Maximum Mean Discrepancy with Gaussian RBF kernel."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    m = len(y)

    # Pairwise distances
    xx = np.exp(-gamma * np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
    yy = np.exp(-gamma * np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1))
    xy = np.exp(-gamma * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1))

    mmd = np.mean(xx) + np.mean(yy) - 2.0 * np.mean(xy)
    return float(max(0.0, mmd))


class SyntheticFidelityEvaluator:
    """Evaluates generative quality of Klint trajectories against empirical market reality."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

    def evaluate_fidelity(
        self,
        real_trajectories: np.ndarray,      # (N_real, seq_len, 5) [O, H, L, C, V]
        synthetic_trajectories: np.ndarray, # (N_synth, seq_len, 5)
        epochs: int = 15,
        batch_size: int = 32,
    ) -> SyntheticFidelityResult:
        """
        Trains an LSTM discriminator to separate real vs synthetic trajectories,
        and computes distributional divergence metrics.
        """
        N_real = len(real_trajectories)
        N_synth = len(synthetic_trajectories)
        L = min(real_trajectories.shape[1], synthetic_trajectories.shape[1])

        # Normalize trajectories across sequence for numerical stability
        real_sub = real_trajectories[:, :L, :].astype(np.float32)
        synth_sub = synthetic_trajectories[:, :L, :].astype(np.float32)

        # Scale features using real stats
        mean = np.mean(real_sub, axis=(0, 1), keepdims=True)
        std = np.std(real_sub, axis=(0, 1), keepdims=True) + 1e-6
        real_norm = (real_sub - mean) / std
        synth_norm = (synth_sub - mean) / std

        # 1. Train LSTM Discriminator
        X = np.concatenate([real_norm, synth_norm], axis=0)
        y = np.concatenate([np.ones(N_real), np.zeros(N_synth)], axis=0).astype(np.float32)

        # Shuffle and split 80/20 train/test
        idx = np.random.permutation(len(X))
        split = int(0.8 * len(X))
        train_idx, test_idx = idx[:split], idx[split:]

        train_ds = TensorDataset(torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx]))
        test_ds = TensorDataset(torch.from_numpy(X[test_idx]), torch.from_numpy(y[test_idx]))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = LSTMDiscriminator(input_dim=X.shape[2], hidden_dim=48, num_layers=2).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for ep in range(epochs):
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                preds = (torch.sigmoid(model(bx)) >= 0.5).float()
                correct += int((preds == by).sum().item())
                total += len(by)

        acc = (correct / total) if total > 0 else 0.5
        # Discriminative score: |Acc - 0.5|
        disc_score = abs(acc - 0.5)

        # 2. Maximum Mean Discrepancy (on flattened sequence features)
        real_feats = real_norm.reshape(N_real, -1)
        synth_feats = synth_norm.reshape(N_synth, -1)
        # Subsample for fast kernel computation
        sub_n = min(200, N_real, N_synth)
        mmd = compute_rbf_mmd(real_feats[:sub_n], synth_feats[:sub_n], gamma=1.0 / real_feats.shape[1])

        # 3. 2-Wasserstein distance on log-returns
        real_ret = np.diff(np.log(real_sub[:, :, 3] + 1e-8), axis=1).flatten()
        synth_ret = np.diff(np.log(synth_sub[:, :, 3] + 1e-8), axis=1).flatten()
        w_dist = float(stats.wasserstein_distance(real_ret, synth_ret))

        # 4. Precision & Recall for Distributions
        # Precision: fraction of synthetic returns falling within 5th-95th percentile of real
        r_p05, r_p95 = np.percentile(real_ret, 5), np.percentile(real_ret, 95)
        prec = float(np.mean((synth_ret >= r_p05) & (synth_ret <= r_p95))) * 100.0

        s_p05, s_p95 = np.percentile(synth_ret, 5), np.percentile(synth_ret, 95)
        rec = float(np.mean((real_ret >= s_p05) & (real_ret <= s_p95))) * 100.0

        return SyntheticFidelityResult(
            discriminator_accuracy=float(acc * 100.0),
            discriminative_score=float(disc_score),
            mmd_score=float(mmd),
            wasserstein_distance=float(w_dist),
            synthetic_precision=prec,
            synthetic_recall=rec,
            real_sample_count=N_real,
            synthetic_sample_count=N_synth,
        )
