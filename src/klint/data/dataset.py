"""Chronological market dataset with leakage-free temporal splits and embargo periods."""

from typing import Tuple, Optional, Literal
import numpy as np
import torch
from torch.utils.data import Dataset
from klint.data.factors import FactorDecomposer, FactorStreams


class ChronologicalMarketDataset(Dataset):
    """
    Chronological PyTorch dataset for sliding-window sequence modeling.
    
    Prevents leakage by strictly partitioning continuous time series into:
    [Train] -> [Embargo] -> [Validation] -> [Embargo] -> [Test]
    """
    def __init__(
        self,
        data: np.ndarray,
        split: Literal["train", "val", "test"] = "train",
        context_bars: int = 1024,
        stride: int = 1,
        embargo_bars: int = 1440,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        decomposer: Optional[FactorDecomposer] = None,
    ):
        super().__init__()
        self.context_bars = context_bars
        self.stride = stride
        self.decomposer = decomposer or FactorDecomposer()

        if data.shape[-1] > 5:
            # Assume cols 1:6 are O, H, L, C, V if standard 11-col format
            if data.shape[-1] == 11:
                ohlcv = data[:, 1:6].astype(np.float64)
            else:
                ohlcv = data[:, :5].astype(np.float64)
        else:
            ohlcv = data.astype(np.float64)

        total_bars = len(ohlcv)
        train_end = int(total_bars * train_ratio)
        val_start = train_end + embargo_bars
        val_end = val_start + int(total_bars * val_ratio)
        test_start = val_end + embargo_bars

        if split == "train":
            split_slice = ohlcv[:train_end]
        elif split == "val":
            if val_start >= total_bars:
                raise ValueError("Dataset too short for validation split with given embargo.")
            split_slice = ohlcv[val_start:val_end]
        elif split == "test":
            if test_start >= total_bars:
                raise ValueError("Dataset too short for test split with given embargo.")
            split_slice = ohlcv[test_start:]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.ohlcv = split_slice.astype(np.float32)
        self.num_bars = len(self.ohlcv)
        
        if self.num_bars < self.context_bars:
            raise ValueError(
                f"Split '{split}' has {self.num_bars} bars, fewer than context window {self.context_bars}."
            )

        # Precompute sliding window start indices
        self.indices = list(range(0, self.num_bars - self.context_bars + 1, self.stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        start = self.indices[idx]
        end = start + self.context_bars
        window_ohlcv = self.ohlcv[start:end]

        factors = self.decomposer.decompose(window_ohlcv)

        return {
            "ohlcv": torch.from_numpy(window_ohlcv).float(),
            "price_path": factors.price_path,
            "range_shape": factors.range_shape,
            "activity": factors.activity,
            "anchor_price": torch.tensor(factors.anchor_price, dtype=torch.float32),
        }
