"""Unit tests for Klint-32M v2 fine-tuning components."""

import os
import pytest
import torch

from klint.models.klint_32m import Klint32M, KlintConfig
from klint.tokenizer.factor_tokenizer import FactorTokenizer
from klint.finetune.loss import PnLWeightedCrossEntropyLoss
from klint.finetune.multi_asset_dataset import MultiAssetFineTuneDataset
from klint.finetune.pnl_trainer import KlintPnLTrainer


@pytest.fixture
def tokenizer():
    return FactorTokenizer()


@pytest.fixture
def small_model():
    cfg = KlintConfig(
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        price_vocab_size=512,
        range_vocab_size=256,
        activity_vocab_size=256,
        max_seq_len=512,
    )
    return Klint32M(cfg)


def test_pnl_loss_gradient_flow(tokenizer):
    """Verifies that PnLWeightedCrossEntropyLoss computes loss and flows gradients back."""
    criterion = PnLWeightedCrossEntropyLoss(tokenizer, lambda_pnl=3.0, gamma_dir=2.0)

    B, seq_len = 2, 24
    lp = torch.randn(B, seq_len, 512, requires_grad=True)
    lr = torch.randn(B, seq_len, 256, requires_grad=True)
    la = torch.randn(B, seq_len, 256, requires_grad=True)
    targets = torch.randint(0, 200, (B, seq_len))

    loss, metrics = criterion(lp, lr, la, targets)

    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert "hit_rate" in metrics
    assert "loss_dir" in metrics
    assert "mean_pnl_weight" in metrics

    loss.backward()
    assert lp.grad is not None
    assert lp.grad.norm().item() > 0
    assert lr.grad is not None
    assert la.grad is not None


def test_multi_asset_dataset_token_shift(tokenizer):
    """Verifies autoregressive shift in MultiAssetFineTuneDataset."""
    ds = MultiAssetFineTuneDataset(
        tickers=[],
        local_files=[],
        tokenizer=tokenizer,
        context_bars=32,
        stride_bars=16,
    )
    assert len(ds) > 0
    sample = ds[0]

    inputs = sample["inputs"]
    targets = sample["targets"]

    assert len(inputs) == 32 * 3 - 1
    assert len(targets) == len(inputs)
    # Target at t is input at t+1
    assert torch.equal(inputs[1:], targets[:-1])


def test_trainer_single_step_and_bundle(small_model, tokenizer, tmp_path):
    """Verifies KlintPnLTrainer step execution and release bundle serialization."""
    trainer = KlintPnLTrainer(
        model=small_model,
        tokenizer=tokenizer,
        learning_rate=1e-3,
        total_steps=5,
        device="cpu",
    )

    B, seq_len = 2, 24
    dummy_batch = {
        "inputs": torch.randint(0, 200, (B, seq_len)),
        "targets": torch.randint(0, 200, (B, seq_len)),
    }

    metrics = trainer.train_step(dummy_batch)
    assert "loss_total" in metrics
    assert "hit_rate" in metrics
    assert trainer.step_count == 1

    # Test release bundle packaging
    bundle_path = os.path.join(tmp_path, "klint_32m_v2_test.pt")
    trainer.save_release_bundle(bundle_path, step=1, val_loss=metrics["loss_total"], val_hit_rate=metrics["hit_rate"])

    assert os.path.exists(bundle_path)
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    assert "config" in bundle
    assert "model_state_dict" in bundle
    assert "tokenizer_state_dict" in bundle
    assert bundle["training_meta"]["version"] == "v2"
