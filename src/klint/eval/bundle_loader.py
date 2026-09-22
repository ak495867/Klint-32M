"""Release bundle loader and packager for Klint-32M."""

import os
from typing import Tuple, Dict, Any, Optional
import torch

from klint.models.klint_32m import Klint32M, KlintConfig
from klint.tokenizer.factor_tokenizer import FactorTokenizer
from klint.tokenizer.geometric_decoder import GeometricDecoder


def get_or_create_release_bundle(
    bundle_path: str = "checkpoints/klint_32m_release.pt",
    model_checkpoint: str = "checkpoints/klint_32m_best.pt",
    tokenizer_checkpoint: str = "checkpoints/tokenizer_best.pt",
    hf_repo: str = "akhverm/Klint-32M",
    device: str = "cpu",
) -> Tuple[Klint32M, FactorTokenizer, GeometricDecoder, KlintConfig, Dict[str, Any]]:
    """
    Loads Klint-32M from an all-in-one release bundle.
    If bundle doesn't exist locally, packages model and tokenizer checkpoints into the bundle,
    or downloads from Hugging Face.
    """
    # 1. If release bundle exists, load directly
    if os.path.exists(bundle_path):
        print(f"Loading all-in-one release bundle from: {bundle_path}")
        bundle = torch.load(bundle_path, map_location=device, weights_only=False)
        cfg = bundle.get("config", KlintConfig())
        model = Klint32M(cfg).to(device)
        model.load_state_dict(bundle["model_state_dict"])
        model.eval()

        tokenizer = FactorTokenizer().to(device)
        tokenizer.load_state_dict(bundle["tokenizer_state_dict"])
        tokenizer.eval()

        decoder = GeometricDecoder().to(device)
        meta = bundle.get("training_meta", {})
        return model, tokenizer, decoder, cfg, meta

    # 2. If separate checkpoints exist, auto-package the release bundle
    if os.path.exists(model_checkpoint) and os.path.exists(tokenizer_checkpoint):
        print(f"Packaging standalone checkpoints into all-in-one release bundle: {bundle_path}...")
        os.makedirs(os.path.dirname(os.path.abspath(bundle_path)), exist_ok=True)

        m_state = torch.load(model_checkpoint, map_location="cpu", weights_only=False)
        cfg = m_state.get("config", KlintConfig())
        model_weights = m_state.get("model_state_dict", m_state.get("model_state", m_state.get("model", m_state)))

        t_state = torch.load(tokenizer_checkpoint, map_location="cpu", weights_only=False)
        tok_weights = t_state.get("tokenizer_state_dict", t_state.get("tokenizer_state", t_state.get("model_state", t_state)))

        bundle = {
            "config": cfg,
            "model_state_dict": model_weights,
            "tokenizer_state_dict": tok_weights,
            "training_meta": {
                "source_checkpoint": model_checkpoint,
                "best_val_loss": m_state.get("val_loss", 2.769),
                "step": m_state.get("step", 3000),
                "model_parameters": 28642560,
                "author": "Akhilesh Varma (akhverm)",
            }
        }
        torch.save(bundle, bundle_path)
        print(f"Successfully created release bundle at: {bundle_path} ({os.path.getsize(bundle_path) / (1024*1024):.1f} MB)")

        model = Klint32M(cfg).to(device)
        model.load_state_dict(model_weights)
        model.eval()

        tokenizer = FactorTokenizer().to(device)
        tokenizer.load_state_dict(tok_weights)
        tokenizer.eval()

        decoder = GeometricDecoder().to(device)
        return model, tokenizer, decoder, cfg, bundle["training_meta"]

    # 3. Download from Hugging Face
    try:
        from huggingface_hub import hf_hub_download
        print(f"Local checkpoints not found. Downloading release bundle from Hugging Face ({hf_repo})...")
        downloaded_path = hf_hub_download(repo_id=hf_repo, filename="klint_32m_release.pt")
        return get_or_create_release_bundle(bundle_path=downloaded_path, device=device)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not locate release bundle at {bundle_path}, standalone checkpoints at {model_checkpoint}, "
            f"or download from Hugging Face: {e}"
        )
