"""
Master CLI and Script Runner for TLSTM-Klint Distillation and 10-Test Battery.

Usage:
  python Tlstm-Klint/run_all_tests.py \
      --teacher_checkpoint checkpoints/klint_32m_v2_release.pt \
      --student_checkpoint checkpoints/tlstm_klint_distilled.pt \
      --max_assets 325 \
      --output_dir Tlstm-Klint/TLSTM-multitest \
      --epochs 5 \
      --batch_size 64
"""

import os
import sys
import argparse
import time
import torch

try:
    from .tlstm_model import TLSTMKlint, TLSTMConfig
    from .trainer import DistillationTrainer
    from .fresh_universe import FRESH_ASSETS_BY_CLASS
    from .data_loader import FreshUniverseDataLoader
    from .evaluator import TLSTMGPUEvaluator
    from .test_battery import TLSTMTestBattery
except (ImportError, ValueError):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tlstm_model import TLSTMKlint, TLSTMConfig
    from trainer import DistillationTrainer
    from fresh_universe import FRESH_ASSETS_BY_CLASS
    from data_loader import FreshUniverseDataLoader
    from evaluator import TLSTMGPUEvaluator
    from test_battery import TLSTMTestBattery


def main():
    parser = argparse.ArgumentParser(description="Distill Klint-32M v2 into TLSTM and run 10-test battery.")
    parser.add_argument("--teacher_checkpoint", type=str, default="checkpoints/klint_32m_v2_release.pt")
    parser.add_argument("--student_checkpoint", type=str, default="checkpoints/tlstm_klint_distilled.pt")
    parser.add_argument("--max_assets", type=int, default=325)
    parser.add_argument("--output_dir", type=str, default="Tlstm-Klint/TLSTM-multitest")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--skip_training", action="store_true", help="Skip training if student checkpoint exists")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("🏆 TLSTM-Klint: Knowledge Distillation & Institutional Multi-Test Suite")
    print(f"   * Teacher Model:      {args.teacher_checkpoint}")
    print(f"   * Student Target:     {args.student_checkpoint}")
    print(f"   * Evaluation Assets:  {args.max_assets} Fresh Assets")
    print(f"   * Output Directory:   {args.output_dir}")
    print(f"   * Execution Device:   {device}")
    print("=" * 80)

    # -------------------------------------------------------------
    # STAGE 1: Distillation Training (if needed)
    # -------------------------------------------------------------
    if not os.path.exists(args.student_checkpoint) or not args.skip_training:
        print("\n[STAGE 1/3] Training TLSTM Student via Knowledge Distillation...")
        trainer = DistillationTrainer(
            teacher_checkpoint=args.teacher_checkpoint,
            student_config=TLSTMConfig(),
            device=device,
        )
        train_loader = trainer.prepare_training_data(
            context_bars=64,
            stride_bars=16,
            period="1y",
            interval="1d",
            verbose=True,
        )
        trainer.train_distillation(
            dataloader=train_loader,
            epochs=args.epochs,
            save_checkpoint=args.student_checkpoint,
            verbose=True,
        )
    else:
        print(f"\n[STAGE 1/3] Found existing student checkpoint at: {args.student_checkpoint}. Skipping training.")

    # -------------------------------------------------------------
    # STAGE 2: Ingest 325 Fresh Assets
    # -------------------------------------------------------------
    print(f"\n[STAGE 2/3] Ingesting {args.max_assets} Strictly Out-of-Sample Fresh Assets...")
    all_fresh_assets = []
    for cls_name, asset_list in FRESH_ASSETS_BY_CLASS.items():
        for asset in asset_list:
            all_fresh_assets.append({
                "ticker": asset["ticker"],
                "name": asset["name"],
                "asset_class": cls_name,
            })
    selected_assets = all_fresh_assets[: args.max_assets]

    loader = FreshUniverseDataLoader(cache_dir="Tlstm-Klint/cache")
    fresh_dataset = loader.load_universe(selected_assets, verbose=True)

    # -------------------------------------------------------------
    # STAGE 3: GPU Evaluation & 10-Test Battery
    # -------------------------------------------------------------
    print(f"\n[STAGE 3/3] Evaluating Student Model and Generating 20 Publication-Grade Plots...")
    evaluator = TLSTMGPUEvaluator(
        checkpoint_path=args.student_checkpoint,
        device=device,
        batch_size=args.batch_size,
    )
    eval_results = evaluator.evaluate_universe(fresh_dataset, verbose=True)

    battery = TLSTMTestBattery(eval_results=eval_results, output_dir=args.output_dir)
    results = battery.run_all(verbose=True)

    print("\n" + "=" * 80)
    print("✅ Benchmark Suite Execution Completed Successfully!")
    print(f"📊 20 Visual Plots and Executive Report available in: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
