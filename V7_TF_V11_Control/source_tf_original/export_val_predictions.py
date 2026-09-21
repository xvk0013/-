"""Export the unchanged best EMA on Val after the CR=0 single-target run."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import scipy.signal  # Match the existing project's import order.
import torch

from config import Config
from data import SceneDataset, load_records
from diagnose_ema_bn import evaluate_variant
from model import M2Mamba
from train import load_checkpoint, make_loader, seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    saved = load_checkpoint(args.checkpoint)
    if saved["phase"] != "single":
        parser.error("This export requires a single-phase checkpoint")
    if Path(saved["data_dir"]).resolve() != args.data_dir.resolve():
        parser.error("Checkpoint and dataset paths differ")
    cfg = Config(**saved["config"])
    if args.workers is not None:
        cfg.workers = args.workers
    cfg.validate()
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("Only cpu/cuda are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    if cfg.scan_backend == "cuda":
        if device.type != "cuda":
            parser.error("The saved CUDA scan backend requires --device cuda")
        from scan import cuda_scan
        cuda_scan()
    seed_all(cfg.seed)
    records = load_records(args.data_dir, "Val", "single")
    loader = make_loader(SceneDataset(records, cfg, saved["stats"]), cfg, False, device)
    model = M2Mamba(cfg, "single").to(device).requires_grad_(False)
    model.load_state_dict(saved["ema"], strict=True)
    output_dir = args.checkpoint.resolve().parent / (
        "val_report_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Epoch={saved['epoch']}, CR={cfg.cr_weight}, Val={len(records)}; Test NOT LOADED", flush=True)
    result = evaluate_variant(model, loader, device, output_dir, "ema")
    report = {"checkpoint": str(args.checkpoint.resolve()), "epoch": saved["epoch"],
              "split": "Val", "variant": "ema", "cr_weight": cfg.cr_weight,
              "config": cfg.to_dict(), "data_dir": str(args.data_dir.resolve()),
              "threshold": 0.5, "precision": "float32", "saved_ema_val": saved.get("val"),
              "test_loaded": False, **result}
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    print(f"COMPLETE: {output_dir}; Test NOT LOADED", flush=True)


if __name__ == "__main__":
    main()
