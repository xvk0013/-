"""Explicit evaluation of a saved EMA checkpoint; never used for model selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import scipy.signal
import torch
from config import Config
from data import SceneDataset, load_records
from model import M2Mamba
from train import load_checkpoint, make_loader, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("Val", "Test"), required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scan-backend", choices=("torch", "cuda"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    saved = load_checkpoint(args.checkpoint)
    if Path(saved["data_dir"]).resolve() != args.data_dir.resolve():
        parser.error("Checkpoint and dataset paths differ")
    cfg = Config(**saved["config"])
    cfg.batch_size = args.batch_size
    cfg.workers = 0
    if args.scan_backend:
        cfg.scan_backend = args.scan_backend
    device = torch.device(args.device)
    model = M2Mamba(cfg, saved["phase"]).to(device)
    model.load_state_dict(saved["ema"], strict=True)
    records = load_records(args.data_dir, args.split, saved["phase"], cfg)
    loader = make_loader(SceneDataset(records, cfg, saved["stats"]), cfg, False, device)
    metrics = evaluate(model, loader, device)
    print(json.dumps({"split": args.split, "phase": saved["phase"], "metrics": metrics},
                     indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
