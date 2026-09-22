"""Predict one mono 16 kHz / 5 s mix WAV using a trained checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import scipy.signal
import torch
from config import Config
from data import read_waveform
from frontend import DualBandTransform, DEMONTransform
from model import M2Mamba
from train import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scan-backend", choices=("torch", "cuda"))
    args = parser.parse_args()
    saved = load_checkpoint(args.checkpoint)
    cfg = Config(**saved["config"])
    if args.scan_backend:
        cfg.scan_backend = args.scan_backend
    device = torch.device(args.device)
    model = M2Mamba(cfg, saved["phase"]).to(device).eval()
    model.load_state_dict(saved["ema"], strict=True)
    waveform = read_waveform(args.wav, cfg)
    stats = saved["stats"]
    features = DualBandTransform(cfg, stats["mean"], stats["std"])(waveform)
    demon = DEMONTransform(cfg)(waveform)
    with torch.no_grad():
        outputs = model(features[None].to(device), demon[None].to(device))
        pred = model.decode(outputs)[0].cpu().bool()
        scores = outputs["presence_logits"][0].sigmoid().cpu().tolist()
    result = {"phase": saved["phase"], "classes": [name for name, present in zip(cfg.class_names, pred) if present],
              "presence_scores": dict(zip(cfg.class_names, scores))}
    if saved["phase"] == "joint":
        from m2_components import LEGAL_SET_MULTI_HOT
        legal_names = ["+".join(name for name, present in zip(cfg.class_names, row) if bool(present))
                       or "noise" for row in LEGAL_SET_MULTI_HOT]
        result["legal_set_probabilities"] = dict(zip(legal_names, outputs["set_log_probs"][0].exp().cpu().tolist()))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
