"""One fresh V7 GAP run, then one full Train/Val EMA report."""
import argparse
import json
import subprocess
import sys
from v7_common import DATASET, HERE, OUTPUT, REPORT_DIR, RUN_DIR, validate_dataset

PROTOCOL = {
    "name": "single_v7_bellhop_gap_ce02_ship5_noise1_s42",
    "training_runs": 1,
    "initialization": "fresh; no previous experiment checkpoint or statistics",
    "seed": 42,
    "epochs": 60,
    "batch_size": 32,
    "ships_per_class_per_epoch": 1800,
    "noise_per_epoch": 1080,
    "optimizer_steps_per_epoch": 203,
    "optimizer_steps_total": 12180,
    "selection": "best EMA Val single_emr",
    "post_training_evaluation": "one best-EMA pass over full Train and full Val",
    "test_loaded": False,
}

def training_command():
    command = [
        sys.executable, "-u", str(HERE / "train_v7.py"),
        "--phase", "single", "--data-dir", str(DATASET),
        "--noise-data-dir", str(DATASET), "--output-dir", str(RUN_DIR),
        "--device", "cuda", "--scan-backend", "cuda",
        "--epochs", "60", "--batch-size", "32", "--workers", "4",
        "--seed", "42", "--head-type", "gap", "--single-ce-weight", "0.2",
        "--weight-decay", "0.0001", "--band-fusion", "early",
        "--local-contrast-bins", "0", "--spectral-aug-prob", "0",
        "--feature-mixup-prob", "0",
    ]
    if (RUN_DIR / "last.pt").is_file():
        command += ["--resume", str(RUN_DIR / "last.pt")]
    return command

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true",
                        help="Print the fixed plan without loading audio or training.")
    args = parser.parse_args()
    generation = validate_dataset()
    print(json.dumps({"dataset": str(DATASET), **PROTOCOL}, indent=2), flush=True)
    print("Same recording-balanced sampling budget as the old GAP baseline; "
          "all new Train slices are candidates, not a full-pool pass each epoch.", flush=True)
    print("Per-epoch EMA and online Val checks are retained; "
          "only EMA selects the best checkpoint.", flush=True)
    if args.describe:
        return
    final = OUTPUT / "summary.json"
    if final.is_file():
        print(f"Already complete; no training or inference repeated. Return: {final}", flush=True)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    completed = OUTPUT / "training_complete.json"
    if not completed.is_file():
        if RUN_DIR.exists() and not (RUN_DIR / "last.pt").is_file():
            raise RuntimeError(f"{RUN_DIR} exists without last.pt. No run is overwritten.")
        print("TRAINING 1/1: GAP seed 42 on V7; Test NOT LOADED", flush=True)
        subprocess.run(training_command(), cwd=HERE, check=True)
        completed.write_text(json.dumps(PROTOCOL, indent=2), encoding="utf-8")
    best = RUN_DIR / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    report_summary = REPORT_DIR / "summary.json"
    if not report_summary.is_file():
        print("FINAL REPORT 1/1: full Train and Val; no model updates", flush=True)
        subprocess.run([
            sys.executable, "-u", str(HERE / "report_v7.py"),
            "--checkpoint", str(best), "--data-dir", str(DATASET),
            "--device", "cuda", "--workers", "4", "--output-dir", str(REPORT_DIR),
        ], cwd=HERE, check=True)
    result = json.loads(report_summary.read_text(encoding="utf-8"))
    result["experiment"] = PROTOCOL
    result["generation_summary"] = generation
    result["training_record"] = json.loads((RUN_DIR / "config.json").read_text(encoding="utf-8"))
    result["training_history"] = [
        json.loads(line) for line in (RUN_DIR / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    temporary = OUTPUT / "summary.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(final)
    print(f"COMPLETE. Only return this file: {final}", flush=True)

if __name__ == "__main__":
    main()
