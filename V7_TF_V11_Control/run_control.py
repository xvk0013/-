"""Train the copied original TF model once, then return one consolidated report."""
import argparse
import json
import subprocess
import sys
from v7_common import (HERE, DATASET, OUTPUT, RUN_DIR, REPORT_DIR, BASELINE_SUMMARY,
                       BN_CONTROL_SUMMARY, TF_V6_SUMMARY, ALTERNATE_ROOT, TF_PROTOCOL, BN_POLICY, validate_dataset)


def training_command():
    command = [sys.executable, "-u", str(HERE / "train.py"),
               "--phase", "single", "--data-dir", str(DATASET), "--noise-data-dir", str(DATASET),
               "--output-dir", str(RUN_DIR), "--device", "cuda", "--scan-backend", "torch",
               "--epochs", "60", "--batch-size", "32", "--workers", "4", "--seed", "42",
               "--head-type", "query", "--single-ce-weight", "0.2", "--weight-decay", "0.0001",
               "--band-fusion", "early", "--local-contrast-bins", "0",
               "--spectral-aug-prob", "0", "--feature-mixup-prob", "0"]
    if (RUN_DIR / "last.pt").is_file():
        command += ["--resume", str(RUN_DIR / "last.pt")]
    return command


def compare(before, after):
    def values(split):
        metrics = split["metrics"]
        out = {"single_emr": metrics["single"]["emr"],
               "single_macro_f1": metrics["single"]["macro_f1"],
               "noise_false_alarm": metrics["noise_false_alarm"]}
        for name in ("Tanker", "Cargo", "Tug"):
            out[name + "_strict_accuracy"] = split["unique_source_level"][name]["formal_exact_accuracy"]
        return out
    a, b = values(before), values(after)
    return {key: {"baseline": a[key], "tf": b[key], "tf_minus_baseline_pp": 100 * (b[key] - a[key])}
            for key in a}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    print(json.dumps(TF_PROTOCOL, indent=2), flush=True)
    if args.describe:
        return
    final = OUTPUT / "summary.json"
    if final.is_file():
        saved = json.loads(final.read_text())
        if saved.get("status") != "complete" or saved.get("tf_protocol") != TF_PROTOCOL:
            raise ValueError("Existing final summary does not match this TF experiment")
        print(f"Already complete; no extra run. Return ONLY: {final}", flush=True)
        return
    generation = validate_dataset()
    for name in ("summary.json", "manifest.json"):
        if not (ALTERNATE_ROOT / name).is_file():
            raise FileNotFoundError(f"Existing channel-augmentation data required: {ALTERNATE_ROOT / name}")
    OUTPUT.mkdir(exist_ok=True)
    marker = OUTPUT / "training_complete.json"
    if not marker.is_file():
        if RUN_DIR.exists() and not (RUN_DIR / "last.pt").is_file():
            if any(RUN_DIR.iterdir()):
                raise RuntimeError("Run contains files without a committed last.pt; refusing overwrite")
            RUN_DIR.rmdir()
        print("STAGE 1/2: TF TRAINING 1/1, total 60 epochs; existing Bellhop data reused", flush=True)
        subprocess.run(training_command(), cwd=HERE, check=True)
        marker.write_text(json.dumps(TF_PROTOCOL, indent=2), encoding="utf-8")
    elif json.loads(marker.read_text()) != TF_PROTOCOL:
        raise ValueError("Training completion marker belongs to a different protocol")
    if not (REPORT_DIR / "summary.json").is_file():
        print("STAGE 2/2: saved best EMA on original full Train/Val; Test NOT LOADED", flush=True)
        subprocess.run([sys.executable, "-u", str(HERE / "report.py"),
                        "--checkpoint", str(RUN_DIR / "best.pt"), "--data-dir", str(DATASET),
                        "--device", "cuda", "--workers", "4", "--output-dir", str(REPORT_DIR)],
                       cwd=HERE, check=True)
    result = json.loads((REPORT_DIR / "summary.json").read_text())
    if result.get("tf_protocol") != TF_PROTOCOL or result.get("bn_policy") != BN_POLICY:
        raise ValueError("Report does not belong to the copied original TF run")
    if abs(result["val_reproduction_emr_delta"]) > 1e-7:
        raise ValueError("Reported Val EMR does not reproduce the selected checkpoint")
    history = [json.loads(line) for line in (RUN_DIR / "history.jsonl").read_text().splitlines() if line.strip()]
    if [row["epoch"] for row in history] != list(range(1, 61)):
        raise ValueError("Expected exactly 60 completed epochs")
    for row in history:
        if (row["train"]["optimizer_steps"] != 203
                or row["source_pool_coverage"]["epoch_ship_draws"] != 5400
                or row["source_pool_coverage"]["epoch_noise_draws"] != 1080):
            raise ValueError("The fixed training exposure budget changed")
    baseline = json.loads(BASELINE_SUMMARY.read_text())
    bn_control = json.loads(BN_CONTROL_SUMMARY.read_text())
    tf_v6 = json.loads(TF_V6_SUMMARY.read_text())
    reference_val = bn_control["variants"]["baseline_bn_calibrated"]
    result.update(
        status="complete", protocol=TF_PROTOCOL, generation_summary=generation,
        training_history=history, training_record=json.loads((RUN_DIR / "config.json").read_text()),
        baseline_summary=str(BASELINE_SUMMARY), baseline_best_epoch=baseline["epoch"],
        comparison_to_historical_baseline={split: compare(baseline["splits"][split], result["splits"][split])
                                           for split in ("Train", "Val")},
        comparison_to_bn_calibrated_baseline={"Val": compare(reference_val, result["splits"]["Val"])},
        comparison_to_full_v6_tf={split: compare(tf_v6["splits"][split], result["splits"][split]) for split in ("Train", "Val")},
        reference_normalization_note="V11 TF uses Train-only calibrated EMA BN. Mamba calibrated reference is a fixed historical checkpoint, not a rerun with epochwise calibrated selection. V6 TF uses GN/LN without BN. Architecture/head comparisons, not pure backbone effects.",
        val_confusion_counts={"mamba_original": baseline["splits"]["Val"]["confusion"],
                              "mamba_bn_calibrated": reference_val["confusion"],
                              "tf_v6": tf_v6["splits"]["Val"]["confusion"],
                              "tf": result["splits"]["Val"]["confusion"]},
        observed_alternate_ship_fraction=sum(row["train"]["alternate_ship_draws"] for row in history) / (60 * 5400),
    )
    result["cargo_tanker_counts"] = {
        name: {"Cargo_to_Tanker": matrix[2][1], "Tanker_to_Cargo": matrix[1][2], "Cargo_to_noise": matrix[2][0]}
        for name, matrix in result["val_confusion_counts"].items()}
    temporary = OUTPUT / "summary.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    from checkpoint_io import replace_checkpoint
    replace_checkpoint(temporary, final)
    print(f"COMPLETE. Return ONLY: {final}", flush=True)


if __name__ == "__main__":
    main()
