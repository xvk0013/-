"""Compare online, EMA and Train-only BN-reestimated EMA on single-phase Val."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import scipy.signal  # Keep the project's scipy-before-torch import order.
import torch
from torch import nn
import torch.nn.functional as F

from config import Config
from data import CLASS_NAMES, SceneDataset, load_records
from metrics import summarize, format_noise_metrics
from model import M2Mamba
from train import load_checkpoint, make_loader, seed_all


# Bit positions follow the saved config class order; retain all eight BCE predictions.
SET_CODES = (0, 1, 2, 4, 3, 5, 6, 7)
TRUE_CODES = (0, 1, 2, 4)
VARIANTS = ("online", "ema", "ema_bn_reestimated")


def set_name(code, class_names=CLASS_NAMES):
    return "+".join(name for i, name in enumerate(class_names) if code & (1 << i)) or "noise"


def set_codes(values):
    return (values.long() * torch.tensor([1, 2, 4])).sum(1)


@torch.no_grad()
def reestimate_bn(model, loader, device):
    """Update only BN buffers; all weights and stochastic layers stay frozen."""
    model.eval()
    layers = [module for module in model.modules()
              if isinstance(module, nn.modules.batchnorm._BatchNorm)
              and module.track_running_stats]
    if not layers:
        raise ValueError("The model has no BatchNorm running statistics to reestimate")
    original_momenta = [module.momentum for module in layers]
    seen = 0
    try:
        for module in layers:
            module.reset_running_stats()
            module.train()
        for step, batch in enumerate(loader, 1):
            size = len(batch["labels"])
            # Sample-weighted cumulative batch statistics, including the last batch.
            for module in layers:
                module.momentum = size / (seen + size)
            outputs = model(batch["features"].to(device, non_blocking=True),
                            batch["demon"].to(device, non_blocking=True))
            if not bool(torch.isfinite(outputs["presence_logits"]).all()):
                raise FloatingPointError("Non-finite logits during Train-only BN reestimation")
            seen += size
            if step == 1 or step % 25 == 0 or step == len(loader):
                print(f"  BN Train: {step}/{len(loader)} batches, {seen} samples", flush=True)
        for module in layers:
            if not bool(torch.isfinite(module.running_mean).all()
                        and torch.isfinite(module.running_var).all()):
                raise FloatingPointError("Non-finite reestimated BN statistics")
    finally:
        for module, momentum in zip(layers, original_momenta):
            module.momentum = momentum
        model.eval()
    return {"split": "Train", "samples": seen, "bn_layers": len(layers),
            "augmentation": False, "dropout": False, "drop_path": False,
            "method": "sample-weighted cumulative batch statistics; one shuffled pass"}


@torch.no_grad()
def evaluate_variant(model, loader, device, output_dir, variant, split_name="Val"):
    class_names = tuple(loader.dataset.cfg.class_names)
    model.eval()
    all_predictions, all_labels, all_sirs = [], [], []
    confusion = torch.zeros((len(TRUE_CODES), len(SET_CODES)), dtype=torch.long)
    n_samples, bce_sum = 0, 0.0
    prediction_file = output_dir / f"predictions_{variant}.csv"
    with prediction_file.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["path", "recording", "truth", "prediction", "exact_match"]
        fields += [f"prob_{name}" for name in class_names]
        fields += [f"logit_{name}" for name in class_names]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step, batch in enumerate(loader, 1):
            labels = batch["labels"].to(device, non_blocking=True)
            outputs = model(batch["features"].to(device, non_blocking=True),
                            batch["demon"].to(device, non_blocking=True))
            logits = outputs["presence_logits"]
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError(f"Non-finite Val logits: {variant}")
            size = len(labels)
            bce_sum += float(F.binary_cross_entropy_with_logits(logits, labels)) * size
            n_samples += size
            predictions = model.decode(outputs).cpu()
            labels = labels.cpu()
            logits = logits.cpu()
            probabilities = logits.sigmoid()
            truth_codes, prediction_codes = set_codes(labels), set_codes(predictions)
            for i, truth in enumerate(TRUE_CODES):
                for j, prediction in enumerate(SET_CODES):
                    confusion[i, j] += ((truth_codes == truth) & (prediction_codes == prediction)).sum()
            for i in range(size):
                truth, prediction = int(truth_codes[i]), int(prediction_codes[i])
                row = {"path": batch["path"][i], "recording": batch["recording"][i],
                       "truth": set_name(truth, class_names), "prediction": set_name(prediction, class_names),
                       "exact_match": int(truth == prediction)}
                row.update({f"prob_{name}": float(probabilities[i, j])
                            for j, name in enumerate(class_names)})
                row.update({f"logit_{name}": float(logits[i, j])
                            for j, name in enumerate(class_names)})
                writer.writerow(row)
            all_predictions.append(predictions)
            all_labels.append(labels)
            all_sirs.append(batch["sir_db"])
            if step == 1 or step % 25 == 0 or step == len(loader):
                print(f"  {variant} {split_name}: {step}/{len(loader)} batches", flush=True)
    if not n_samples:
        raise ValueError("Empty Val loader")
    metrics = summarize(torch.cat(all_predictions), torch.cat(all_labels), torch.cat(all_sirs),
                        class_names=class_names)
    metrics.update(bce=bce_sum / n_samples, nll=None)
    with (output_dir / f"confusion_{variant}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["truth / prediction"] + [set_name(code, class_names) for code in SET_CODES])
        for code, row in zip(TRUE_CODES, confusion.tolist()):
            writer.writerow([set_name(code, class_names)] + row)
    directions = []
    for i, truth in enumerate(TRUE_CODES):
        total = int(confusion[i].sum())
        for j, prediction in enumerate(SET_CODES):
            count = int(confusion[i, j])
            if prediction != truth and count:
                directions.append({"truth": set_name(truth, class_names), "prediction": set_name(prediction, class_names),
                                   "count": count, "fraction_of_true_class": count / total})
    directions.sort(key=lambda item: (-item["count"], item["truth"], item["prediction"]))
    print(f"{variant}: EMR={metrics['emr']:.6f}, single EMR={metrics['single']['emr']:.6f}, "
          f"single Macro-F1={metrics['single']['macro_f1']:.6f}, "
          f"recall={metrics['single']['recall']}; {format_noise_metrics(metrics)}", flush=True)
    return {"metrics": metrics, "confusion": confusion.tolist(), "error_directions": directions}


def write_summary(output_dir, report):
    class_names = tuple(report["config"]["class_names"])
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Online / EMA / Train-only BN reestimation", "",
             f"Checkpoint: {report['checkpoint']}", f"Epoch: {report['epoch']}", "",
             "All metrics use the same Val samples, saved Train feature normalization, FP32, "
             "and the original independent 0.5 thresholds. Only BN running buffers are "
             "changed for ema_bn_reestimated; no optimizer updates or checkpoint writes.", "",
             "BN calibration: one shuffled pass over all single-phase Train samples (including noise); "
             "no augmentation, Dropout or DropPath. Test NOT LOADED.", "",
             "| Variant | Overall EMR | Ship single EMR | Ship Macro-F1 | "
             + " | ".join(name + " recall" for name in class_names) + " | Noise FA | BCE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        metrics = report["variants"][variant]["metrics"]
        single = metrics["single"]
        values = [metrics["emr"], single["emr"], single["macro_f1"]]
        values += [single["recall"][name] for name in class_names]
        values += [metrics["noise_false_alarm"]]
        lines.append(f"| {variant} | " + " | ".join(f"{value:.4%}" for value in values)
                     + f" | {metrics['bce']:.6f} |")
    lines += ["", "Saved EMA validation metrics are retained in summary.json for reproduction comparison.",
              "Multiple-positive predictions count as errors on a single ship. "
              "Confusion CSV rows are truth and columns are predicted sets."]
    for variant in VARIANTS:
        lines += ["", f"## {variant}: error directions", "",
                  "| Truth | Predicted set | Count | Fraction of true class |", "|---|---|---:|---:|"]
        for row in report["variants"][variant]["error_directions"]:
            lines.append(f"| {row['truth']} | {row['prediction']} | {row['count']} | "
                         f"{row['fraction_of_true_class']:.4%} |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="New directory; defaults to a timestamped checkpoint sibling")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scan-backend", choices=("torch", "cuda"))
    parser.add_argument("--batch-size", type=int, help="Defaults to the checkpoint's training batch size")
    parser.add_argument("--workers", type=int, help="Defaults to the checkpoint configuration")
    args = parser.parse_args()
    saved = load_checkpoint(args.checkpoint)
    if saved["phase"] != "single":
        parser.error("This diagnostic is for a single-phase checkpoint")
    if Path(saved["data_dir"]).resolve() != args.data_dir.resolve():
        parser.error("Checkpoint and dataset paths differ; use the original training dataset path")
    cfg = Config(**saved["config"])
    for key in ("scan_backend", "batch_size", "workers"):
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)
    cfg.validate()
    class_names = tuple(cfg.class_names)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("Only cpu/cuda are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    if cfg.scan_backend == "cuda":
        if device.type != "cuda":
            parser.error("--scan-backend cuda requires --device cuda")
        from scan import cuda_scan
        cuda_scan()
    seed_all(cfg.seed)
    # Keep the saved feature normalization; never fit it again or read Test metadata.
    train_records = load_records(args.data_dir, "Train", "single", cfg)
    val_records = load_records(args.data_dir, "Val", "single", cfg)
    train_dataset = SceneDataset(train_records, cfg, saved["stats"], training=False)
    val_dataset = SceneDataset(val_records, cfg, saved["stats"], training=False)
    # Shuffling calibration avoids class-sorted batches; it does not enable augmentation.
    train_loader = make_loader(train_dataset, cfg, True, device)
    val_loader = make_loader(val_dataset, cfg, False, device)
    output_dir = args.output_dir or args.checkpoint.parent / (
        "ema_bn_diagnostic_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    model = M2Mamba(cfg, "single").to(device).requires_grad_(False)
    report = {"checkpoint": str(args.checkpoint.resolve()), "epoch": saved["epoch"],
              "data_dir": str(args.data_dir.resolve()), "evaluation_split": "Val",
              "train_samples": len(train_records), "val_samples": len(val_records),
              "config": cfg.to_dict(), "feature_stats": saved["stats"],
              "precision": "float32 (no autocast)", "threshold": 0.5,
              "torch_version": str(torch.__version__), "device": str(device),
              "test_loaded": False, "saved_ema_val": saved.get("val"),
              "confusion_rows": [set_name(code, class_names) for code in TRUE_CODES],
              "confusion_columns": [set_name(code, class_names) for code in SET_CODES], "variants": {}}
    print(f"Checkpoint epoch={saved['epoch']}; Train={len(train_records)}, Val={len(val_records)}; "
          "Test NOT LOADED", flush=True)
    print(f"Outputs: {output_dir}", flush=True)
    for variant in VARIANTS:
        model.load_state_dict(saved["model"] if variant == "online" else saved["ema"], strict=True)
        if variant == "ema_bn_reestimated":
            report["bn_reestimation"] = reestimate_bn(model, train_loader, device)
        report["variants"][variant] = evaluate_variant(model, val_loader, device, output_dir, variant)
    write_summary(output_dir, report)
    print(f"COMPLETE: {output_dir / 'summary.md'}; no checkpoint written; Test NOT LOADED", flush=True)


if __name__ == "__main__":
    main()
