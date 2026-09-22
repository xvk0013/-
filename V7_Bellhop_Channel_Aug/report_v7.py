"""Evaluate saved GAP EMA on full Train and Val without augmentation or updates."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from v7_common import bootstrap
bootstrap()
from diagnose_ranking_threshold import CLASSES, classify, summarize, write_csv

def aggregate(prediction_file, data_dir, split, out_dir):
    mapping = {}
    for name in CLASSES:
        with (data_dir/name/split/"all_info.txt").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                start = row["start_sample"] if "start_sample" in row else row["segment_index"]
                mapping[(name, row["file_name"])] = (name, row["raw_relative_path"].casefold(), start)
    with prediction_file.open(encoding="utf-8-sig", newline="") as handle:
        rows = [classify(row) for row in csv.DictReader(handle)]
    unique = {}
    for row in rows:
        key = ("noise", row["path"]) if row["truth"] == "noise" else mapping[(row["truth"], Path(row["path"]).name)]
        if key in unique:
            if unique[key]["prediction"] != row["prediction"] or unique[key]["top_class"] != row["top_class"]:
                raise ValueError("Repeated source predictions disagree; cannot silently deduplicate")
        else:
            unique[key] = row
    groups = defaultdict(list)
    for key, row in unique.items():
        if row["truth"] != "noise":
            groups[(key[0], key[1])].append(row)
    recording_rows = []
    for (name, recording), group in sorted(groups.items()):
        n = len(group)
        recording_rows.append({
            "class_name": name, "recording": recording, "n_unique_sources": n,
            "exact_accuracy": sum(int(r["exact_match"]) for r in group)/n,
            "ship_only_top1_accuracy": sum(int(r["top1_correct"]) for r in group)/n,
            "predicted_tanker_fraction": sum(r["prediction"] == "Tanker" for r in group)/n,
            "predicted_cargo_fraction": sum(r["prediction"] == "Cargo" for r in group)/n,
            "rejected_fraction": sum(r["prediction"] == "noise" for r in group)/n})
    balanced = {}
    for name in CLASSES:
        group = [r for r in recording_rows if r["class_name"] == name]
        if not group:
            raise ValueError(f"No recordings for {name}")
        balanced[name] = {
            "n_recordings": len(group),
            "exact_accuracy": sum(r["exact_accuracy"] for r in group)/len(group),
            "ship_only_top1_accuracy": sum(r["ship_only_top1_accuracy"] for r in group)/len(group)}
    balanced["class_macro"] = {
        k: sum(balanced[c][k] for c in CLASSES)/len(CLASSES)
        for k in ("exact_accuracy", "ship_only_top1_accuracy")}
    write_csv(out_dir/"per_recording.csv", recording_rows)
    result = {"scene_level": summarize(rows),
              "unique_source_level": summarize(list(unique.values())),
              "recording_equal_weight": balanced}
    (out_dir/"ranking_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # Import the training runtime only for actual inference.
    import scipy.signal
    import torch
    from config import Config
    from data import SceneDataset, load_records, check_recording_split
    from diagnose_ema_bn import evaluate_variant
    from model import M2Mamba
    from train import load_checkpoint, make_loader, seed_all
    saved = load_checkpoint(args.checkpoint)
    from v7_common import AUGMENTATION
    if saved.get("channel_augmentation") != AUGMENTATION:
        parser.error("Checkpoint is not from this channel augmentation experiment")
    cfg = Config(**saved["config"])
    cfg.workers = args.workers
    cfg.validate()
    if saved["phase"] != "single" or cfg.head_type != "gap":
        parser.error("Use the existing single-target GAP best.pt")
    if Path(saved["data_dir"]).resolve() != args.data_dir.resolve():
        parser.error("Checkpoint and dataset paths differ")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("Only cpu/cuda supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    if cfg.scan_backend == "cuda":
        if device.type != "cuda":
            parser.error("Saved CUDA backend requires CUDA")
        from scan import cuda_scan
        cuda_scan()
    seed_all(cfg.seed)
    records = {split: load_records(args.data_dir, split, "single", cfg) for split in ("Train", "Val")}
    check_recording_split(records["Train"], records["Val"])
    model = M2Mamba(cfg, "single").to(device).eval().requires_grad_(False)
    model.load_state_dict(saved["ema"], strict=True)
    before = {name: value.detach().clone() for name, value in model.named_buffers()}
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result = {"checkpoint": str(args.checkpoint.resolve()), "epoch": saved["epoch"],
              "seed": cfg.seed, "config": cfg.to_dict(), "variant": "ema", "threshold": 0.5,
              "precision": "float32", "augmentation": False, "bn_reestimated": False,
              "normalization": "saved Train statistics reused, not refitted",
              "test_loaded": False,
              "note": "Full Train candidate pool evaluated once without sampler. Train is in-sample diagnosis, not generalization. Val is checkpoint-selection data, not independent Test.",
              "train_seen_indices_at_checkpoint": len(saved["source_pool_seen"]) if "source_pool_seen" in saved else None,
              "splits": {}}
    for split in ("Train", "Val"):
        folder = out/split
        folder.mkdir(exist_ok=True)
        dataset = SceneDataset(records[split], cfg, saved["stats"], training=False)
        loader = make_loader(dataset, cfg, False, device)
        print(f"Evaluating {split}: {len(records[split])} records; no augmentation/update", flush=True)
        # Both passes use inference mode, including the Train pass.
        with torch.inference_mode():
            measured = evaluate_variant(model, loader, device, folder, "ema", split_name=split)
        measured.update(aggregate(folder/"predictions_ema.csv", args.data_dir, split, folder))
        result["splits"][split] = measured
    for name, value in model.named_buffers():
        if not torch.equal(value, before[name]):
            raise RuntimeError(f"Inference changed a model buffer: {name}")
    result["val_reproduction_emr_delta"] = (
        result["splits"]["Val"]["metrics"]["emr"] - saved["val"]["emr"])
    gaps = {}
    for name in ("all_ships",)+CLASSES:
        train = result["splits"]["Train"]["unique_source_level"][name]
        val = result["splits"]["Val"]["unique_source_level"][name]
        gaps[name] = {k: train[k]-val[k] for k in
                     ("formal_exact_accuracy", "ship_only_top1_accuracy_diagnostic")}
    result["unique_source_train_minus_val"] = gaps
    (out/"summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Train / Val 泛化诊断", "",
             "固定最佳 EMA，FP32，无增强，不更新模型/BN，不重估标准化，不读取 Test。",
             "Train 为训练池上的拟合诊断，不能作为泛化性能。Val 曾用于选模。", ""]
    for split in ("Train", "Val"):
        lines += [f"## {split}", "",
            "| 类别 | 不同源切片数 | 完全正确率 | 船舶条件 Top-1 |",
            "|---|---:|---:|---:|"]
        for name in ("all_ships",)+CLASSES:
            x=result["splits"][split]["unique_source_level"][name]
            lines.append(f"| {name} | {x['n']} | {x['formal_exact_accuracy']:.2%} | {x['ship_only_top1_accuracy_diagnostic']:.2%} |")
        lines += ["", "| 类别 | 录音等权完全正确率 | 录音等权 Top-1 |", "|---|---:|---:|"]
        for name in CLASSES+("class_macro",):
            x=result["splits"][split]["recording_equal_weight"][name]
            lines.append(f"| {name} | {x['exact_accuracy']:.2%} | {x['ship_only_top1_accuracy']:.2%} |")
        lines += [""]
    (out/"summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"COMPLETE: {out}", flush=True)

if __name__ == "__main__":
    main()
