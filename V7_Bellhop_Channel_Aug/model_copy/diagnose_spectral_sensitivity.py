"""Fixed paired spectral sensitivity diagnosis. Val only; no training."""
import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import scipy.signal
import torch
from data import SceneDataset, read_waveform, load_records
from spectral_augment import smooth_frequency_response
from config import Config
from model import M2Mamba
from train import load_checkpoint, make_loader, seed_all
from diagnose_ema_bn import evaluate_variant
from diagnose_ranking_threshold import CLASSES, classify, summarize, write_csv


class FixedResponseDataset(SceneDataset):
    def __init__(self, records, cfg, stats, response_seed):
        super().__init__(records, cfg, stats, training=False)
        self.response_seed = response_seed

    def __getitem__(self, index):
        scene = self.records[index]
        wave = read_waveform(scene.path, self.cfg)
        if self.response_seed is not None:
            # Restart the generator: exactly the same response for every waveform/model.
            wave = torch.from_numpy(smooth_frequency_response(
                wave.numpy(), self.cfg.sample_rate, 3.0,
                np.random.default_rng(self.response_seed)))
        features, demon = self.feature(wave), self.demon(wave)
        if not bool(torch.isfinite(features).all() and torch.isfinite(demon).all()):
            raise FloatingPointError("Non-finite features")
        return {"features": features, "demon": demon,
                "labels": torch.tensor(scene.labels, dtype=torch.float32),
                "recording": scene.single_recording, "sir_db": scene.sir_db,
                "path": str(scene.path)}


def unique_records(records, data_dir):
    mapping = {}
    for name in CLASSES:
        with (data_dir/name/"Val"/"all_info.txt").open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                start = row["start_sample"] if "start_sample" in row else row["segment_index"]
                mapping[(name, row["file_name"])] = (name, row["raw_relative_path"].casefold(), start)
    unique = {}
    for scene in records:
        if sum(scene.labels) == 0:
            key = ("noise", str(scene.path))
        else:
            name = CLASSES[scene.labels.index(1)]
            key = mapping[(name, scene.path.name)]
        unique.setdefault(key, scene)
    return list(unique.values())


def paired_stats(clean, changed):
    if len(clean) != len(changed) or any(a["path"] != b["path"] for a,b in zip(clean,changed)):
        raise ValueError("Paired prediction identities differ")
    result = {}
    for name in ("all_ships",) + CLASSES + ("noise",):
        pairs = [(a,b) for a,b in zip(clean,changed)
                 if (a["truth"] != "noise" if name == "all_ships" else a["truth"] == name)]
        n = len(pairs)
        result[name] = {"n": n,
            "prediction_flip_rate": sum(a["prediction"] != b["prediction"] for a,b in pairs)/n,
            "correct_to_wrong": sum(int(a["exact_match"]) and not int(b["exact_match"]) for a,b in pairs),
            "wrong_to_correct": sum(not int(a["exact_match"]) and int(b["exact_match"]) for a,b in pairs),
            "prediction_transitions": dict(Counter(a["prediction"]+" -> "+b["prediction"]
                for a,b in pairs if a["prediction"] != b["prediction"]))}
        if name != "noise":
            result[name]["top1_flip_rate"] = sum(a["top_class"] != b["top_class"] for a,b in pairs)/n
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--augmented", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = args.output_dir/("diagnostic_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True)
    device = torch.device(args.device)
    # Predeclared diagnostic probes; never search/select responses by Val performance.
    conditions = {"clean": None, "response_101": 101, "response_202": 202, "response_303": 303}
    gains = {name: np.random.default_rng(seed).uniform(-3,3,8).tolist()
             for name,seed in conditions.items() if seed is not None}
    result = {"split": "Val", "test_loaded": False, "variant": "ema",
        "precision": "float32", "threshold": 0.5, "rms_preserved": True,
        "anchors_hz": [0,125,250,500,1000,2000,4000,8000], "anchor_gains_db": gains,
        "note": "Unique ship sources plus noise. Fixed probes, not new independent samples. Robustness does not prove recording-identity reliance.",
        "models": {}}
    reference_paths = None
    for tag, path in (("baseline",args.baseline),("augmented",args.augmented)):
        saved = load_checkpoint(path)
        cfg = Config(**saved["config"])
        cfg.workers = args.workers
        cfg.validate()
        if cfg.head_type != "gap" or saved["phase"] != "single" or cfg.sample_rate != 16000:
            raise ValueError("Requires single GAP 16 kHz checkpoint")
        if Path(saved["data_dir"]).resolve() != args.data_dir.resolve():
            raise ValueError("Checkpoint dataset mismatch")
        if cfg.scan_backend == "cuda":
            from scan import cuda_scan
            cuda_scan()
        seed_all(cfg.seed)
        records = unique_records(load_records(args.data_dir,"Val","single",cfg),args.data_dir)
        paths = [str(r.path) for r in records]
        if reference_paths is not None and paths != reference_paths:
            raise ValueError("Model input identities differ")
        reference_paths = paths
        model = M2Mamba(cfg,"single").to(device).eval().requires_grad_(False)
        model.load_state_dict(saved["ema"],strict=True)
        buffers = {n:b.clone() for n,b in model.named_buffers()}
        entry = {"checkpoint":str(path.resolve()),"epoch":saved["epoch"],
                 "config":cfg.to_dict(),"conditions":{}}
        clean = None
        for name, response_seed in conditions.items():
            folder = out/tag/name
            folder.mkdir(parents=True)
            dataset = FixedResponseDataset(records,cfg,saved["stats"],response_seed)
            loader = make_loader(dataset,cfg,False,device)
            print(f"{tag}: {name}; n={len(records)}",flush=True)
            with torch.inference_mode():
                evaluate_variant(model,loader,device,folder,"ema",split_name="Val")
            with (folder/"predictions_ema.csv").open(encoding="utf-8-sig",newline="") as f:
                rows = [classify(r) for r in csv.DictReader(f)]
            if clean is None:
                clean = rows
            entry["conditions"][name] = {"metrics":summarize(rows),"paired_with_clean":paired_stats(clean,rows)}
        if any(not torch.equal(b,buffers[n]) for n,b in model.named_buffers()):
            raise RuntimeError("Inference changed model buffers")
        result["models"][tag] = entry
        del model
    (out/"sensitivity_summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    compact=[]
    for tag, entry in result["models"].items():
        for condition, values in entry["conditions"].items():
            for name, metrics in values["metrics"].items():
                paired=values["paired_with_clean"][name]
                compact.append({"model":tag,"condition":condition,"class":name,
                    "n":metrics["n"],"exact_accuracy":metrics["formal_exact_accuracy"],
                    "prediction_flip_rate":paired["prediction_flip_rate"],
                    "correct_to_wrong":paired["correct_to_wrong"],"wrong_to_correct":paired["wrong_to_correct"]})
    write_csv(out/"comparison.csv",compact)
    print(f"COMPLETE: {out}",flush=True)


if __name__ == "__main__":
    main()
