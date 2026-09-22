"""Paths and completion metadata for this one V7 baseline."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model_copy"
DATASET_WINDOWS = "D:/LSJ/Data/data_gen_v7_single/20260920_163131_193/dataset"

def local_path(value):
    value = str(value).replace("\\", "/")
    if os.name != "nt" and len(value) > 2 and value[1:3] == ":/":
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    return Path(value).resolve()

DATASET = local_path(DATASET_WINDOWS)
OUTPUT = HERE / "output"
RUN_DIR = OUTPUT / "run"
REPORT_DIR = OUTPUT / "report"

def bootstrap():
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(MODEL_DIR)
    sys.path.insert(0, str(MODEL_DIR))

def validate_dataset():
    summary = json.loads((DATASET.parent / "summary.json").read_text(encoding="utf-8-sig"))
    if (summary.get("status") != "complete"
            or summary.get("revision") != "data_gen_v7_single_bellhop_expanded"
            or summary.get("ship_audio_bellhop") is not True
            or summary.get("failure")):
        raise ValueError("This entry requires the completed V7 Bellhop dataset.")
    if local_path(summary["dataset_dir"]) != DATASET or not DATASET.is_dir():
        raise ValueError("The generated dataset path does not match this V7 baseline.")
    return summary

BASELINE_SUMMARY = HERE / "baseline_copy/output/summary.json"
CHANNEL_BANK = OUTPUT / "channel_bank.mat"
ALTERNATE_ROOT = OUTPUT / "alternate_train"
AUGMENTATION = {
    "name": "v7_two_bellhop_views_v1",
    "split": "Train", "alternate_probability": 0.5,
    "channel_assignment_seed": 17042,
    "views_per_source": 2,
    "alternate_geometry": "uniform among the other 38 original range-depth combinations",
    "source": "original normalized 8-second input context, NOT propagated WAV",
    "evaluation": "unchanged original Bellhop Train and Val; no augmentation",
    "test_loaded": False,
}
