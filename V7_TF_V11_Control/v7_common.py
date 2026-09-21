"""Fixed paths and protocol for the copied M2_TF_V11_CR0_CE model."""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
RUN_DIR = OUTPUT / "run"
REPORT_DIR = OUTPUT / "report"
BASELINE_SUMMARY = HERE / "reference/channel_baseline_summary.json"
BN_CONTROL_SUMMARY = HERE / "reference/ema_bn_control_summary.json"
TF_STRUCTURE = json.loads((HERE / "reference/tf_architecture.json").read_text())
TF_V6_SUMMARY = HERE / "reference/tf_v6_summary.json"


def local_path(value):
    value = str(value).replace("\\", "/")
    if os.name != "nt" and len(value) > 2 and value[1:3] == ":/":
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    return Path(value).resolve()


DATASET = local_path("D:/LSJ/Data/data_gen_v7_single/20260920_163131_193/dataset")
ALTERNATE_ROOT = HERE.parent / "V7_Bellhop_Channel_Aug/output/alternate_train"
CHANNEL_BANK = HERE.parent / "V7_Bellhop_Channel_Aug/output/channel_bank.mat"
AUGMENTATION = {
    "name": "v7_two_bellhop_views_v1", "split": "Train", "alternate_probability": 0.5,
    "channel_assignment_seed": 17042, "views_per_source": 2,
    "alternate_geometry": "uniform among the other 38 original range-depth combinations",
    "source": "original normalized 8-second input context, NOT propagated WAV",
    "evaluation": "unchanged original Bellhop Train and Val; no augmentation", "test_loaded": False,
}
BN_POLICY = json.loads((HERE / "reference/bn_policy.json").read_text())
TF_PROTOCOL = {
    "name": "m2_tf_v11_single_v7_s42_bncal_v1",
    "source": "M2_TF_V11_CR0_CE/model.py + backbone.py + m2_components.py",
    "source_configuration": "M2_TF_V11_CR0_CE/config.py architecture defaults; copied unmodified model files",
    "model_structure": TF_STRUCTURE,
    "head": "original V11 192-D Query + 128-D DEMON; original shared evidence scorer",
    "single_adaptation": "count/pair heads retained and frozen; bypassed in single forward",
    "initialization": "fresh original TF initialization, seed42; no historical weights",
    "new_training_runs": 1, "epochs": 60, "seed": 42, "batch_size": 32,
    "ships_per_class_per_epoch": 1800, "noise_per_epoch": 1080,
    "optimizer_steps_per_epoch": 203, "optimizer_steps_total": 12180,
    "loss": "BCE + 0.2 ship CE; SupCon/CR/query orthogonality disabled",
    "sampler": "unchanged channel-baseline SourcePoolSampler, not cross-recording batch rearrangement",
    "augmentation": AUGMENTATION, "normalization": "saved V7 Bellhop Train frontend statistics",
    "selection": "highest EMA Val single EMR; independent three-logit 0.5 threshold",
    "bn_policy": BN_POLICY,
    "comparison_scope": "V11 TF + Query versus Mamba + GAP and full V6 TF + Query; not a backbone-only comparison",
    "new_data_generation": False, "test_loaded": False,
}


def bootstrap():
    # All source dependencies are local copies; never import the original model directory.
    if not (HERE / "model.py").is_file():
        raise FileNotFoundError("The copied V11 TF model files are required")


def validate_dataset():
    summary = json.loads((DATASET.parent / "summary.json").read_text(encoding="utf-8-sig"))
    if (summary.get("status") != "complete"
            or summary.get("revision") != "data_gen_v7_single_bellhop_expanded"
            or summary.get("ship_audio_bellhop") is not True or summary.get("failure")):
        raise ValueError("This experiment requires the completed current V7 Bellhop dataset")
    if local_path(summary["dataset_dir"]) != DATASET or not DATASET.is_dir():
        raise ValueError("V7 dataset path mismatch")
    return summary
