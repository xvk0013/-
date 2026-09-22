"""Adapter for RUN_DEEPSHIP_CUT.m named-class 5-second FLOAT WAVs.

Read only selected classes and the requested split. This adapter does not
propagate, denoise, resample, relabel PassengerShip, or regenerate audio.
"""
import csv
from pathlib import Path
from ptt_slices import _noise_paths


CLASS_FOLDERS = {"Tanker": ("Tanker", 1), "Cargo": ("Cargo", 0),
                 "PassengerShip": ("Passengership", 2), "Tug": ("Tug", 3)}


def load_v6_slice_records(data_dir, split, phase, cfg, scene_type):
    if split not in ("Train", "Val", "Test") or phase != "single":
        raise ValueError("Named-class slices support single-target splits only")
    if not cfg.noise_data_dir:
        raise ValueError("--noise-data-dir is required")
    if len(cfg.class_names) != 3 or len(set(cfg.class_names)) != 3:
        raise ValueError("The current M2 head requires three distinct ship classes")
    root = Path(data_dir).expanduser().resolve()
    ships = []
    for index, class_name in enumerate(cfg.class_names):
        folder, class_id = CLASS_FOLDERS[class_name]
        directory = root / folder / split
        table = directory / "all_info.txt"
        n_rows = 0
        with table.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"file_name", "audio_path", "class_id", "class_name", "split",
                        "raw_relative_path", "fs", "duration_s", "QC_status"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Unsupported named-class metadata: {table}")
            for row in reader:
                if (row["split"] != split or row["class_name"] != folder
                        or int(row["class_id"]) != class_id):
                    raise ValueError(f"Class/split mismatch in {table}")
                if row["QC_status"] not in ("PASS", "REVIEW"):
                    raise ValueError(f"Unsaved/rejected clip appears in {table}")
                if (int(row["fs"]) != cfg.sample_rate
                        or float(row["duration_s"]) * cfg.sample_rate != cfg.audio_len):
                    raise ValueError(f"Expected mono 16 kHz / 5 s metadata: {table}")
                name = row["file_name"]
                if not name or Path(name).name != name or "/" in name or "\\" in name:
                    raise ValueError("file_name must be a WAV basename")
                if row["audio_path"].replace("\\", "/") != f"{folder}/{split}/{name}":
                    raise ValueError(f"Audio path does not match class/split: {table}")
                path = directory / name
                if not path.is_file():
                    raise FileNotFoundError(path)
                recording = row["raw_relative_path"].strip().replace("\\", "/")
                parts = recording.split("/")
                if (len(parts) < 2 or parts[0] != folder
                        or any(part in ("", ".", "..") for part in parts)
                        or not recording.lower().endswith(".wav")):
                    raise ValueError(f"Invalid original recording identity: {table}")
                labels = tuple(int(i == index) for i in range(len(cfg.class_names)))
                ships.append(scene_type(path, labels, (recording.casefold(),), float("nan")))
                n_rows += 1
        if n_rows == 0:
            raise ValueError(f"No saved clips: {table}")
    # New ship pool is much larger than the existing noise pool. Use every
    # same-split noise WAV once; do not duplicate noise to force a percentage.
    noise = [scene_type(path, (0, 0, 0), (), float("nan"))
             for path in _noise_paths(cfg.noise_data_dir, split)]
    if not noise:
        raise ValueError(f"No {split} noise samples")
    records = noise + ships
    if len({scene.path for scene in records}) != len(records):
        raise ValueError("Duplicate input paths in named-class dataset")
    return records
