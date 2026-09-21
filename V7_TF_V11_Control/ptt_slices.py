"""Read the existing PTT slice split and a fixed subset of v11 noise WAVs.

Only metadata and filenames are read here; waveform processing stays in data.py.
Folder IDs are 1=Tanker, 2=PassengerShip, 3=Tug. No Cargo relabeling.
"""
import csv
import math
from pathlib import Path
import random


PTT_CLASSES = ("Tanker", "PassengerShip", "Tug")


def _recording_map(mapping_dir, folder, class_name):
    table = Path(mapping_dir) / folder / "segment_mapping.txt"
    with table.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = {name.strip() for name in reader.fieldnames or []}
        required = {"Generated_File_Name", "Parent_Folder", "Original_File_Name"}
        if not required.issubset(fields):
            raise ValueError(f"Unsupported source mapping: {table}")
        result = {}
        for original in reader:
            row = {key.strip(): (value or "").strip() for key, value in original.items()
                   if key is not None}
            name = row["Generated_File_Name"]
            if not name or name in result:
                raise ValueError(f"Missing/duplicate segment name in {table}: {name}")
            parent = row["Parent_Folder"].replace("\\", "/")
            stem = row["Original_File_Name"]
            if stem.lower().endswith(".wav"):
                stem = stem[:-4]
            if not parent or not stem:
                raise ValueError(f"Missing recording identity in {table}: {name}")
            # Recordings in different original class folders are distinct files.
            result[name] = f"{class_name}/{parent}/{stem}".casefold()
    return result


def _noise_paths(data_dir, split):
    directory = Path(data_dir).expanduser().resolve() / "noise" / split
    table = directory / "all_info.txt"
    paths = []
    with table.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        labels = ("label_Tanker", "label_Cargo", "label_Tug")
        required = {"file_name", *labels}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Unsupported noise metadata: {table}")
        for row in reader:
            if row.get("split", split) != split or any(float(row[key]) != 0 for key in labels):
                raise ValueError(f"Non-noise label or wrong split in {table}")
            name = row["file_name"]
            if not name or Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError("Noise file_name must be a WAV basename")
            path = directory / "mix" / name
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
    if len(paths) != len(set(paths)):
        raise ValueError(f"Duplicate noise paths: {table}")
    return paths


def load_ptt_records(data_dir, split, phase, cfg, scene_type):
    if split not in ("Train", "Val", "Test") or phase != "single":
        raise ValueError("PTT slices support single-target Train/Val or explicit Test only")
    if tuple(cfg.class_names) != PTT_CLASSES:
        raise ValueError("PTT class order must be Tanker/PassengerShip/Tug")
    if not cfg.noise_data_dir:
        raise ValueError("Set --noise-data-dir to retain a noise class")
    fraction = cfg.ptt_noise_fraction
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Invalid PTT noise fraction")
    root = Path(data_dir).expanduser().resolve()
    mapping_dir = Path(cfg.mapping_dir) if cfg.mapping_dir else root.parent / "deepship_16k_5s"
    ships = []
    for index, (folder, class_name) in enumerate(zip(("1", "2", "3"), PTT_CLASSES)):
        mapping = _recording_map(mapping_dir, folder, class_name)
        directory = root / split / folder
        files = sorted(directory.glob("*.wav"))
        if not files:
            raise FileNotFoundError(f"No PTT WAVs: {directory}")
        labels = tuple(int(i == index) for i in range(3))
        for path in files:
            if path.name not in mapping:
                raise ValueError(f"No original recording mapping for {path}")
            ships.append(scene_type(path, labels, (mapping[path.name],), float("nan")))
    noise_paths = _noise_paths(cfg.noise_data_dir, split)
    n_noise = math.ceil(len(ships) * fraction / (1.0 - fraction))
    if len(noise_paths) < n_noise:
        raise ValueError(f"{split} needs {n_noise} noise WAVs; only {len(noise_paths)} available")
    # Keep one deterministic noise subset across epochs and report exports.
    split_seed = {"Train": 0, "Val": 1, "Test": 2}[split]
    indices = sorted(random.Random(cfg.seed + split_seed).sample(range(len(noise_paths)), n_noise))
    noise = [scene_type(noise_paths[i], (0, 0, 0), (), float("nan")) for i in indices]
    return noise + ships
