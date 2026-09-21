"""Adapter for data_gen_v6_lazy_pairing_v11; no data is required at import time."""
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from frontend import DualBandTransform, DEMONTransform, SpecAugment


CLASS_NAMES = ("Tanker", "Cargo", "Tug")
FOLDER_LABELS = {
    "noise": (0, 0, 0), "0": (0, 1, 0), "1": (1, 0, 0), "2": (0, 0, 1),
    "0_1": (1, 1, 0), "0_2": (0, 1, 1), "1_2": (1, 0, 1),
}


@dataclass(frozen=True)
class Scene:
    path: Path
    labels: tuple
    recordings: tuple
    sir_db: float

    @property
    def single_recording(self):
        return self.recordings[0] if sum(self.labels) == 1 else ""


def parse_label(value):
    if str(value).lower() in ("true", "false"):
        return int(str(value).lower() == "true")
    number = float(value)
    if number not in (0, 1):
        raise ValueError(f"Expected binary label, got {value!r}")
    return int(number)


def load_records(data_dir, split, phase, cfg=None):
    # Calls without a config retain the historical v11/synthetic interface.
    if cfg is not None and cfg.dataset_format == "deepship_ptt_slices":
        from ptt_slices import load_ptt_records
        return load_ptt_records(data_dir, split, phase, cfg, Scene)
    if cfg is not None and cfg.dataset_format == "deepship_v6_slices":
        from v6_slices import load_v6_slice_records
        return load_v6_slice_records(data_dir, split, phase, cfg, Scene)
    return _load_v11_records(data_dir, split, phase)


def _load_v11_records(data_dir, split, phase):
    root = Path(data_dir).expanduser().resolve()
    if split not in ("Train", "Val", "Test") or phase not in ("single", "joint"):
        raise ValueError("Invalid split/phase")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset is not ready: {root}. Set --data-dir after generation completes.")
    records = []
    for folder, expected in FOLDER_LABELS.items():
        if phase == "single" and sum(expected) > 1:
            continue
        directory = root / folder / split
        table = directory / "all_info.txt"
        if not table.is_file():
            raise FileNotFoundError(f"Missing metadata: {table}; dataset may still be generating.")
        with table.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"file_name", "label_Tanker", "label_Cargo", "label_Tug",
                        "recording1", "recording2"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Unsupported all_info.txt fields: {table}")
            n_rows = 0
            for row in reader:
                labels = tuple(parse_label(row[f"label_{name}"]) for name in CLASS_NAMES)
                if labels != expected:
                    raise ValueError(f"Folder/label mismatch in {table}: {row['file_name']}")
                if row.get("split", split) != split:
                    raise ValueError(f"Split mismatch in {table}")
                name = row["file_name"]
                if Path(name).name != name or "/" in name or "\\" in name:
                    raise ValueError("file_name must be a WAV basename")
                path = directory / "mix" / name
                if not path.is_file():
                    raise FileNotFoundError(f"Missing mixture WAV: {path}")
                recs = []
                for i in range(1, sum(labels) + 1):
                    rec = row[f"recording{i}"].strip().replace("\\", "/").casefold()
                    if rec in ("", "none", "nan"):
                        raise ValueError(f"Missing source recording for {path}")
                    recs.append(rec)
                sir = float(row.get("sir_db") or "nan")
                records.append(Scene(path, labels, tuple(recs), sir))
                n_rows += 1
            if not n_rows:
                raise ValueError(f"Empty metadata: {table}")
    paths = [s.path for s in records]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate mixture paths in metadata")
    return records


def check_recording_split(train, val):
    train_ids = {r for scene in train for r in scene.recordings}
    val_ids = {r for scene in val for r in scene.recordings}
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(f"Train/Val share {len(overlap)} source recordings")


def read_waveform(path, cfg):
    wave, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != cfg.sample_rate or wave.shape != (cfg.audio_len,):
        raise ValueError(f"Expected mono 16 kHz / 5 seconds, got {sr}, {wave.shape}: {path}")
    if not np.isfinite(wave).all():
        raise ValueError(f"Non-finite waveform: {path}")
    # Float WAV can exceed +/-1. Never clip or convert it to PCM16.
    rms = float(np.sqrt(np.mean(np.square(wave, dtype=np.float64))))
    if rms == 0:
        raise ValueError(f"Silent mixture WAV: {path}")
    return torch.from_numpy((wave.astype(np.float64) / rms).astype(np.float32))


def frontend_signature(cfg):
    names = ("sample_rate", "audio_len", "lofar_n_fft", "lofar_hop", "lofar_win",
             "lofar_n_bins", "wideband_n_fft", "wideband_hop", "wideband_freq_lo_hz",
             "wideband_freq_max_hz", "normalization_mode")
    signature = {name: getattr(cfg, name) for name in names}
    # Omit when disabled so saved two-channel statistics remain compatible.
    if cfg.local_contrast_bins:
        signature["local_contrast_bins"] = cfg.local_contrast_bins
    return signature


def fit_train_stats(records, cfg):
    """Deterministic Train-only subsample. Never reads Val/Test or old caches."""
    candidates = list(range(len(records)))
    random.Random(cfg.seed).shuffle(candidates)
    candidates = candidates[:min(cfg.stats_samples, len(candidates))]
    if not candidates:
        raise ValueError("No Train samples available for normalization")
    transform = DualBandTransform(cfg)
    mean = torch.zeros(cfg.input_channels, dtype=torch.float64)
    m2 = torch.zeros_like(mean)
    count = 0
    for i, index in enumerate(candidates, 1):
        x = transform(read_waveform(records[index].path, cfg)).double().flatten(1)
        n = x.shape[1]
        batch_mean = x.mean(1)
        delta = batch_mean - mean
        m2 += ((x - batch_mean[:, None]) ** 2).sum(1) + delta.square() * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
        if i % 100 == 0:
            print(f"Train normalization: {i}/{len(candidates)}", flush=True)
    std = (m2 / count).sqrt()
    if not bool(torch.isfinite(std).all()) or bool((std <= 0).any()):
        raise ValueError("Invalid Train feature statistics")
    return {"mean": mean.tolist(), "std": std.tolist(), "n_samples": len(candidates),
            "split": "Train", "frontend": frontend_signature(cfg)}


class SceneDataset(Dataset):
    def __init__(self, records, cfg, stats, training=False):
        if stats.get("frontend") != frontend_signature(cfg) or stats.get("split") != "Train":
            raise ValueError("Normalization statistics do not match this frontend/Train split")
        self.records, self.cfg = records, cfg
        self.spectral_enabled = training and cfg.spectral_aug_prob > 0
        self.spectral_rng = None
        self.feature = DualBandTransform(cfg, stats["mean"], stats["std"])
        self.demon = DEMONTransform(cfg)
        self.augment = SpecAugment(n_freq_masks=cfg.freq_masks, n_time_masks=0,
                                   freq_mask_param=cfg.freq_mask_width) if training else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        scene = self.records[index]
        wave = read_waveform(scene.path, self.cfg)
        if self.spectral_enabled:
            from spectral_augment import smooth_frequency_response
            if self.spectral_rng is None:
                # Separate stream: do not consume SpecAugment or model RNG draws.
                self.spectral_rng = np.random.default_rng(torch.initial_seed() ^ 0x53504543)
            if self.spectral_rng.random() < self.cfg.spectral_aug_prob:
                wave = torch.from_numpy(smooth_frequency_response(
                    wave.numpy(), self.cfg.sample_rate, self.cfg.spectral_aug_db, self.spectral_rng))
        features, demon = self.feature(wave), self.demon(wave)
        if self.augment is not None:
            features = self.augment(features)
        if not bool(torch.isfinite(features).all() and torch.isfinite(demon).all()):
            raise FloatingPointError(f"Non-finite acoustic features: {scene.path}")
        return {"features": features, "demon": demon,
                "labels": torch.tensor(scene.labels, dtype=torch.float32),
                "recording": scene.single_recording, "sir_db": scene.sir_db,
                "path": str(scene.path)}
