"""Train-only source identities, fixed alternate channels and original convolution."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.fft import rfft, irfft
from v7_common import DATASET, CHANNEL_BANK, AUGMENTATION, local_path

CLASSES = ("Tanker", "Cargo", "Tug")

def train_rows(root=DATASET, classes=CLASSES):
    rows = {}
    for name in classes:
        with (Path(root)/name/"Train/all_info.txt").open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                key = row["audio_path"].replace("\\", "/")
                if (row["split"] != "Train" or row["class_name"] != name
                        or key != f"{name}/Train/{row['file_name']}" or key in rows):
                    raise ValueError("Invalid/duplicate Train ship source")
                if int(row["stop_sample"]) - int(row["start_sample"]) + 1 != 80000:
                    raise ValueError("Expected complete five-second core")
                rows[key] = row
    source_ids = [int(r["source1_index"]) for r in rows.values()]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Expected one original Bellhop scene per Train source")
    return rows

def source_identity(row):
    return [row["class_name"], row["raw_relative_path"].replace("\\", "/").casefold(),
            int(row["start_sample"]), int(row["stop_sample"]), int(row["source1_index"])]

def alternate_index(original, source_index, n_channels=39):
    if n_channels < 2 or not 0 <= original < n_channels:
        raise ValueError("Need an original and at least one distinct alternate channel")
    rng = np.random.default_rng(np.random.SeedSequence(
        [AUGMENTATION["channel_assignment_seed"], int(source_index)]))
    chosen = int(rng.integers(n_channels-1))
    return chosen + int(chosen >= original)

def load_bank(path=CHANNEL_BANK):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Run PREPARE_CHANNEL_BANK.m in MATLAB first: {path}")
    b = loadmat(path, squeeze_me=True)
    for key in ("nfft", "fs", "history_samples", "core_samples", "memory_samples", "schema_version"):
        b[key] = int(b[key])
    if (b["schema_version"] != 1 or b["fs"] != 16000 or b["history_samples"] != 48000
            or b["core_samples"] != 80000 or b["memory_samples"] != 48000
            or b["nfft"] < 176000 or float(b["output_gain"]) != 1
            or b["H"].shape != (b["nfft"]//2+1, 39)
            or not np.isfinite(b["H"]).all()):
        raise ValueError("Channel bank does not match the original V7 processing chain")
    if local_path(str(b["generation_root"])) != DATASET.parent:
        raise ValueError("Channel bank belongs to a different generation")
    expected = {(r, d) for r in np.arange(1, 4.01, .25) for d in (5.,10.,15.)}
    if set(zip(b["ranges_km"],b["depths_m"])) != expected:
        raise ValueError("Unexpected physical channel grid")
    return b

def propagate(x, H, nfft, history_samples, core_samples, memory_samples, gain=1.0):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if (len(x) != history_samples+core_samples or history_samples < memory_samples
            or nfft < len(x)+memory_samples or not np.isfinite(x).all()):
        raise ValueError("Invalid source context or insufficient linear convolution length")
    # Same real causal filter and core window as propagate_sources.m.
    spectrum = rfft(x, n=nfft)*H
    spectrum[0] = spectrum[0].real
    if nfft % 2 == 0:
        spectrum[-1] = spectrum[-1].real
    wave = irfft(spectrum, n=nfft)[history_samples:history_samples+core_samples]*gain
    if not np.isfinite(wave).all() or not np.any(wave):
        raise ValueError("Invalid propagated waveform; no sample is silently discarded")
    return wave.astype(np.float32)

def source_context(row, generation_root=DATASET.parent):
    relative = Path(row["input_path"].replace("\\", "/"))
    if relative.parts[:2] != ("inputs","Train") or ".." in relative.parts or relative.is_absolute():
        raise ValueError("Only original Train source contexts are allowed")
    obj = loadmat(Path(generation_root)/relative, squeeze_me=True, variable_names=["x","fs"])
    x = np.asarray(obj["x"],dtype=np.float64).reshape(-1)
    if int(obj["fs"]) != 16000 or x.shape != (128000,):
        raise ValueError("Expected normalized 3+5 second source context")
    rms = np.sqrt(np.mean(x[48000:]**2))
    if not np.isfinite(rms) or abs(rms-1.) > 1e-5:
        raise ValueError("Original core normalization is not RMS=1")
    return x

def make_entries(rows, bank):
    entries = []
    for key,row in sorted(rows.items(),key=lambda item:int(item[1]["source1_index"])):
        candidates = np.flatnonzero((bank["ranges_km"] == float(row["range1_km"]))
                                   & (bank["depths_m"] == float(row["source_depth1_m"])))
        if len(candidates) != 1:
            raise ValueError("Original propagation coordinates missing from bank")
        old = int(candidates[0]); new = alternate_index(old,int(row["source1_index"]))
        if float(row["common_scale"]) != float(bank["output_gain"]):
            raise ValueError("Output gain differs from the original generation")
        filename = (f"src_{int(row['source1_index']):05d}_"
                    f"r{int(bank['range_indices'][new]):02d}_d{int(bank['depth_indices'][new]):02d}.wav")
        entries.append(dict(key=key,source_identity=source_identity(row),
            alternate_path=f"{row['class_name']}/Train/{filename}",
            original_channel_index=old,alternate_channel_index=new,
            original_range_km=float(bank["ranges_km"][old]),
            original_depth_m=float(bank["depths_m"][old]),
            alternate_range_km=float(bank["ranges_km"][new]),
            alternate_depth_m=float(bank["depths_m"][new]),
            history_padding_samples=int(row["history_padding_samples"])))
    return entries

def write_json(path, obj):
    path=Path(path);tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    tmp.replace(path)
