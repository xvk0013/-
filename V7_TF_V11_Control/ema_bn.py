"""Match EMA BatchNorm statistics to EMA weights using fixed Train-only data."""
import random
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from data import SceneDataset
from bn_reestimate import reestimate_bn

EMA_BN_POLICY = {
    "name": "ema_train_only_bn_v1", "split": "Train",
    "ships_per_class": 128, "noise": 77, "samples": 461,
    "selection_seed": 27043, "batch_size": 32,
    "source": "original Train waves, saved frontend statistics",
    "augmentation": False, "gradient_updates": False,
    "when": "before every EMA Val evaluation; saved with EMA checkpoint",
}


def select_calibration_records(records):
    rng = random.Random(EMA_BN_POLICY["selection_seed"])
    selected = []
    for label in range(3):
        by_recording = defaultdict(list)
        for scene in records:
            if scene.labels[label]:
                by_recording[scene.single_recording].append(scene)
        groups = list(by_recording.values())
        rng.shuffle(groups)
        for group in groups:
            rng.shuffle(group)
        for k in range(EMA_BN_POLICY["ships_per_class"]):
            available = [group for group in groups if group]
            if not available:
                raise ValueError("Not enough Train clips for fixed BN calibration")
            selected.append(available[k % len(available)].pop())
    noise = [scene for scene in records if not sum(scene.labels)]
    selected.extend(rng.sample(noise, EMA_BN_POLICY["noise"]))
    rng.shuffle(selected)
    return selected


def make_calibration_batches(train_records, cfg, stats):
    dataset = SceneDataset(select_calibration_records(train_records), cfg, stats, training=False)
    # Cache only these fixed features in host RAM; never touch the training loader RNG.
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=min(2, cfg.workers),
                        generator=torch.Generator().manual_seed(27044))
    batches = list(loader)
    print(f"Cached Train-only EMA BN calibration: {len(dataset)} clips, {len(batches)} forward batches", flush=True)
    return batches


def calibrate_ema(model, batches, device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        result = reestimate_bn(model, batches, device)
    result["policy"] = EMA_BN_POLICY
    return result

