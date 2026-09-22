"""Class-count-preserving recording-balanced indices for single-phase Train.

Implements the DataLoader sampler protocol without loading audio or importing
model code. Epoch-local RNG makes the sampling order reproducible on resume.
"""
import random


class RecordingBalancedSampler:
    def __init__(self, records, seed):
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = len(records)
        self.noise_indices = []
        self.groups = {}
        self.class_counts = {}
        if not self.num_samples:
            raise ValueError("Recording-balanced sampling requires nonempty records")
        for index, scene in enumerate(records):
            labels = tuple(scene.labels)
            if len(labels) != 3 or any(value not in (0, 1) for value in labels):
                raise ValueError("Expected three binary labels")
            count = sum(labels)
            if count == 0:
                self.noise_indices.append(index)
                continue
            if count != 1:
                raise ValueError("Recording-balanced sampler is for single-target Train only")
            recording = scene.single_recording
            if not recording:
                raise ValueError("Single-target scene has no source recording")
            label = labels.index(1)
            self.groups.setdefault(label, {}).setdefault(recording, []).append(index)
            self.class_counts[label] = self.class_counts.get(label, 0) + 1

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        if int(epoch) != epoch or epoch < 0:
            raise ValueError("Epoch must be a nonnegative integer")
        self.epoch = int(epoch)

    def __iter__(self):
        # Independent of global augmentation RNG and DataLoader worker RNG.
        rng = random.Random(self.seed + self.epoch)
        indices = list(self.noise_indices)  # Every noise scene exactly once.
        for label in sorted(self.groups):
            groups = self.groups[label]
            recordings = sorted(groups)
            rng.shuffle(recordings)  # Randomize which recordings get the remainder.
            quota, extra = divmod(self.class_counts[label], len(recordings))
            for position, recording in enumerate(recordings):
                remaining = quota + int(position < extra)
                pool = groups[recording]
                # Sample without replacement within each cycle. Repeat only when
                # the recording has fewer scenes than its assigned epoch quota.
                while remaining:
                    take = min(remaining, len(pool))
                    indices.extend(rng.sample(pool, take))
                    remaining -= take
        # Preserve the original per-epoch class/noise totals, then mix all classes.
        rng.shuffle(indices)
        return iter(indices)
