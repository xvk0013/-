"""Fixed 5:1 ship/noise exposure, with an identical recording schedule in A/B."""
import hashlib
import random
from record_balanced_sampler import RecordingBalancedSampler as BaseSampler

class SourcePoolSampler(BaseSampler):
    def __init__(self, records, seed):
        super().__init__(records, seed)
        if set(self.groups) != {0, 1, 2} or len(self.noise_indices) < 1080:
            raise ValueError("Requires three classes and at least 1080 noise clips")
        self.class_counts = {i: 1800 for i in range(3)}
        self.num_samples = 6480
        self.seen = set()
        self.coverage = {}

    def __iter__(self):
        schedule = random.Random(self.seed + self.epoch)
        indices = schedule.sample(self.noise_indices, 1080)
        ships = []
        for label in sorted(self.groups):
            groups = self.groups[label]
            recordings = sorted(groups)
            schedule.shuffle(recordings)
            quota, extra = divmod(1800, len(recordings))
            for position, recording in enumerate(recordings):
                # Candidate-pool size never changes recording/noise/order RNG.
                token = f"{self.seed}|{self.epoch}|{label}|{recording}".encode()
                rng = random.Random(int.from_bytes(hashlib.sha256(token).digest()[:8], "big"))
                remaining = quota + int(position < extra)
                pool = groups[recording]
                while remaining:
                    take = min(remaining, len(pool))
                    ships.extend(rng.sample(pool, take))
                    remaining -= take
        indices.extend(ships)
        schedule.shuffle(indices)
        self.seen.update(ships)
        self.coverage = {"unique_ship_clips_this_epoch": len(set(ships)),
                         "unique_ship_clips_cumulative": len(self.seen),
                         "epoch_ship_draws": 5400, "epoch_noise_draws": 1080}
        return iter(indices)
