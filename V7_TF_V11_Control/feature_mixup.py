"""Training-only GAP feature interpolation; no acoustic mixture or new parameters."""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class FeatureMixupPlan:
    rows: torch.Tensor
    partners: torch.Tensor
    weights: torch.Tensor

    def apply(self, values):
        """Mix pooled features and soft class targets using the same weights.

        Untouched rows (including every noise row) are copied exactly. Both
        feature endpoints retain gradients; interpolate in FP32 under AMP.
        """
        if values.ndim != 2:
            raise ValueError("Feature Mixup expects a [batch, dimension] tensor")
        mixed = values.clone()
        weights = self.weights[:, None]
        mixed[self.rows] = (weights * values[self.rows].float()
                            + (1 - weights) * values[self.partners].float()).to(values.dtype)
        return mixed


def make_feature_mixup_plan(labels, recording_ids, cfg, global_step, device):
    """Draw one plan per successful-update index, before the AMP retry loop.

    With probability p, mix every eligible ship row with a uniformly selected
    ship from a DIFFERENT recording in the same batch (partner reuse allowed).
    Same/different classes follow the same rule; noise and unknown recording IDs
    never participate. Otherwise leave the entire batch unchanged.

    A private RNG keyed by seed/global_step preserves existing augmentation and
    dropout RNG streams. Repeating an AMP attempt or resuming at the same step
    recreates exactly the same plan without extra checkpoint RNG state.
    """
    if cfg.feature_mixup_prob == 0:
        return None
    labels = labels.detach().cpu()
    if labels.ndim != 2 or labels.shape[1] != 3 or len(recording_ids) != len(labels):
        raise ValueError("Feature Mixup expects three-class labels and one recording ID per row")
    if not bool(((labels == 0) | (labels == 1)).all()) or bool((labels.sum(1) > 1).any()):
        raise ValueError("Feature Mixup input must contain original single-target/noise labels")
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, global_step, 0x4D495855]))
    if rng.random() >= cfg.feature_mixup_prob:
        return None
    recordings = ["" if value is None else str(value).strip() for value in recording_ids]
    ships = [i for i in range(len(labels)) if labels[i].sum().item() == 1 and recordings[i]]
    rows, partners, weights = [], [], []
    for i in ships:
        candidates = [j for j in ships if recordings[j] != recordings[i]]
        if candidates:
            rows.append(i)
            partners.append(int(rng.choice(candidates)))
            weights.append(float(rng.beta(cfg.feature_mixup_alpha, cfg.feature_mixup_alpha)))
    if not rows:
        return None
    return FeatureMixupPlan(torch.tensor(rows, dtype=torch.long, device=device),
                            torch.tensor(partners, dtype=torch.long, device=device),
                            torch.tensor(weights, dtype=torch.float32, device=device))
