"""Same-class/different-recording supervised contrastive consistency."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def cross_recording_consistency_loss(
    query_features: torch.Tensor,
    labels: torch.Tensor,
    sample_types: torch.Tensor,
    recording_ids: Sequence[str],
    tau: float = 0.10,
    return_masks: bool = False,
):
    """Return M2 CR loss and batch diagnostics.

    Only exact single-target rows enter the contrastive set. Positives have the
    same target class and a different source recording. Different classes are
    negatives. Same-class/same-recording pairs are ignored.
    """

    if query_features.ndim != 3 or query_features.shape[1] != 3:
        raise ValueError("query_features must have shape [B, 3, D]")
    batch_size = query_features.shape[0]
    if labels.shape != (batch_size, 3):
        raise ValueError("labels must have shape [B, 3]")
    if sample_types.shape != (batch_size,):
        raise ValueError("sample_types must have shape [B]")
    if len(recording_ids) != batch_size:
        raise ValueError("recording_ids length must equal B")
    if not tau > 0:
        raise ValueError("tau must be positive")
    if not torch.isfinite(query_features).all():
        raise FloatingPointError("query_features contain NaN or Inf")

    exact_single = (sample_types == 1) & (labels.sum(dim=1) == 1)
    has_recording = torch.tensor(
        [bool(str(value)) for value in recording_ids],
        dtype=torch.bool, device=query_features.device,
    )
    single_mask = exact_single & has_recording
    full_targets = labels.argmax(dim=1)
    selected_rows = torch.nonzero(single_mask, as_tuple=False).flatten()
    class_totals = [int(((full_targets == k) & single_mask).sum().item()) for k in range(3)]

    empty_masks = torch.zeros((len(selected_rows), len(selected_rows)), dtype=torch.bool,
                              device=query_features.device)
    if selected_rows.numel() == 0:
        zero = query_features.sum() * 0.0
        stats = {
            "single_count": 0, "eligible_anchor_count": 0,
            "eligible_fraction": 0.0, "positive_pair_count": 0,
            "negative_pair_count": 0, "ignored_same_record_pair_count": 0,
            "class_single_counts": class_totals, "class_eligible_counts": [0, 0, 0],
        }
        if return_masks:
            stats.update({"positive_mask": empty_masks, "negative_mask": empty_masks,
                          "ignored_same_record_mask": empty_masks,
                          "selected_rows": selected_rows})
        return zero, stats

    targets = full_targets[selected_rows]
    # Keep the contrastive softmax numerically stable under the frozen AMP
    # recipe while preserving gradients to the original query representation.
    embeddings = F.normalize(
        query_features[selected_rows, targets].float(), p=2, dim=1, eps=1e-12
    )
    similarities = embeddings @ embeddings.transpose(0, 1)
    n_single = len(selected_rows)
    eye = torch.eye(n_single, dtype=torch.bool, device=query_features.device)
    same_class = targets[:, None] == targets[None, :]
    selected_recordings = [str(recording_ids[int(index)]) for index in selected_rows.cpu()]
    same_recording = torch.tensor(
        [[left == right for right in selected_recordings] for left in selected_recordings],
        dtype=torch.bool, device=query_features.device,
    )
    positive_mask = same_class & ~same_recording & ~eye
    negative_mask = ~same_class & ~eye
    ignored_same_record = same_class & same_recording & ~eye
    valid_mask = positive_mask | negative_mask
    positive_counts = positive_mask.sum(dim=1)
    eligible_mask = positive_counts > 0

    if bool(eligible_mask.any()):
        logits = similarities / float(tau)
        log_denominator = torch.logsumexp(logits.masked_fill(~valid_mask, -torch.inf), dim=1)
        positive_logit_mean = (
            logits.masked_fill(~positive_mask, 0.0).sum(dim=1)
            / positive_counts.clamp_min(1)
        )
        loss = (log_denominator - positive_logit_mean)[eligible_mask].mean()
    else:
        loss = query_features.sum() * 0.0

    if not torch.isfinite(loss):
        raise FloatingPointError("M2 cross-recording loss is NaN or Inf")
    class_eligible = [
        int(((targets == k) & eligible_mask).sum().item()) for k in range(3)
    ]
    stats = {
        "single_count": n_single,
        "eligible_anchor_count": int(eligible_mask.sum().item()),
        "eligible_fraction": float(eligible_mask.float().mean().item()),
        "positive_pair_count": int(positive_mask.sum().item()),
        "negative_pair_count": int(negative_mask.sum().item()),
        "ignored_same_record_pair_count": int(ignored_same_record.sum().item()),
        "class_single_counts": class_totals,
        "class_eligible_counts": class_eligible,
    }
    if return_masks:
        stats.update({
            "positive_mask": positive_mask,
            "negative_mask": negative_mask,
            "ignored_same_record_mask": ignored_same_record,
            "selected_rows": selected_rows,
        })
    return loss, stats
