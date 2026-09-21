"""The same single-target BCE + 0.2 ship CE used by the V7 channel baseline."""
import torch
import torch.nn.functional as F


def loss_components(model, outputs, labels, recording_ids, cfg, *, soft_targets=None):
    if model.phase != "single" or soft_targets is not None:
        raise ValueError("This experiment trains only hard-label single-target/noise examples")
    if not bool(((labels == 0) | (labels == 1)).all()) or bool((labels.sum(1) > 1).any()):
        raise ValueError("Invalid single-target/noise label")
    logits = outputs["presence_logits"].float()
    bce = F.binary_cross_entropy_with_logits(logits, labels.float())
    ship = labels.sum(1) == 1
    ce = F.cross_entropy(logits[ship], labels[ship].argmax(1)) if bool(ship.any()) else logits.sum() * 0
    zero = bce.detach().new_zeros(())
    return bce + cfg.single_ce_weight * ce, {
        "bce": bce.detach(), "single_ce": ce.detach(), "nll": zero, "cr": zero, "cr_anchors": 0}
