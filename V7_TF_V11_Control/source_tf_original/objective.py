import torch
import torch.nn.functional as F
from cross_recording import cross_recording_consistency_loss
from m2_components import multihot_to_set_targets


def loss_components(model, outputs, labels, recording_ids, cfg):
    targets = multihot_to_set_targets(labels)
    if model.phase == "single" and bool((labels.sum(1) > 1).any()):
        raise ValueError("Double-target labels cannot enter the single-target phase")
    bce = F.binary_cross_entropy_with_logits(outputs["presence_logits"].float(), labels)
    single_ce = bce.new_zeros(())
    # Single-phase auxiliary classification: exclude noise and reuse the three logits.
    if model.phase == "single" and cfg.single_ce_weight > 0:
        ship_mask = labels.sum(dim=1) == 1
        if bool(ship_mask.any()):
            single_ce = F.cross_entropy(
                outputs["presence_logits"][ship_mask].float(),
                labels[ship_mask].argmax(dim=1))
    nll = bce.new_zeros(())
    main = bce
    if model.phase == "joint":
        nll = F.nll_loss(outputs["set_log_probs"], targets)
        main = nll + cfg.presence_weight * bce
    queries = F.normalize(model.classifier.query_attn.queries.float(), dim=-1)
    similarity = queries @ queries.T
    off_diagonal = ~torch.eye(3, dtype=torch.bool, device=similarity.device)
    ortho = similarity[off_diagonal].square().sum()
    cr, cr_stats = cross_recording_consistency_loss(
        outputs["query_features"], labels, labels.sum(1).long(), recording_ids, cfg.cr_tau)
    loss = (main + cfg.single_ce_weight * single_ce
            + cfg.ortho_weight * ortho + cfg.cr_weight * cr)
    return loss, {"bce": bce.detach(), "nll": nll.detach(), "cr": cr.detach(),
                  "single_ce": single_ce.detach(),
                  "cr_anchors": cr_stats["eligible_anchor_count"]}
