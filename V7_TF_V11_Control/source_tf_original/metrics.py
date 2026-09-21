import torch
from data import CLASS_NAMES


def _classification_metrics(pred, labels):
    pred, truth = pred.bool(), labels.bool()
    tp = (pred & truth).sum(0).float()
    fp = (pred & ~truth).sum(0).float()
    fn = (~pred & truth).sum(0).float()
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    return {"n": len(labels), "emr": float((pred == truth).all(1).float().mean()),
            "macro_f1": float(f1.mean()),
            "recall": {name: float(recall[i]) for i, name in enumerate(CLASS_NAMES)}}


def summarize(pred, labels, sir):
    result = _classification_metrics(pred, labels)
    count = labels.sum(1)
    for name, mask in (("single", count == 1), ("double", count == 2)):
        result[name] = _classification_metrics(pred[mask], labels[mask]) if mask.any() else None
    noise = count == 0
    result["noise_false_alarm"] = float(pred[noise].bool().any(1).float().mean()) if noise.any() else None
    result["sir_groups"] = {}
    for name, lower, upper in (("0_to_3", 0, 3), ("3_to_6", 3, 6),
                                ("6_to_10", 6, 10), ("10_plus", 10, float("inf"))):
        mask = (count == 2) & torch.isfinite(sir) & (sir >= lower) & (sir < upper)
        result["sir_groups"][name] = _classification_metrics(pred[mask], labels[mask]) if mask.any() else None
    return result
