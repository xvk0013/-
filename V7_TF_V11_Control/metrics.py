import torch
from data import CLASS_NAMES


def _classification_metrics(pred, labels, class_names=CLASS_NAMES):
    pred, truth = pred.bool(), labels.bool()
    tp = (pred & truth).sum(0).float()
    fp = (pred & ~truth).sum(0).float()
    fn = (~pred & truth).sum(0).float()
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    return {"n": len(labels), "emr": float((pred == truth).all(1).float().mean()),
            "macro_f1": float(f1.mean()),
            "recall": {name: float(recall[i]) for i, name in enumerate(class_names)}}


def noise_metrics_from_counts(n_true, n_predicted, n_correct):
    """Noise means the predicted set is empty; precision includes rejected ships."""
    if n_true == 0:
        return None
    return {"n": n_true, "correct": n_correct,
            "recall": n_correct / n_true,
            "precision": n_correct / n_predicted if n_predicted else 0.0,
            "f1": 2 * n_correct / (n_true + n_predicted),
            "false_alarm": (n_true - n_correct) / n_true,
            "ships_as_noise": n_predicted - n_correct}


def format_noise_metrics(metrics):
    noise = metrics["noise"]
    if noise is None:
        return "noise: n=0"
    return (f"noise recall={noise['recall']:.4%} ({noise['correct']}/{noise['n']}), "
            f"precision={noise['precision']:.4%}, F1={noise['f1']:.4%}, "
            f"FA={noise['false_alarm']:.4%}")


def selection_score(metrics, phase, cfg):
    if phase == "joint":
        return metrics["emr"]
    key = {"single_macro_f1": "macro_f1", "single_emr": "emr"}[cfg.selection_metric]
    return metrics["single"][key]


def summarize(pred, labels, sir, class_names=CLASS_NAMES):
    result = _classification_metrics(pred, labels, class_names)
    count = labels.sum(1)
    for name, mask in (("single", count == 1), ("double", count == 2)):
        result[name] = _classification_metrics(pred[mask], labels[mask], class_names) if mask.any() else None
    noise = count == 0
    predicted_noise = ~pred.bool().any(1)
    result["noise"] = noise_metrics_from_counts(
        int(noise.sum()), int(predicted_noise.sum()), int((noise & predicted_noise).sum()))
    result["noise_false_alarm"] = result["noise"]["false_alarm"] if result["noise"] else None
    result["sir_groups"] = {}
    for name, lower, upper in (("0_to_3", 0, 3), ("3_to_6", 3, 6),
                                ("6_to_10", 6, 10), ("10_plus", 10, float("inf"))):
        mask = (count == 2) & torch.isfinite(sir) & (sir >= lower) & (sir < upper)
        result["sir_groups"][name] = _classification_metrics(pred[mask], labels[mask], class_names) if mask.any() else None
    return result
