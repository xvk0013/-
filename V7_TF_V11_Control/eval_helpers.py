"""Existing FP32, three-logit report implementation reused without BN diagnostics."""
import csv
import torch
import torch.nn.functional as F
from data import CLASS_NAMES
from metrics import summarize, format_noise_metrics
SET_CODES = (0, 1, 2, 4, 3, 5, 6, 7)
TRUE_CODES = (0, 1, 2, 4)


def set_name(code, class_names=CLASS_NAMES):
    return "+".join(name for i, name in enumerate(class_names) if code & (1 << i)) or "noise"

def set_codes(values):
    return (values.long() * torch.tensor([1, 2, 4])).sum(1)

@torch.no_grad()
def evaluate_variant(model, loader, device, output_dir, variant, split_name="Val"):
    class_names = tuple(loader.dataset.cfg.class_names)
    model.eval()
    all_predictions, all_labels, all_sirs = [], [], []
    confusion = torch.zeros((len(TRUE_CODES), len(SET_CODES)), dtype=torch.long)
    n_samples, bce_sum = 0, 0.0
    prediction_file = output_dir / f"predictions_{variant}.csv"
    with prediction_file.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["path", "recording", "truth", "prediction", "exact_match"]
        fields += [f"prob_{name}" for name in class_names]
        fields += [f"logit_{name}" for name in class_names]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step, batch in enumerate(loader, 1):
            labels = batch["labels"].to(device, non_blocking=True)
            outputs = model(batch["features"].to(device, non_blocking=True),
                            batch["demon"].to(device, non_blocking=True))
            logits = outputs["presence_logits"]
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError(f"Non-finite Val logits: {variant}")
            size = len(labels)
            bce_sum += float(F.binary_cross_entropy_with_logits(logits, labels)) * size
            n_samples += size
            predictions = model.decode(outputs).cpu()
            labels = labels.cpu()
            logits = logits.cpu()
            probabilities = logits.sigmoid()
            truth_codes, prediction_codes = set_codes(labels), set_codes(predictions)
            for i, truth in enumerate(TRUE_CODES):
                for j, prediction in enumerate(SET_CODES):
                    confusion[i, j] += ((truth_codes == truth) & (prediction_codes == prediction)).sum()
            for i in range(size):
                truth, prediction = int(truth_codes[i]), int(prediction_codes[i])
                row = {"path": batch["path"][i], "recording": batch["recording"][i],
                       "truth": set_name(truth, class_names), "prediction": set_name(prediction, class_names),
                       "exact_match": int(truth == prediction)}
                row.update({f"prob_{name}": float(probabilities[i, j])
                            for j, name in enumerate(class_names)})
                row.update({f"logit_{name}": float(logits[i, j])
                            for j, name in enumerate(class_names)})
                writer.writerow(row)
            all_predictions.append(predictions)
            all_labels.append(labels)
            all_sirs.append(batch["sir_db"])
            if step == 1 or step % 25 == 0 or step == len(loader):
                print(f"  {variant} {split_name}: {step}/{len(loader)} batches", flush=True)
    if not n_samples:
        raise ValueError("Empty Val loader")
    metrics = summarize(torch.cat(all_predictions), torch.cat(all_labels), torch.cat(all_sirs),
                        class_names=class_names)
    metrics.update(bce=bce_sum / n_samples, nll=None)
    with (output_dir / f"confusion_{variant}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["truth / prediction"] + [set_name(code, class_names) for code in SET_CODES])
        for code, row in zip(TRUE_CODES, confusion.tolist()):
            writer.writerow([set_name(code, class_names)] + row)
    directions = []
    for i, truth in enumerate(TRUE_CODES):
        total = int(confusion[i].sum())
        for j, prediction in enumerate(SET_CODES):
            count = int(confusion[i, j])
            if prediction != truth and count:
                directions.append({"truth": set_name(truth, class_names), "prediction": set_name(prediction, class_names),
                                   "count": count, "fraction_of_true_class": count / total})
    directions.sort(key=lambda item: (-item["count"], item["truth"], item["prediction"]))
    print(f"{variant}: EMR={metrics['emr']:.6f}, single EMR={metrics['single']['emr']:.6f}, "
          f"single Macro-F1={metrics['single']['macro_f1']:.6f}, "
          f"recall={metrics['single']['recall']}; {format_noise_metrics(metrics)}", flush=True)
    return {"metrics": metrics, "confusion": confusion.tolist(), "error_directions": directions}
