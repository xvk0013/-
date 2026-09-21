"""Offline ranking/threshold diagnosis; no training or inference."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

CLASSES = ("Tanker", "Cargo", "Tug")

def classify(row):
    probs = [float(row[f"prob_{c}"]) for c in CLASSES]
    logits = [float(row[f"logit_{c}"]) for c in CLASSES]
    if not all(math.isfinite(x) for x in probs + logits) or not all(0 <= x <= 1 for x in probs):
        raise ValueError("Invalid scores")
    truth, prediction = row["truth"], row["prediction"]
    active = [] if prediction == "noise" else prediction.split("+")
    if any(c not in CLASSES for c in active):
        raise ValueError("Unknown predicted label")
    winners = [CLASSES[i] for i, x in enumerate(logits) if x == max(logits)]
    result = dict(row)
    result.update(top_class="|".join(winners), max_probability=max(probs),
                  active_count=len(active), true_logit_margin="", top1_correct="")
    if int(row["exact_match"]) != int(prediction == truth):
        raise ValueError("Inconsistent exact_match")
    if truth == "noise":
        result["category"] = "noise_correct" if not active else "noise_false_alarm"
        return result
    if truth not in CLASSES:
        raise ValueError("Single-target reports only")
    i = CLASSES.index(truth)
    margin = logits[i] - max(x for j, x in enumerate(logits) if i != j)
    top = len(winners) == 1 and winners[0] == truth
    ranking = "tie" if len(winners) > 1 else ("correct" if top else "wrong")
    if prediction == truth:
        category = "exact_correct"
    elif not active:
        category = f"rejected_top1_{ranking}"
    elif len(active) > 1:
        category = f"multiple_top1_{ranking}"
    else:
        category = "wrong_single_class"
    result.update(category=category, top1_correct=int(top), true_logit_margin=margin)
    return result

def summarize(rows):
    out = {}
    for name in ("all_ships",) + CLASSES + ("noise",):
        group = [r for r in rows if (r["truth"] != "noise" if name == "all_ships" else r["truth"] == name)]
        n = len(group)
        entry = {"n": n, "categories": dict(Counter(r["category"] for r in group)),
                 "formal_exact_accuracy": sum(int(r["exact_match"]) for r in group)/n if n else None}
        if name != "noise":
            entry["ship_only_top1_accuracy_diagnostic"] = sum(int(r["top1_correct"]) for r in group)/n if n else None
            entry["top1_correct_but_formal_wrong"] = sum(int(r["top1_correct"]) and not int(r["exact_match"]) for r in group)
            entry["top1_wrong_or_tied"] = sum(not int(r["top1_correct"]) for r in group)
        else:
            scores = sorted(float(r["max_probability"]) for r in group)
            entry["max_ship_probability_quantiles"] = {
                str(q): scores[round((n-1)*q)] if n else None for q in (0.5, 0.9, 0.99, 1.0)}
        out[name] = entry
    return out

def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.report_dir/"summary.json").read_text(encoding="utf-8"))
    if report["split"] != "Val" or report["variant"] != "ema" or report["threshold"] != 0.5:
        raise ValueError("Requires original EMA Val report at threshold 0.5")
    if tuple(report["config"]["class_names"]) != CLASSES:
        raise ValueError("Unexpected class order")
    with (args.report_dir/"predictions_ema.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = [classify(r) for r in csv.DictReader(f)]
    if len(rows) != report["metrics"]["n"]:
        raise ValueError("Prediction count differs")
    if abs(sum(int(r["exact_match"]) for r in rows)/len(rows)-report["metrics"]["emr"]) > 1e-6:
        raise ValueError("Formal accuracy differs")
    mapping = {}
    for name in CLASSES:
        with (args.data_dir/name/"Val"/"all_info.txt").open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                identity = r["start_sample"] if "start_sample" in r else r["segment_index"]
                mapping[(name, r["file_name"])] = (name, r["raw_relative_path"].casefold(), identity)
    unique = {}
    for r in rows:
        key = ("noise", r["path"]) if r["truth"] == "noise" else mapping[(r["truth"], Path(r["path"]).name)]
        if key in unique:
            old = unique[key]
            if old["prediction"] != r["prediction"] or old["top_class"] != r["top_class"]:
                raise ValueError("Repeated source predictions differ; inspect before deduplication")
        else:
            unique[key] = r
    source_rows = list(unique.values())
    groups = defaultdict(list)
    for r in source_rows:
        if r["truth"] != "noise":
            groups[(r["truth"], r["recording"])].append(r)
    recording_rows = []
    for (name, recording), group in sorted(groups.items()):
        stats = summarize(group)[name]
        recording_rows.append({"class_name": name, "recording": recording,
            "n_unique_sources": len(group), "formal_exact_accuracy": stats["formal_exact_accuracy"],
            "ship_only_top1_accuracy_diagnostic": stats["ship_only_top1_accuracy_diagnostic"],
            "top1_correct_but_formal_wrong": stats["top1_correct_but_formal_wrong"],
            "top1_wrong_or_tied": stats["top1_wrong_or_tied"],
            "categories": json.dumps(stats["categories"], ensure_ascii=False)})
    out = args.report_dir/("ranking_diagnostic_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir()
    result = {"source_report": str(args.report_dir.resolve()), "epoch": report["epoch"],
        "seed": report["config"]["seed"], "threshold": 0.5,
        "note": "Ship-only top1 uses known ship membership for diagnosis, not a deployable metric or achievable bound. No threshold search, exclusions or Test access.",
        "scene_level": summarize(rows), "unique_source_level": summarize(source_rows)}
    (out/"diagnostic_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out/"sample_diagnostics.csv", rows)
    write_csv(out/"per_recording_diagnostic.csv", recording_rows)
    lines = ["# 类别排名与 0.5 阈值诊断", "",
        "正式预测保持原样。Top-1 仅在已知真实船舶样本上诊断，不能替代包含 noise 的正式评价。",
        "未调整阈值、训练模型、删除样本或读取 Test。", ""]
    for level in ("scene_level", "unique_source_level"):
        lines += [f"## {level}", "", "| 类别 | 数量 | 正式完全正确率 | 船舶条件 Top-1 | 排名对但正式错误 | 排名错/并列 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name in ("all_ships",)+CLASSES:
            x=result[level][name]
            lines.append(f"| {name} | {x['n']} | {x['formal_exact_accuracy']:.2%} | "
                f"{x['ship_only_top1_accuracy_diagnostic']:.2%} | {x['top1_correct_but_formal_wrong']} | {x['top1_wrong_or_tied']} |")
        lines += ["", "错误分组（数量）：", ""]
        for name in CLASSES:
            lines.append(f"- {name}: "+json.dumps(result[level][name]["categories"], ensure_ascii=False))
        lines += [""]
    lines += ["## 分组说明", "",
        "- rejected_top1_correct：三类均未检出，但真实类唯一排名第一。",
        "- rejected_top1_wrong/tie：拒识同时伴随排名错误/并列。",
        "- multiple_top1_correct：真实类排名第一，但额外类别也越过阈值。",
        "- multiple_top1_wrong/tie：多类别误报，同时排名错误/并列。",
        "- wrong_single_class：只检出一个错误类别。",
        "- Top-1 差距不能直接视为降低阈值可实现的收益；还需考虑 noise 虚警。"]
    (out/"summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"COMPLETE: {out}")

if __name__ == "__main__":
    main()
