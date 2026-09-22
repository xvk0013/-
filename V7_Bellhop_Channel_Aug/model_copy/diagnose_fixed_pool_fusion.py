"""Fixed equal-logit fusion of two saved EMA Val predictions, without tuning."""
import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from diagnose_ranking_threshold import CLASSES, classify, summarize, write_csv

PREDICTIONS = ("noise", "Tanker", "Cargo", "Tug", "Tanker+Cargo",
               "Tanker+Tug", "Cargo+Tug", "Tanker+Cargo+Tug")

def sigmoid(x):
    if x >= 0:
        return 1/(1+math.exp(-x))
    e = math.exp(x)
    return e/(1+e)

def fuse(a, b):
    if any(a[k] != b[k] for k in ("path", "recording", "truth")):
        raise ValueError("Prediction rows do not refer to the same sample")
    result = {k: a[k] for k in ("path", "recording", "truth")}
    values = [(float(a["logit_"+c])+float(b["logit_"+c]))/2 for c in CLASSES]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Nonfinite logits")
    result["prediction"] = "+".join(c for c,x in zip(CLASSES,values) if x >= 0) or "noise"
    result["exact_match"] = int(result["truth"] == result["prediction"])
    for c,x in zip(CLASSES,values):
        result["prob_"+c] = sigmoid(x)
        result["logit_"+c] = x
    return classify(result)

def metrics(rows):
    ships = [r for r in rows if r["truth"] != "noise"]
    noise = [r for r in rows if r["truth"] == "noise"]
    def macro(group):
        scores = []
        for c in CLASSES:
            tp = sum(r["truth"] == c and c in r["prediction"].split("+") for r in group)
            fp = sum(r["truth"] != c and c in r["prediction"].split("+") for r in group)
            fn = sum(r["truth"] == c and c not in r["prediction"].split("+") for r in group)
            scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
        return sum(scores)/len(scores)
    return {"n": len(rows),
        "overall_emr": sum(int(r["exact_match"]) for r in rows)/len(rows),
        "single_emr": sum(int(r["exact_match"]) for r in ships)/len(ships),
        "single_macro_f1": macro(ships), "overall_macro_f1": macro(rows),
        "noise_n": len(noise),
        "noise_false_alarms": sum(r["prediction"] != "noise" for r in noise),
        "noise_false_alarm_rate": sum(r["prediction"] != "noise" for r in noise)/len(noise),
        "ships_as_noise": sum(r["prediction"] == "noise" for r in ships),
        "multiple_predictions_on_ships": sum("+" in r["prediction"] for r in ships),
        "diagnostic": summarize(rows)}

def read_report(folder):
    meta = json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    if meta["split"] != "Val" or meta["variant"] != "ema" or meta["threshold"] != 0.5:
        raise ValueError("Requires EMA Val reports at threshold 0.5")
    if tuple(meta["config"]["class_names"]) != CLASSES:
        raise ValueError("Unexpected class order")
    with (folder/"predictions_ema.csv").open(encoding="utf-8-sig",newline="") as f:
        rows = [classify(r) for r in csv.DictReader(f)]
    by_path = {r["path"]:r for r in rows}
    if len(by_path) != len(rows) or len(rows) != meta["metrics"]["n"]:
        raise ValueError("Duplicate paths or inconsistent counts")
    return meta, by_path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gap-report",type=Path,required=True)
    p.add_argument("--freq4-report",type=Path,required=True)
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--output-root",type=Path,default=Path("runs/fixed_gap_freq4_fusion_s42"))
    args = p.parse_args()
    ma,a = read_report(args.gap_report)
    mb,b = read_report(args.freq4_report)
    if a.keys() != b.keys() or ma["data_dir"] != mb["data_dir"] or ma["config"]["seed"] != mb["config"]["seed"]:
        raise ValueError("Reports must have identical samples, dataset and seed")
    if ma["config"]["head_type"] != "gap" or mb["config"]["head_type"] != "freq4":
        raise ValueError("Expected GAP then freq4")
    fused = [fuse(a[k],b[k]) for k in a]
    mapping = {}
    for name in CLASSES:
        with (args.data_dir/name/"Val"/"all_info.txt").open(encoding="utf-8-sig",newline="") as f:
            for row in csv.DictReader(f,delimiter="\t"):
                start = row["start_sample"] if "start_sample" in row else row["segment_index"]
                mapping[(name,row["file_name"])] = (name,row["raw_relative_path"].casefold(),start)
    def dedup(rows):
        seen = {}
        for r in rows:
            key = ("noise",r["path"]) if r["truth"] == "noise" else mapping[(r["truth"],Path(r["path"]).name)]
            if key in seen:
                if seen[key]["prediction"] != r["prediction"] or seen[key]["top_class"] != r["top_class"]:
                    raise ValueError("Repeated source predictions disagree")
            else:
                seen[key] = r
        return list(seen.values())
    variants = {"gap":list(a.values()),"freq4":list(b.values()),"fixed_equal_logit_fusion":fused}
    result = {"weights":[0.5,0.5],"threshold":0.5,"seed":ma["config"]["seed"],
        "source_reports":[str(args.gap_report),str(args.freq4_report)],
        "note":"Equal raw logits, not probabilities. No calibration, weight/threshold search, exclusions or Test. Ensemble diagnosis is not a single-backbone fusion result.",
        "scene_level":{k:metrics(v) for k,v in variants.items()},
        "unique_source_level":{k:metrics(dedup(v)) for k,v in variants.items()}}
    out = args.output_root/("diagnostic_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True)
    (out/"fusion_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    write_csv(out/"predictions_fused.csv",fused)
    with (out/"confusion_fused.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["truth / prediction",*PREDICTIONS])
        for truth in ("noise",)+CLASSES:
            writer.writerow([truth,*[sum(r["truth"]==truth and r["prediction"]==pred for r in fused) for pred in PREDICTIONS]])
    lines=["# 固定 1:1 logits 融合", "",
        "不训练、不调参、不剔除样本；两模型集成结果不能直接证明单主干融合头有效。", ""]
    for level in ("scene_level","unique_source_level"):
        lines += [f"## {level}", "",
            "| 方案 | 单目标 EMR | 单目标 Macro-F1 | Tanker | Cargo | Tug | 船舶拒识 | noise 虚警 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name,x in result[level].items():
            acc=[x["diagnostic"][c]["formal_exact_accuracy"] for c in CLASSES]
            lines.append(f"| {name} | {x['single_emr']:.2%} | {x['single_macro_f1']:.2%} | "
                f"{acc[0]:.2%} | {acc[1]:.2%} | {acc[2]:.2%} | {x['ships_as_noise']} | {x['noise_false_alarms']} |")
        lines += [""]
    (out/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"COMPLETE: {out}")

if __name__ == "__main__":
    main()
