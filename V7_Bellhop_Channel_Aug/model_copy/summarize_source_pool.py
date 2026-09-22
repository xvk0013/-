"""Aggregate the existing Val predictions by unique source and recording."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    a = p.parse_args()
    report = sorted(a.run_dir.glob("val_report_*"))[-1]
    source_ids = {}
    for name in ("Tanker", "Cargo", "Tug"):
        with (a.data_dir/name/"Val"/"all_info.txt").open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                source_ids[(name, r["file_name"])] = (name, r["raw_relative_path"].casefold(), r["start_sample"] if "start_sample" in r else r["segment_index"])
    unique = {}
    with (report/"predictions_ema.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["truth"] == "noise":
                continue
            key = source_ids[(r["truth"], Path(r["path"]).name)]
            if key in unique and unique[key]["prediction"] != r["prediction"]:
                raise ValueError("Repeated source has inconsistent predictions")
            unique[key] = r
    by_record = defaultdict(list)
    by_class = defaultdict(list)
    for (name, recording, _), r in unique.items():
        by_record[(name, recording)].append(r)
        by_class[name].append(r)
    def metrics(rows):
        return {"n_unique_sources": len(rows),
                "exact_accuracy": sum(int(r["exact_match"]) for r in rows)/len(rows),
                "predicted_tanker_fraction": sum(r["prediction"] == "Tanker" for r in rows)/len(rows)}
    result = {name: metrics(rows) for name, rows in by_class.items()}
    (report/"unique_source_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (report/"per_recording.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class_name", "recording", "n_unique_sources", "exact_accuracy", "predicted_tanker_fraction"])
        w.writeheader()
        for (name, recording), rows in sorted(by_record.items()):
            w.writerow({"class_name": name, "recording": recording, **metrics(rows)})
    print(json.dumps(result, indent=2))
    print(f"Recording/source reports: {report}")

if __name__ == "__main__":
    main()
