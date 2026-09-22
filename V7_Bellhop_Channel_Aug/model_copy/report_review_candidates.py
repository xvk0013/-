"""Report predeclared review candidates; never filter data or change metrics."""
import argparse
import csv
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report = sorted(args.run_dir.glob("val_report_*"))[-1]
    wanted = {
        "cargo/20171113-13/13.wav",
        "cargo/20171115d-20/20.wav",
        "tug/20171221a-62/144544.wav",
    }
    with (report/"per_recording.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["recording"].casefold() in wanted]
    result = {"status": "pending independent audio/label review",
              "invalid_recordings": [], "excluded_recordings": [],
              "selection_basis": "prior A/B results; descriptive follow-up only",
              "warning": "Poor model recognition alone is not evidence of invalid audio.",
              "audio_or_label_review_performed": False,
              "main_metrics_use_full_validation_set": True,
              "recordings": rows}
    (report/"review_candidates.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Review candidates retained in full Val: {report}")

if __name__ == "__main__":
    main()
