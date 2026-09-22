"""Build C from Train metadata only; retain A recording quotas, spread over B."""
import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

CLASSES = ("Tanker", "Cargo", "Tug")

def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        return reader.fieldnames, rows

def spread_indices(n, quota):
    if not 0 < quota <= n:
        raise ValueError(f"Invalid quota {quota} for {n} candidates")
    if quota == 1:
        return [(n - 1) // 2]
    # Deterministic rounded quantiles of time-ordered eligible candidates.
    return [(2 * i * (n - 1) + quota - 1) // (2 * (quota - 1))
            for i in range(quota)]

def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return str(dst)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "/mnt/d/LSJ/Data/deepship_source_pool_ab/20260916_225548_748"))
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root/"COMPLETE.txt").is_file():
        raise FileNotFoundError("Complete A/B generation first")
    out = root/"C_time_spread"
    if out.exists():
        raise FileExistsError(f"Will not overwrite: {out}")
    plans = {}
    summaries = []
    for name in CLASSES:
        _, a = read_table(root/"A_original"/name/"Train"/"all_info.txt")
        fields, b = read_table(root/"B_expanded"/name/"Train"/"all_info.txt")
        groups_a, groups_b = defaultdict(list), defaultdict(list)
        for rows, groups in ((a, groups_a), (b, groups_b)):
            for r in rows:
                if r["split"] != "Train" or r["class_name"] != name:
                    raise ValueError("Unexpected split/class")
                groups[r["raw_relative_path"].casefold()].append(r)
        if len(a) != 700 or set(groups_a) != set(groups_b):
            raise ValueError("A/B must have identical recording sets and 700 A clips")
        selected = []
        for recording in sorted(groups_a):
            original = groups_a[recording]
            pool = sorted(groups_b[recording], key=lambda r: int(r["start_sample"]))
            positions = [int(r["start_sample"]) for r in pool]
            if len(set(positions)) != len(pool):
                raise ValueError("Duplicate candidate positions")
            chosen = [pool[i] for i in spread_indices(len(pool), len(original))]
            old_starts = {int(r["start_sample"]) for r in original}
            if not old_starts.issubset(set(positions)):
                raise ValueError("A is not contained in B")
            selected.extend(chosen)
            summaries.append({
                "class_name": name, "recording": recording, "A_quota": len(original),
                "B_candidates": len(pool), "C_selected": len(chosen),
                "C_overlap_A": sum(int(r["start_sample"]) in old_starts for r in chosen),
                "A_first_sample": min(old_starts), "A_last_sample": max(old_starts),
                "C_first_sample": int(chosen[0]["start_sample"]),
                "C_last_sample": int(chosen[-1]["start_sample"])})
        selected.sort(key=lambda r: r["file_name"])
        assert len(selected) == 700
        for r in selected:
            if r["audio_path"] != f'{name}/Train/{r["file_name"]}':
                raise ValueError("Unexpected audio path")
            if not (root/"B_expanded"/r["audio_path"]).is_file():
                raise FileNotFoundError(r["audio_path"])
        plans[name] = (fields, selected)
    out.mkdir()
    for name, (fields, selected) in plans.items():
        directory = out/name/"Train"
        directory.mkdir(parents=True)
        for r in selected:
            link_or_copy(root/"B_expanded"/r["audio_path"], directory/r["file_name"])
        with (directory/"all_info.txt").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(selected)
        # Keep every Val sample, including difficult recordings. No Test access.
        shutil.copytree(root/"A_original"/name/"Val", out/name/"Val",
                        copy_function=link_or_copy)
        print(f"{name}: 700 Train clips; unchanged Val")
    with (out/"selection_by_recording.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    protocol = {"rule": "rounded rank quantiles over time-ordered eligible Train clips",
                "quota": "same per-recording quota as A", "selection_seed": None,
                "validation_used_for_selection": False, "excluded_val_recordings": [],
                "waveforms": "linked to B or copied byte-for-byte; do not edit linked files",
                "source": str(root/"B_expanded")}
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (out/"COMPLETE.txt").write_text("C export complete\n", encoding="utf-8")
    print(f"COMPLETE: {out}")

if __name__ == "__main__":
    main()
