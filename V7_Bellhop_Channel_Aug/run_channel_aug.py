"""Prepare one alternate Train view, train ONCE, report against saved baseline."""
import argparse
import json
import subprocess
import sys
from v7_common import (HERE, DATASET, OUTPUT, RUN_DIR, REPORT_DIR,
    BASELINE_SUMMARY, ALTERNATE_ROOT, AUGMENTATION, validate_dataset)
from comparison import compare

PROTOCOL = {
    "name":"v7_bellhop_two_views_s42", "new_training_runs":1,
    "initialization":"fresh original Mamba seed42; no checkpoint initialization",
    "normalization":"unchanged original Bellhop Train feature statistics",
    "seed":42, "epochs":60, "batch_size":32,
    "ships_per_class_per_epoch":1800, "noise_per_epoch":1080,
    "optimizer_steps_per_epoch":203, "optimizer_steps_total":12180,
    "selection":"best EMA Val single_emr; independent sigmoid threshold 0.5",
    "loss":"original BCE + 0.2 ship CE; no distillation or consistency term",
    "augmentation":AUGMENTATION, "inference_parameters":2090723,
    "post_training_evaluation":"original full Train and Val; alternate views NOT evaluated",
    "test_loaded":False,
}

def training_command():
    command=[
        sys.executable,"-u",str(HERE/"train_v7.py"),
        "--phase","single","--data-dir",str(DATASET),"--noise-data-dir",str(DATASET),
        "--output-dir",str(RUN_DIR),"--device","cuda","--scan-backend","cuda",
        "--epochs","60","--batch-size","32","--workers","4","--seed","42",
        "--head-type","gap","--single-ce-weight","0.2","--weight-decay","0.0001",
        "--band-fusion","early","--local-contrast-bins","0",
        "--spectral-aug-prob","0","--feature-mixup-prob","0",
    ]
    if (RUN_DIR/"last.pt").is_file():
        command += ["--resume",str(RUN_DIR/"last.pt")]
    return command

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe",action="store_true")
    args=parser.parse_args()
    print(json.dumps(PROTOCOL,indent=2),flush=True)
    if args.describe: return
    final=OUTPUT/"summary.json"
    if final.is_file():
        print(f"Already complete. No repeated training/report: {final}",flush=True)
        return
    generation=validate_dataset()
    baseline=json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8-sig"))
    OUTPUT.mkdir(exist_ok=True)
    marker=OUTPUT/"training_complete.json"
    if not marker.is_file():
        print("STAGE 1/3: one additional Bellhop view per Train ship; no training",flush=True)
        subprocess.run([sys.executable,"-u",str(HERE/"prepare_alternate.py")],cwd=HERE,check=True)
        if RUN_DIR.exists() and not (RUN_DIR/"last.pt").is_file():
            if any(RUN_DIR.iterdir()):
                raise RuntimeError("Run contains files without last.pt; refusing overwrite")
            RUN_DIR.rmdir()
        print("STAGE 2/3: TRAINING 1/1; 60 epochs, 203 successful updates/epoch",flush=True)
        subprocess.run(training_command(),cwd=HERE,check=True)
        marker.write_text(json.dumps(PROTOCOL,indent=2),encoding="utf-8")
    if not (REPORT_DIR/"summary.json").is_file():
        print("STAGE 3/3: best EMA original full Train/Val report; Test NOT LOADED",flush=True)
        subprocess.run([sys.executable,"-u",str(HERE/"report_v7.py"),
            "--checkpoint",str(RUN_DIR/"best.pt"),"--data-dir",str(DATASET),
            "--device","cuda","--workers","4","--output-dir",str(REPORT_DIR)],cwd=HERE,check=True)
    result=json.loads((REPORT_DIR/"summary.json").read_text())
    result.update(status="complete",protocol=PROTOCOL,comparison=compare(baseline,result),
        generation_summary=generation,baseline_summary=str(BASELINE_SUMMARY),
        alternate_generation=json.loads((ALTERNATE_ROOT/"summary.json").read_text()),
        training_record=json.loads((RUN_DIR/"config.json").read_text()),
        training_history=[json.loads(line) for line in (RUN_DIR/"history.jsonl").read_text().splitlines() if line.strip()])
    history=result["training_history"]
    if [r["epoch"] for r in history]!=list(range(1,61)):
        raise ValueError("Expected exactly 60 committed epochs")
    if any(r["train"]["optimizer_steps"]!=203 for r in history):
        raise ValueError("Training update budget changed")
    if any(not 0<=r["train"]["alternate_ship_draws"]<=5400 for r in history):
        raise ValueError("Invalid alternate-view exposure")
    result["observed_alternate_ship_fraction"]=sum(r["train"]["alternate_ship_draws"] for r in history)/(60*5400)
    temporary=OUTPUT/"summary.tmp"
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    from checkpoint_io import replace_checkpoint
    replace_checkpoint(temporary,final)
    print(f"COMPLETE. Return only: {final}",flush=True)

if __name__=="__main__":
    main()
