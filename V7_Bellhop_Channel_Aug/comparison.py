def compare(baseline,result):
    out={}
    for split in ("Train","Val"):
        a,b=baseline["splits"][split],result["splits"][split]
        fields={
            "single_emr":(a["metrics"]["single"]["emr"],b["metrics"]["single"]["emr"]),
            "single_macro_f1":(a["metrics"]["single"]["macro_f1"],b["metrics"]["single"]["macro_f1"]),
            "noise_false_alarm":(a["metrics"]["noise_false_alarm"],b["metrics"]["noise_false_alarm"]),
        }
        for name in ("Tanker","Cargo","Tug"):
            fields[name+"_strict_accuracy"]=(
                a["unique_source_level"][name]["formal_exact_accuracy"],
                b["unique_source_level"][name]["formal_exact_accuracy"])
        out[split]={k:{"baseline":x,"augmented":y,"augmented_minus_baseline_pp":100*(y-x)}
                    for k,(x,y) in fields.items()}
    return out

