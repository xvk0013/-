"""Generate exactly one extra physical channel per Train ship; no model runs."""
import json
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from channel_data import (train_rows, load_bank, make_entries, source_context, propagate, write_json)
from v7_common import DATASET, ALTERNATE_ROOT, AUGMENTATION, validate_dataset

def render(row, bank, index, context=None):
    return propagate(source_context(row) if context is None else context,
        bank["H"][:,index],bank["nfft"],bank["history_samples"],bank["core_samples"],
        bank["memory_samples"],float(row["common_scale"]))

def check_original_chain(rows, entries, bank):
    checks=[]
    # Three representative original Train waves only; protects MATLAB/Python parity.
    for name in ("Tanker","Cargo","Tug"):
        e=next(e for e in entries if e["source_identity"][0]==name)
        row=rows[e["key"]]
        expected,sr=sf.read(DATASET/e["key"],dtype="float32")
        actual=render(row,bank,e["original_channel_index"])
        if sr!=16000 or expected.shape!=(80000,):
            raise ValueError("Original WAV format changed")
        relative=float(np.linalg.norm(actual.astype(float)-expected)/np.linalg.norm(expected))
        if relative>2e-5:
            raise ValueError(f"Propagation does not reproduce original {e['key']}: relative L2={relative}")
        checks.append({"key":e["key"],"relative_l2_error":relative})
    return checks

def main():
    validate_dataset()
    bank=load_bank()
    final=ALTERNATE_ROOT/"summary.json"
    if final.is_file():
        result=json.loads(final.read_text())
        if result.get("status")!="complete" or result.get("augmentation")!=AUGMENTATION:
            raise ValueError("Alternate data protocol mismatch")
        print(f"Alternate Train already complete: {final}",flush=True)
        return
    started=time.perf_counter()
    rows=train_rows()
    if len(rows)!=17118:
        raise ValueError(f"Expected 17118 original Train ships, found {len(rows)}")
    entries=make_entries(rows,bank)
    manifest={"augmentation":AUGMENTATION,"source_dataset":str(DATASET),"entries":entries}
    ALTERNATE_ROOT.mkdir(parents=True,exist_ok=True)
    path=ALTERNATE_ROOT/"manifest.json"
    if path.exists():
        if json.loads(path.read_text())!=manifest:
            raise ValueError("Existing alternate data belongs to another assignment; refusing overwrite")
    else:
        write_json(path,manifest)
    checks=check_original_chain(rows,entries,bank)
    resumed=0
    for i,e in enumerate(entries,1):
        target=ALTERNATE_ROOT/e["alternate_path"]
        if target.is_file():
            info=sf.info(target)
            if (info.frames,info.samplerate,info.channels,info.subtype)!=(80000,16000,1,"FLOAT"):
                raise ValueError(f"Invalid previously committed alternate WAV: {target}")
            resumed+=1
        else:
            wave=render(rows[e["key"]],bank,e["alternate_channel_index"])
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_suffix(".tmp")
            sf.write(temporary,wave,16000,format="WAV",subtype="FLOAT")
            temporary.replace(target)
        if i==1 or i%500==0 or i==len(entries):
            print(f"Alternate Bellhop Train: {i}/{len(entries)}; elapsed {time.perf_counter()-started:.1f}s",flush=True)
    from collections import Counter
    result={"status":"complete","augmentation":AUGMENTATION,"source_dataset":str(DATASET),
        "original_train_ships":len(rows),"additional_train_waveforms":len(entries),
        "unique_train_sources_after":len(rows),"additional_noise":0,"additional_val":0,
        "test_loaded":False,"class_counts":dict(Counter(e["source_identity"][0] for e in entries)),
        "retained_zero_history_padded_sources":sum(e["history_padding_samples"]>0 for e in entries),
        "alternate_channel_counts":dict(Counter(str(e["alternate_channel_index"]) for e in entries)),
        "original_chain_reproduction":checks,"generation_seconds_this_invocation":time.perf_counter()-started,
        "resumed_waveforms":resumed,"float32_audio_payload_gib":len(entries)*80000*4/2**30}
    write_json(final,result)
    print(f"READY: {final}; no training has run yet.",flush=True)

if __name__=="__main__":
    main()
