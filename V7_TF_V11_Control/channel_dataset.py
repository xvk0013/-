"""One source draw -> one original OR alternate propagated wave, never both."""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from v7_common import bootstrap, DATASET, ALTERNATE_ROOT, AUGMENTATION, local_path
bootstrap()
import torch
from data import SceneDataset
from channel_data import train_rows, source_identity

class ChannelSceneDataset(SceneDataset):
    def __init__(self, records, cfg, stats, training=True, *, root=DATASET, alternate_root=ALTERNATE_ROOT):
        if not training:
            raise ValueError("Use unmodified SceneDataset for validation/reporting")
        if cfg.spectral_aug_prob or cfg.feature_mixup_prob:
            raise ValueError("Other waveform/feature mixing ablations must be disabled")
        super().__init__(records,cfg,stats,True)
        root=Path(root); alternate_root=Path(alternate_root)
        summary=json.loads((alternate_root/"summary.json").read_text())
        manifest=json.loads((alternate_root/"manifest.json").read_text())
        if (summary.get("status")!="complete" or summary.get("augmentation")!=AUGMENTATION
                or manifest.get("augmentation")!=AUGMENTATION
                or local_path(manifest["source_dataset"])!=root.resolve()):
            raise ValueError("Alternate Train data is incomplete or uses a different protocol")
        rows=train_rows(root)
        entries={e["key"]:e for e in manifest["entries"]}
        if len(entries)!=len(manifest["entries"]) or entries.keys()!=rows.keys():
            raise ValueError("Alternate sources must match the entire original Train ship pool")
        alternate_records=[]
        ship_keys=set()
        for scene in records:
            if sum(scene.labels)==0:
                alternate_records.append(scene)
                continue
            key=scene.path.relative_to(root).as_posix()
            if key not in entries or source_identity(rows[key])!=entries[key]["source_identity"]:
                raise ValueError("Alternate source identity mismatch")
            entry=entries[key]
            if (entry["original_range_km"],entry["original_depth_m"]) == (
                    entry["alternate_range_km"],entry["alternate_depth_m"]):
                raise ValueError("Alternate propagation must use a different channel")
            relative=Path(entry["alternate_path"])
            if relative.is_absolute() or ".." in relative.parts or relative.parts[:2]!=(rows[key]["class_name"],"Train"):
                raise ValueError("Alternate path is not a Train ship")
            path=alternate_root/relative
            if not path.is_file():
                raise FileNotFoundError(path)
            alternate_records.append(replace(scene,path=path))
            ship_keys.add(key)
        if ship_keys!=rows.keys():
            raise ValueError("Training loader must retain all original ship sources")
        self.alternate=SceneDataset(alternate_records,cfg,stats,True)
        self.channel_rng=None

    def __getitem__(self,index):
        scene=self.records[index]
        use_alternate=False
        if sum(scene.labels)==1:
            if self.channel_rng is None:
                # Private worker stream; SpecAugment/model/sampler RNG remain untouched.
                self.channel_rng=np.random.default_rng(torch.initial_seed() ^ 0x42484F50)
            use_alternate=bool(self.channel_rng.random()<AUGMENTATION["alternate_probability"])
        result=self.alternate[index] if use_alternate else super().__getitem__(index)
        result["path"]=str(scene.path)  # Coverage always tracks the original unique source.
        result["alternate_channel"]=use_alternate
        return result
