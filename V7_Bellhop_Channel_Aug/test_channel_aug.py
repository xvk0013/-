"""Focused synthetic checks; no real dataset inference or training."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
from scipy.fft import rfft
from v7_common import bootstrap, AUGMENTATION, BASELINE_SUMMARY
bootstrap()
import torch
from config import Config
from data import Scene, SceneDataset
from channel_data import propagate, alternate_index, source_identity, write_json
from channel_dataset import ChannelSceneDataset
from train_v7 import make_loader, seed_all
from model import M2Mamba, parameter_counts
from objective import loss_components
from source_pool_sampler import SourcePoolSampler
from run_channel_aug import training_command, PROTOCOL
from comparison import compare

torch.set_num_threads(2)

def fixture(root):
    original=root/"original";alternate=root/"alternate"
    baseline=json.loads(BASELINE_SUMMARY.read_text())
    cfg=Config(**baseline["config"]);cfg.scan_backend="torch";cfg.batch_size=2;cfg.workers=1
    records=[];entries=[];time=np.arange(80000)/16000
    for i,name in enumerate(cfg.class_names):
        folder=original/name/"Train";folder.mkdir(parents=True)
        path=folder/"original.wav"
        wave=np.sin(2*np.pi*(100+100*i)*time).astype(np.float32)
        sf.write(path,wave,16000,subtype="FLOAT")
        relative=f"{name}/Train/alternate.wav"
        (alternate/name/"Train").mkdir(parents=True)
        sf.write(alternate/relative,np.sin(2*np.pi*(450+50*i)*time),16000,subtype="FLOAT")
        row={"class_name":name,"split":"Train","file_name":"original.wav",
             "audio_path":f"{name}/Train/original.wav","raw_relative_path":f"{name}/record.wav",
             "start_sample":"1","stop_sample":"80000","source1_index":str(i+1)}
        with (folder/"all_info.txt").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(row),delimiter="\t")
            writer.writeheader();writer.writerow(row)
        entries.append(dict(key=row["audio_path"],source_identity=source_identity(row),
                            alternate_path=relative,original_range_km=1.,original_depth_m=5.,
                            alternate_range_km=2.,alternate_depth_m=10.))
        records.append(Scene(path,tuple(int(i==j) for j in range(3)),
                             (row["raw_relative_path"].casefold(),),float("nan")))
    noise=original/"noise.wav";sf.write(noise,np.cos(2*np.pi*1500*time),16000,subtype="FLOAT")
    records.insert(0,Scene(noise,(0,0,0),(),float("nan")))
    # Shuffled manifest is intentional: lookup must use source identity, not row number.
    write_json(alternate/"manifest.json",dict(augmentation=AUGMENTATION,source_dataset=str(original),entries=entries[::-1]))
    write_json(alternate/"summary.json",dict(status="complete",augmentation=AUGMENTATION))
    return original,alternate,records,cfg,baseline["training_record"]["stats"]

class Checks(unittest.TestCase):
    def test_linear_propagation_crop_and_zero_history(self):
        x=np.array([0.,0.,0.,1.,2.,3.,4.,5.])
        h=np.array([.6,.2,-.1,.05])
        result=propagate(x,rfft(h,n=16),16,3,5,3,gain=.5)
        np.testing.assert_allclose(result,np.convolve(x,h)[3:8]*.5,rtol=1e-6,atol=1e-7)
        with self.assertRaises(ValueError):
            propagate(x,rfft(h,n=8),8,3,5,3)

    def test_alternate_channel_is_distinct_and_deterministic(self):
        for original in range(39):
            for source in (1,17,17118):
                chosen=alternate_index(original,source)
                self.assertNotEqual(chosen,original)
                self.assertTrue(0<=chosen<39)
                self.assertEqual(chosen,alternate_index(original,source))

    def test_dataset_pairing_noise_and_original_eval(self):
        with tempfile.TemporaryDirectory() as temp:
            root,alt,records,cfg,stats=fixture(Path(temp))
            ds=ChannelSceneDataset(records,cfg,stats,root=root,alternate_root=alt)
            state=torch.get_rng_state()
            item=ds[0]
            self.assertFalse(item["alternate_channel"])
            # Noise does not initialize or draw from the channel stream.
            self.assertIsNone(ds.channel_rng)
            torch.set_rng_state(state)
            expected=SceneDataset(records,cfg,stats,True)[0]
            self.assertTrue(torch.equal(item["features"],expected["features"]))
            for i in range(1,4):
                item=ds[i]
                self.assertEqual(item["path"],str(records[i].path))
                self.assertEqual(tuple(item["features"].shape),(2,513,79))
                self.assertEqual(tuple(item["demon"].shape),(1,126))
                self.assertTrue(torch.equal(item["labels"],torch.tensor(records[i].labels)))
            with self.assertRaises(ValueError):
                ChannelSceneDataset(records,cfg,stats,False,root=root,alternate_root=alt)
            obj=json.loads((alt/"manifest.json").read_text())
            obj["entries"][0]["source_identity"][2]+=80000
            write_json(alt/"manifest.json",obj)
            with self.assertRaisesRegex(ValueError,"identity"):
                ChannelSceneDataset(records,cfg,stats,root=root,alternate_root=alt)

    def test_worker_rng_epoch_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root,alt,records,cfg,stats=fixture(Path(temp))
            # Repeated draws of each source are allowed; use the same original pool.
            def loader():
                ds=ChannelSceneDataset(records,cfg,stats,root=root,alternate_root=alt)
                return make_loader(ds,cfg,True,torch.device("cpu"))
            seed_all(42);first=loader()
            list(first)  # epoch one
            saved=first.generator.get_state()
            uninterrupted=list(first)  # epoch two
            seed_all(42);resumed=loader();resumed.generator.set_state(saved)
            resumed_batches=list(resumed)
            for a,b in zip(uninterrupted,resumed_batches):
                self.assertEqual(a["path"],b["path"])
                self.assertTrue(torch.equal(a["alternate_channel"],b["alternate_channel"]))
                self.assertTrue(torch.equal(a["features"],b["features"]))

    def test_fixed_sampler_budget_and_original_model_loss(self):
        baseline=json.loads(BASELINE_SUMMARY.read_text())
        cfg=Config(**baseline["config"]);cfg.scan_backend="torch"
        records=[Scene(Path(f"noise{i}.wav"),(0,0,0),(),float("nan")) for i in range(1080)]
        records += [Scene(Path(f"ship{i}.wav"),tuple(int(i==j) for j in range(3)),
                          (f"record{i}",),float("nan")) for i in range(3)]
        a=SourcePoolSampler(records,42);b=SourcePoolSampler(records,42)
        indices=list(a)
        self.assertEqual(indices,list(b));self.assertEqual(len(indices),6480)
        self.assertEqual(sum(sum(records[i].labels)==0 for i in indices),1080)
        torch.manual_seed(42)
        model=M2Mamba(cfg,"single").train()
        self.assertEqual(parameter_counts(model)["total"],2090723)
        out=model(torch.randn(2,2,513,79),torch.randn(2,1,126))
        loss,_=loss_components(model,out,torch.tensor([[1.,0,0],[0,0,0]]),["record",""],cfg)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertEqual(PROTOCOL["new_training_runs"],1)
        command=training_command()
        self.assertEqual(command[command.index("--epochs")+1],"60")
        self.assertTrue(all(v["augmented_minus_baseline_pp"]==0
                            for fields in compare(baseline,baseline).values() for v in fields.values()))

if __name__=="__main__":
    unittest.main()
