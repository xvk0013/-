"""Synthetic framework checks; never read generated data or real recordings."""
import argparse
import csv
import io
from pathlib import Path
import tempfile
import numpy as np
import scipy.signal
import soundfile as sf
import torch
from config import Config
from data import (FOLDER_LABELS, SceneDataset, load_records, read_waveform,
                  check_recording_split, fit_train_stats)
from model import M2Mamba, parameter_counts
from scan import selective_scan
from train import (EMA, evaluate, make_loader, make_optimizer, make_scaler, train_epoch,
                   capture_rng, restore_rng, seed_all)


def assert_checkpoint_equal(expected_model, loaded_model):
    """Saved parameters and persistent buffers must survive loading exactly."""
    expected, actual = expected_model.state_dict(), loaded_model.state_dict()
    if expected.keys() != actual.keys():
        raise AssertionError("Checkpoint state keys changed after loading")
    for name in expected:
        torch.testing.assert_close(
            actual[name], expected[name], rtol=0, atol=0,
            msg=f"Checkpoint parameter/buffer changed after loading: {name}")
    print("PASS: checkpoint parameters and buffers are exactly equal", flush=True)


def assert_eval_close(actual, expected, label):
    """FP32 GPU evaluation is numerically close, not necessarily bitwise identical.

    These tolerances apply only to computed outputs. Checkpoint tensor equality
    and restored RNG sequences retain zero tolerance.
    """
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise FloatingPointError(f"{label}: non-finite evaluation output")
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    max_error = float((actual - expected).abs().max())
    print(f"PASS: {label}; max_abs_error={max_error:.3e} "
          "(rtol=1e-5, atol=1e-6)", flush=True)


def check_scan(device, backend):
    torch.manual_seed(7)
    values = [torch.randn(2, 4, 7, device=device), torch.randn(2, 4, 7, device=device),
              -torch.arange(1, 4, device=device).float().repeat(4, 1),
              torch.randn(2, 3, 7, device=device), torch.randn(2, 3, 7, device=device),
              torch.randn(4, device=device), torch.randn(4, device=device)]
    args = [v.double().requires_grad_() for v in values]
    u, delta, a, b, c, d, bias = args
    dt = torch.nn.functional.softplus(delta + bias[None, :, None])
    # Independent closed-form expansion, rather than a second recurrence.
    reference = []
    for t in range(u.shape[-1]):
        state = torch.zeros(2, 4, 3, device=device, dtype=torch.float64)
        for k in range(t + 1):
            decay = torch.exp(dt[:, :, k+1:t+1].sum(-1)[:, :, None] * a)
            state = state + decay * dt[:, :, k, None] * b[:, None, :, k] * u[:, :, k, None]
        reference.append((state * c[:, None, :, t]).sum(-1) + d * u[:, :, t])
    reference = torch.stack(reference, -1)
    actual = selective_scan(u, delta, a, b, c, d, delta_bias=bias)
    torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-10)
    weights = torch.randn_like(actual)
    g1 = torch.autograd.grad((actual * weights).sum(), args, retain_graph=True)
    g2 = torch.autograd.grad((reference * weights).sum(), args)
    for one, two in zip(g1, g2):
        torch.testing.assert_close(one, two, rtol=1e-9, atol=1e-9)
    if backend == "cuda":
        xs = [v.clone().requires_grad_() for v in values]
        portable = selective_scan(*xs[:6], delta_bias=xs[6], backend="torch")
        native = selective_scan(*xs[:6], delta_bias=xs[6], backend="cuda")
        torch.testing.assert_close(native, portable, rtol=3e-4, atol=3e-4)
        weights = torch.randn_like(native)
        g1 = torch.autograd.grad((portable * weights).sum(), xs, retain_graph=True)
        g2 = torch.autograd.grad((native * weights).sum(), xs)
        for one, two in zip(g1, g2):
            torch.testing.assert_close(one, two, rtol=5e-4, atol=5e-4)
    print("PASS: scan output/gradient reference" + (" + CUDA parity" if backend == "cuda" else ""), flush=True)


def synthetic_dataset(root, cfg):
    fields = ["file_name", "split", "label_Tanker", "label_Cargo", "label_Tug",
              "recording1", "recording2", "sir_db"]
    rng = np.random.default_rng(42)
    time = np.arange(cfg.audio_len) / cfg.sample_rate
    for split in ("Train", "Val"):
        for i, (folder, labels) in enumerate(FOLDER_LABELS.items()):
            directory = root / folder / split
            (directory / "mix").mkdir(parents=True)
            wave = (2.5 * np.sin(2 * np.pi * (85 + 17 * i) * time)
                    + rng.normal(0, 0.1, cfg.audio_len)).astype(np.float32)
            sf.write(directory / "mix" / "sample.wav", wave, cfg.sample_rate, subtype="FLOAT")
            active = [j for j, present in enumerate(labels) if present]
            recs = [f"{split}_source_{j}.wav" for j in active]
            recs += ["none"] * (2 - len(recs))
            with (directory / "all_info.txt").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(fields)
                writer.writerow(["sample.wav", split, *labels, *recs, 12.0 if len(active) == 2 else "NaN"])
    # No Test directory or s1/s2/s3 files: accidental reads fail.


def check_optimizer_update(model, loader, optimizer, ema, scaler, device, cfg):
    """Allow bounded AMP scale calibration, but require a real finite update."""
    if len(loader) != 1:
        raise AssertionError("Smoke update expects exactly one synthetic batch")
    # Retry the same inputs/masks; no checkpoint or real-data training is involved.
    batch = next(iter(loader))
    initial_rng = capture_rng(loader)
    probe = model.classifier.single_scorer[-1].weight
    initial_probe = probe.detach().clone()
    attempts = 8 if scaler.is_enabled() else 1
    for attempt in range(1, attempts + 1):
        restore_rng(initial_rng, loader)
        result, step = train_epoch(model, [batch], optimizer, ema, scaler, device, cfg, 0)
        if step == 1:
            if result["optimizer_steps"] != 1 or result["skipped_amp_steps"] != 0:
                raise AssertionError("Inconsistent optimizer-update accounting")
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    if parameter.grad is None:
                        raise AssertionError(f"Missing gradient after update: {name}")
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise FloatingPointError(f"Non-finite gradient after update: {name}")
                    if not bool(torch.isfinite(parameter).all()):
                        raise FloatingPointError(f"Non-finite parameter after update: {name}")
            if torch.equal(initial_probe, probe.detach()):
                raise AssertionError("Optimizer reported an update but evidence weights did not change")
            print(f"PASS: {model.phase} optimizer update; attempt={attempt}, "
                  f"AMP scale={scaler.get_scale():g}", flush=True)
            return
        if not scaler.is_enabled() or result["skipped_amp_steps"] != 1:
            raise AssertionError("Optimizer did not update for a reason other than AMP overflow")
    raise FloatingPointError(
        f"{model.phase}: no finite optimizer update after {attempts} AMP calibration attempts "
        f"(scale={scaler.get_scale():g}). Run the same smoke command without --amp "
        "to distinguish mixed-precision overflow from a general gradient problem."
    )


def check_framework(device, backend, amp):
    cfg = Config(scan_backend=backend, batch_size=8, stats_samples=4, epochs=2, amp=amp,
                 dataset_format="v11", class_names=("Tanker", "Cargo", "Tug"))
    seed_all(cfg.seed)
    with tempfile.TemporaryDirectory(prefix="m2_synthetic_") as temporary:
        root = Path(temporary)
        synthetic_dataset(root, cfg)
        train = load_records(root, "Train", "single")
        val = load_records(root, "Val", "single")
        check_recording_split(train, val)
        assert len(train) == 4 and train[1].labels == (0, 1, 0)
        try:
            check_recording_split(train, train)
        except ValueError:
            pass
        else:
            raise AssertionError("Recording leakage was not rejected")
        path = train[1].path
        raw, _ = sf.read(path, dtype="float32")
        assert abs(raw).max() > 1 and sf.info(path).subtype == "FLOAT"
        expected = raw.astype(np.float64) / np.sqrt(np.mean(raw.astype(np.float64)**2))
        torch.testing.assert_close(read_waveform(path, cfg), torch.from_numpy(expected.astype(np.float32)))
        stats = fit_train_stats(train, cfg)
        ds = SceneDataset(train, cfg, stats, training=True)
        sample = ds[0]
        assert sample["features"].shape == (2, 513, 79)
        assert sample["demon"].shape == (1, 126)
        loader = make_loader(ds, cfg, True, device)
        model = M2Mamba(cfg, "single").to(device)
        print(f"Parameters: {parameter_counts(model)}", flush=True)
        shape = []
        hook = model.backbone.register_forward_hook(lambda _, inputs, output: shape.append(tuple(output.shape)))
        optimizer = make_optimizer(model, cfg)
        ema = EMA(model, cfg.ema_decay)
        scaler = make_scaler(cfg, device)
        print(f"AMP enabled={scaler.is_enabled()}, initial scale={scaler.get_scale():g}", flush=True)
        check_optimizer_update(model, loader, optimizer, ema, scaler, device, cfg)
        hook.remove()
        assert shape[-1][1:] == (256, 33, 20)
        assert all(p.grad is None for p in model.classifier.tc_head.parameters())
        assert all(p.grad is None for p in model.classifier.pair_scorer.parameters())
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, f"Missing gradient: {name}"
                assert bool(torch.isfinite(parameter.grad).all()), f"Bad gradient: {name}"
        assert model.backbone.stages[2].blocks[0].mixer.A_log.grad.abs().sum() > 0
        val_loader = make_loader(SceneDataset(val, cfg, stats), cfg, False, device)
        metrics = evaluate(ema.model, val_loader, device)
        assert metrics["n"] == 4 and metrics["single"]["n"] == 3
        stream = io.BytesIO()
        torch.save(ema.model.state_dict(), stream)
        stream.seek(0)
        joint = M2Mamba(cfg, "joint").to(device).eval()
        joint.load_state_dict(torch.load(stream, map_location=device, weights_only=True), strict=True)
        assert_checkpoint_equal(ema.model, joint)
        x, d = sample["features"][None].to(device), sample["demon"][None].to(device)
        with torch.no_grad():
            before = ema.model(x, d)["presence_logits"]
            outputs, after = joint(x, d), joint(x, d)
        assert_eval_close(outputs["presence_logits"], before, "single-to-joint evidence")
        assert_eval_close(after["set_log_probs"], outputs["set_log_probs"], "repeated joint evaluation")
        torch.testing.assert_close(outputs["set_log_probs"].exp().sum(1), torch.ones(1, device=device))
        joint_records = load_records(root, "Train", "joint")
        joint_loader = make_loader(SceneDataset(joint_records, cfg, stats, True), cfg, True, device)
        optimizer = make_optimizer(joint, cfg)
        joint_ema = EMA(joint, cfg.ema_decay)
        check_optimizer_update(joint, joint_loader, optimizer, joint_ema, scaler, device, cfg)
        assert joint.classifier.tc_head[-1].weight.grad.abs().sum() > 0
        assert joint.classifier.pair_scorer[-1].weight.grad.abs().sum() > 0
        assert all(torch.isfinite(p.grad).all() for p in joint.parameters() if p.grad is not None)
        state = capture_rng(joint_loader)
        first = torch.rand(4)
        restore_rng(state, joint_loader)
        torch.testing.assert_close(torch.rand(4), first, rtol=0, atol=0)
        restored = M2Mamba(cfg, "joint").to(device)
        restored.load_state_dict(joint.state_dict(), strict=True)
        assert_checkpoint_equal(joint, restored)
        restored_optimizer = make_optimizer(restored, cfg)
        restored_optimizer.load_state_dict(optimizer.state_dict())
        assert len(restored_optimizer.state) == len(optimizer.state)
        print("PASS: float32 I/O, labels, recording split, Train-only statistics, frontend shapes")
        print("PASS: single/joint gradients, frozen heads, EMA, checkpoint transfer, optimizer/RNG restore")
        print("PASS: no Test or source-component files accessed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scan-backend", choices=("torch", "cuda"), default="torch")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if args.scan_backend == "cuda" and device.type != "cuda":
        parser.error("CUDA scan needs --device cuda")
    check_scan(device, args.scan_backend)
    check_framework(device, args.scan_backend, args.amp)
    print("SYNTHETIC CHECK COMPLETE; no real data/training run was used.")


if __name__ == "__main__":
    main()
