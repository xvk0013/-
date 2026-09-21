"""Train/Val only. Test is evaluated through a separate explicit entry."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import numpy as np
import scipy.signal  # initialize scipy before torch on the current Windows environment
import torch
from torch.utils.data import DataLoader
from config import Config
from data import SceneDataset, load_records, check_recording_split, fit_train_stats
from model import M2Mamba, parameter_counts
from objective import loss_components
from metrics import summarize


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def make_loader(dataset, cfg, training, device):
    generator = torch.Generator().manual_seed(cfg.seed + int(not training))
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=training,
                      num_workers=cfg.workers, pin_memory=device.type == "cuda",
                      worker_init_fn=seed_worker, generator=generator,
                      persistent_workers=False, drop_last=False)


class EMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for target, value in zip(self.model.parameters(), model.parameters()):
            target.lerp_(value.detach(), 1 - self.decay)
        for target, value in zip(self.model.buffers(), model.buffers()):
            target.copy_(value)


def make_optimizer(model, cfg):
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (no_decay if getattr(parameter, "_no_weight_decay", False) else decay).append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)


def make_scaler(cfg, device):
    """Shared by training and smoke checks; retain dynamic overflow detection."""
    return torch.amp.GradScaler(
        "cuda", enabled=cfg.amp and device.type == "cuda",
        init_scale=cfg.amp_init_scale)


def lr_for_step(step, steps_per_epoch, cfg):
    total = cfg.epochs * steps_per_epoch
    warmup = min(cfg.warmup_epochs * steps_per_epoch, total)
    if warmup and step < warmup:
        return cfg.lr * (step + 1) / warmup
    fraction = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup - 1)))
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * fraction))


def train_epoch(model, loader, optimizer, ema, scaler, device, cfg, global_step):
    model.train()
    sums = {"loss": 0.0, "bce": 0.0, "nll": 0.0, "cr": 0.0, "single_ce": 0.0}
    n_samples = skipped = consecutive_skips = 0
    starting_step = global_step
    for step, batch in enumerate(loader, 1):
        for group in optimizer.param_groups:
            group["lr"] = lr_for_step(global_step, len(loader), cfg)
        x = batch["features"].to(device, non_blocking=True)
        d = batch["demon"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=scaler.is_enabled()):
            outputs = model(x, d)
            loss, pieces = loss_components(model, outputs, y, batch["recording"], cfg)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite training loss")
        old_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip,
                                       error_if_nonfinite=not scaler.is_enabled())
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            ema.update(model)
            global_step += 1
            consecutive_skips = 0
        else:
            skipped += 1
            consecutive_skips += 1
            print(f"  AMP skipped update: scale {old_scale:g} -> {scaler.get_scale():g} "
                  f"(non-finite scaled gradients; consecutive={consecutive_skips})", flush=True)
            if consecutive_skips >= 8:
                raise FloatingPointError(
                    "AMP skipped 8 consecutive updates. Persistent non-finite gradients "
                    "require investigation; no precision/backend fallback was performed."
                )
        n = len(y)
        n_samples += n
        sums["loss"] += float(loss.detach()) * n
        for name in ("bce", "nll", "cr", "single_ce"):
            sums[name] += float(pieces[name]) * n
        if step == 1 or step % 100 == 0:
            print(f"  {step}/{len(loader)} loss={sums['loss']/n_samples:.4f}", flush=True)
    return {**{k: v / n_samples for k, v in sums.items()},
            "skipped_amp_steps": skipped, "optimizer_steps": global_step - starting_step,
            "amp_scale": scaler.get_scale()}, global_step


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, labels, sirs = [], [], []
    bce = nll = 0.0
    for batch in loader:
        y = batch["labels"].to(device)
        # Evaluation uses float32, independently of training AMP.
        outputs = model(batch["features"].to(device), batch["demon"].to(device))
        if not bool(torch.isfinite(outputs["presence_logits"]).all()):
            raise FloatingPointError("Non-finite evaluation logits")
        predictions.append(model.decode(outputs).cpu())
        labels.append(y.cpu())
        sirs.append(batch["sir_db"].cpu())
        bce += float(torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["presence_logits"], y)) * len(y)
        if model.phase == "joint":
            from m2_components import multihot_to_set_targets
            nll += float(torch.nn.functional.nll_loss(
                outputs["set_log_probs"], multihot_to_set_targets(y))) * len(y)
    if not labels:
        raise ValueError("Empty evaluation loader")
    y = torch.cat(labels)
    result = summarize(torch.cat(predictions), y, torch.cat(sirs))
    result.update(bce=bce / len(y), nll=nll / len(y) if model.phase == "joint" else None)
    return result


def load_checkpoint(path):
    # Load only this project's own/trusted checkpoints: optimizer/RNG state is included.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    return checkpoint


def capture_rng(loader):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader": loader.generator.get_state()}


def restore_rng(state, loader):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    loader.generator.set_state(state["loader"])


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("single", "joint"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="Completed v11 dataset directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--init-checkpoint", type=Path, help="single best.pt for a new joint run")
    source.add_argument("--resume", type=Path, help="last.pt to resume the same run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scan-backend", choices=("torch", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.resume or args.init_checkpoint) if (args.resume or args.init_checkpoint) else None
    cfg = Config(**checkpoint["config"]) if checkpoint else Config()
    for key in ("scan_backend", "epochs", "batch_size", "workers", "seed"):
        value = getattr(args, key)
        if value is not None:
            if args.resume and key not in ("workers", "scan_backend") and value != getattr(cfg, key):
                parser.error(f"--resume cannot change {key}; start a new run instead")
            setattr(cfg, key, value)
    if args.no_amp:
        cfg.amp = False
    cfg.validate()
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("This framework supports cpu/cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; select --device cpu")
    if cfg.scan_backend == "cuda":
        if device.type != "cuda":
            parser.error("--scan-backend cuda requires --device cuda")
        from scan import cuda_scan
        cuda_scan()
    if checkpoint:
        if Path(checkpoint["data_dir"]).resolve() != args.data_dir.resolve():
            parser.error("Checkpoint belongs to another dataset; do not reuse its normalization statistics")
        if args.init_checkpoint and (args.phase != "joint" or checkpoint["phase"] != "single"):
            parser.error("--init-checkpoint requires single checkpoint -> joint phase")
        if args.resume and checkpoint["phase"] != args.phase:
            parser.error("--resume cannot change training phase")
        if args.resume and args.output_dir.resolve() != args.resume.parent.resolve():
            parser.error("--resume must use the original --output-dir")
    seed_all(cfg.seed)
    # No Test metadata, waveform, normalization or evaluation is loaded here.
    train_records = load_records(args.data_dir, "Train", args.phase)
    val_records = load_records(args.data_dir, "Val", args.phase)
    check_recording_split(train_records, val_records)
    if not args.resume:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    stats = checkpoint["stats"] if checkpoint else fit_train_stats(train_records, cfg)
    train_loader = make_loader(SceneDataset(train_records, cfg, stats, True), cfg, True, device)
    val_loader = make_loader(SceneDataset(val_records, cfg, stats), cfg, False, device)
    model = M2Mamba(cfg, args.phase).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"] if args.resume else checkpoint["ema"], strict=True)
    optimizer = make_optimizer(model, cfg)
    ema = EMA(model, cfg.ema_decay)
    scaler = make_scaler(cfg, device)
    start_epoch, best, stale, global_step = 0, -1.0, 0, 0
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema.model.load_state_dict(checkpoint["ema"], strict=True)
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, best, stale, global_step = (checkpoint[k] for k in ("epoch", "best", "stale", "global_step"))
        restore_rng(checkpoint["rng"], train_loader)
    config_record = {"config": cfg.to_dict(), "phase": args.phase, "data_dir": str(args.data_dir.resolve()),
                     "train_samples": len(train_records), "val_samples": len(val_records),
                     "stats": stats, "parameters": parameter_counts(model)}
    (args.output_dir / "config.json").write_text(json.dumps(config_record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{cfg.architecture}: {parameter_counts(model)}, phase={args.phase}; TF convolutions", flush=True)
    print(f"Train={len(train_records)}, Val={len(val_records)}; Test NOT LOADED", flush=True)
    for epoch in range(start_epoch, cfg.epochs):
        if stale >= cfg.patience:
            break
        train_metrics, global_step = train_epoch(model, train_loader, optimizer, ema, scaler, device, cfg, global_step)
        val_metrics = evaluate(ema.model, val_loader, device)
        score = val_metrics["single"]["macro_f1"] if args.phase == "single" else val_metrics["emr"]
        improved = score > best
        best, stale = (score, 0) if improved else (best, stale + 1)
        row = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics,
               "selection_score": score, "best": best, "lr": optimizer.param_groups[0]["lr"]}
        with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        payload = {"schema_version": 1, "config": cfg.to_dict(), "phase": args.phase,
                   "data_dir": str(args.data_dir.resolve()), "stats": stats,
                   "epoch": epoch + 1, "best": best, "stale": stale, "global_step": global_step,
                   "model": model.state_dict(), "ema": ema.model.state_dict(),
                   "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                   "rng": capture_rng(train_loader), "val": val_metrics}
        save_checkpoint(args.output_dir / "last.pt", payload)
        if improved:
            save_checkpoint(args.output_dir / "best.pt", payload)
        recalls = val_metrics["single"]["recall"]
        print(f"Epoch {epoch+1}: Val EMR={val_metrics['emr']:.6f}, "
              f"single Macro-F1={val_metrics['single']['macro_f1']:.6f}, "
              f"recall={recalls}, noise FA={val_metrics['noise_false_alarm']:.4f}", flush=True)
    print(f"COMPLETE: best selection score={best:.6f}; Test NOT LOADED", flush=True)


if __name__ == "__main__":
    main()
