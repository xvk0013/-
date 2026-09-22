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
from config import Config, checkpoint_config
from data import SceneDataset, load_records, check_recording_split, fit_train_stats
from model import M2Mamba, parameter_counts
from objective import loss_components
from feature_mixup import make_feature_mixup_plan
from metrics import summarize, format_noise_metrics, selection_score
from source_pool_sampler import SourcePoolSampler as RecordingBalancedSampler


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


def make_loader(dataset, cfg, training, device, *, record_balanced=False):
    generator = torch.Generator().manual_seed(cfg.seed + int(not training))
    if record_balanced and not training:
        raise ValueError("Recording-balanced sampling must not be used for evaluation")
    sampler = RecordingBalancedSampler(dataset.records, cfg.seed) if record_balanced else None
    return DataLoader(dataset, batch_size=cfg.batch_size,
                      shuffle=training and sampler is None, sampler=sampler,
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
    mixup_batches = mixup_ships = 0
    for step, batch in enumerate(loader, 1):
        for group in optimizer.param_groups:
            group["lr"] = lr_for_step(global_step, len(loader), cfg)
        x = batch["features"].to(device, non_blocking=True)
        d = batch["demon"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        # Fix plan/targets outside AMP retries. No extra backbone pass is needed.
        # A private step-keyed RNG also reproduces the plan after epoch resume.
        mixup = make_feature_mixup_plan(batch["labels"], batch["recording"], cfg, global_step, device)
        soft_targets = mixup.apply(y) if mixup is not None else None
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        buffers = [(buf, buf.detach().clone()) for buf in model.buffers()]
        while True:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                outputs = model(x, d, mixup=mixup) if mixup is not None else model(x, d)
                loss, pieces = loss_components(model, outputs, y, batch["recording"], cfg,
                                                soft_targets=soft_targets)
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
                break
            skipped += 1
            consecutive_skips += 1
            print(f"  AMP retry same batch: scale {old_scale:g} -> {scaler.get_scale():g} "
                  f"(attempt={consecutive_skips})", flush=True)
            with torch.no_grad():
                for buf, saved in buffers:
                    buf.copy_(saved)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
            if consecutive_skips >= 8:
                raise FloatingPointError("AMP failed 8 attempts on the same batch")
        n = len(y)
        if mixup is not None:
            mixup_batches += 1
            mixup_ships += mixup.rows.numel()
        n_samples += n
        sums["loss"] += float(loss.detach()) * n
        for name in ("bce", "nll", "cr", "single_ce"):
            sums[name] += float(pieces[name]) * n
        if step == 1 or step % 100 == 0:
            print(f"  {step}/{len(loader)} loss={sums['loss']/n_samples:.4f}", flush=True)
    return {**{k: v / n_samples for k, v in sums.items()},
            "feature_mixup_batches": mixup_batches, "feature_mixup_ships": mixup_ships,
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
    result = summarize(torch.cat(predictions), y, torch.cat(sirs),
                       class_names=loader.dataset.cfg.class_names)
    result.update(bce=bce / len(y), nll=nll / len(y) if model.phase == "joint" else None)
    return result


def load_checkpoint(path):
    # Load only this project's own/trusted checkpoints: optimizer/RNG state is included.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    checkpoint["config"] = checkpoint_config(checkpoint["config"])
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
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset directory; this experiment uses the new named-class 5-second slices")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--noise-data-dir", type=Path, help="Existing v11 dataset containing noise/{split}/mix")
    parser.add_argument("--mapping-dir", type=Path, help="deepship_16k_5s mapping root; defaults to a sibling of data-dir")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--init-checkpoint", type=Path, help="single best.pt for a new joint run")
    source.add_argument("--resume", type=Path, help="last.pt to resume the same run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scan-backend", choices=("torch", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--head-type", choices=("query", "gap", "freq4"))
    parser.add_argument("--single-ce-weight", type=float, help="Override ship CE weight for a new controlled run")
    parser.add_argument("--band-fusion", choices=("early", "late_shared"))
    parser.add_argument("--local-contrast-bins", type=int)
    parser.add_argument("--spectral-aug-prob", type=float)
    parser.add_argument("--spectral-aug-db", type=float)
    parser.add_argument("--feature-mixup-prob", type=float, help="Training batch probability; ship features only")
    parser.add_argument("--feature-mixup-alpha", type=float, help="Symmetric Beta distribution parameter")
    parser.add_argument("--weight-decay", type=float, help="Override AdamW weight decay for a new controlled run")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.init_checkpoint:
        parser.error("Cannot initialize from another experiment; use --resume for this run")
    if args.phase != "single" or args.data_dir.name not in ("A_original", "B_expanded", "C_time_spread"):
        parser.error("Use single phase with A_original, B_expanded or C_time_spread")
    if not (args.data_dir.parent / "COMPLETE.txt").is_file():
        parser.error("Run EXPORT_SOURCE_POOL_AB.m to completion first")
    if args.data_dir.name == "C_time_spread" and not (args.data_dir / "COMPLETE.txt").is_file():
        parser.error("Run prepare_source_pool_c.py to completion first")
    checkpoint = load_checkpoint(args.resume or args.init_checkpoint) if (args.resume or args.init_checkpoint) else None
    cfg = Config(**checkpoint["config"]) if checkpoint else Config()
    for key in ("scan_backend", "epochs", "batch_size", "workers", "seed", "single_ce_weight", "head_type", "weight_decay", "spectral_aug_prob", "spectral_aug_db", "local_contrast_bins", "band_fusion", "feature_mixup_prob", "feature_mixup_alpha"):
        value = getattr(args, key)
        if value is not None:
            if args.resume and key not in ("workers", "scan_backend") and value != getattr(cfg, key):
                parser.error(f"--resume cannot change {key}; start a new run instead")
            setattr(cfg, key, value)
    for key in ("noise_data_dir", "mapping_dir"):
        value = getattr(args, key)
        if value is not None:
            value = str(value.expanduser().resolve())
            if checkpoint and value != getattr(cfg, key):
                parser.error(f"Checkpoint fixes {key}; start a new run to change it")
            setattr(cfg, key, value)
    if cfg.dataset_format in ("deepship_ptt_slices", "deepship_v6_slices"):
        if args.phase != "single":
            parser.error("Slice experiments support the single-target phase only")
        if not cfg.noise_data_dir:
            parser.error("--noise-data-dir is required to retain noise examples")
        if cfg.dataset_format == "deepship_ptt_slices" and not cfg.mapping_dir:
            cfg.mapping_dir = str(args.data_dir.resolve().parent / "deepship_16k_5s")
    if args.no_amp:
        cfg.amp = False
    if cfg.epochs != 60 or cfg.batch_size != 32:
        parser.error("Controlled comparison fixes epochs=60 and batch-size=32")
    if cfg.dataset_format != "deepship_v6_slices" or cfg.single_train_sampler != "record_balanced":
        parser.error("Controlled comparison requires named-class data and recording-balanced sampling")
    cfg.patience = cfg.epochs + 1  # Full and equal update budget; no early stop.
    if cfg.head_type in ("gap", "freq4") and not checkpoint:
        cfg.ortho_weight = 0.0
    if not math.isfinite(cfg.weight_decay) or cfg.weight_decay < 0:
        parser.error("weight_decay must be finite and nonnegative")
    if cfg.spectral_aug_prob > 0 and cfg.workers == 0:
        parser.error("Spectral experiment requires workers > 0 for reproducible per-epoch RNG")
    cfg.validate()
    print(f"Train feature Mixup: batch p={cfg.feature_mixup_prob:g}, alpha={cfg.feature_mixup_alpha:g}; "
          "different-recording ships only, noise unchanged; disabled for evaluation", flush=True)
    print(f"Band fusion={cfg.band_fusion}", flush=True)
    print(f"Input channels={cfg.input_channels}; local contrast bins={cfg.local_contrast_bins}", flush=True)
    print(f"Train spectral augmentation: p={cfg.spectral_aug_prob:g}, max dB={cfg.spectral_aug_db:g}", flush=True)
    print(f"Classifier head={cfg.head_type}; query orthogonality={cfg.ortho_weight:g}", flush=True)
    print(f"Ship CE weight={cfg.single_ce_weight:g}; CR={cfg.cr_weight:g}", flush=True)
    print(f"AdamW weight decay={cfg.weight_decay:g}", flush=True)
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
    train_records = load_records(args.data_dir, "Train", args.phase, cfg)
    val_records = load_records(args.data_dir, "Val", args.phase, cfg)
    check_recording_split(train_records, val_records)
    print(f"Input={cfg.dataset_format}; class order={tuple(cfg.class_names)}", flush=True)
    selection_name = cfg.selection_metric if args.phase == "single" else "overall_emr"
    print(f"Best checkpoint: EMA {selection_name}; online Val comparison={cfg.compare_online_val}", flush=True)
    for split_name, records in (("Train", train_records), ("Val", val_records)):
        counts = {name: sum(scene.labels[i] for scene in records)
                  for i, name in enumerate(cfg.class_names)}
        counts["noise"] = sum(sum(scene.labels) == 0 for scene in records)
        print(f"{split_name} counts={counts}", flush=True)
    if not args.resume:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    stats = checkpoint["stats"] if checkpoint else fit_train_stats(train_records, cfg)
    train_loader = make_loader(SceneDataset(train_records, cfg, stats, True), cfg, True, device,
                               record_balanced=(args.phase == "single"
                                                and cfg.single_train_sampler == "record_balanced"))
    val_loader = make_loader(SceneDataset(val_records, cfg, stats), cfg, False, device)
    seed_all(cfg.seed)  # Identical initialization regardless of statistics fitting.
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
        if "source_pool_seen" in checkpoint:
            train_loader.sampler.seen = set(checkpoint["source_pool_seen"])
        else:
            for completed_epoch in range(start_epoch):
                train_loader.sampler.set_epoch(completed_epoch)
                list(train_loader.sampler)
        if global_step != start_epoch * len(train_loader):
            parser.error("Saved run has an incompatible update budget")
        restore_rng(checkpoint["rng"], train_loader)
        print(f"Resuming after epoch {start_epoch}; next={start_epoch + 1}", flush=True)
    config_record = {"config": cfg.to_dict(), "phase": args.phase, "data_dir": str(args.data_dir.resolve()),
                     "train_samples": len(train_records), "val_samples": len(val_records),
                     "stats": stats, "parameters": parameter_counts(model),
                     "source_pool_protocol": {"ships_per_class": 1800, "noise_per_epoch": 1080,
                                              "batches_per_epoch": len(train_loader), "early_stop": False}}
    (args.output_dir / "config.json").write_text(json.dumps(config_record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"M2 Mamba: {parameter_counts(model)}, phase={args.phase}, scan={cfg.scan_backend}", flush=True)
    print(f"DEMON features enabled={cfg.use_demon_features}; "
          f"carrier={cfg.demon_bp_low}-{cfg.demon_bp_high} Hz; "
          f"modulation=0-{cfg.demon_max_hz} Hz; "
          f"single sampler={cfg.single_train_sampler}", flush=True)
    print(f"Train={len(train_records)}, Val={len(val_records)}; Test NOT LOADED", flush=True)
    if isinstance(train_loader.sampler, RecordingBalancedSampler):
        print("Source-pool A/B: each epoch 1800/class + 1080 noise; 5:1; 203 batches; 60 full epochs", flush=True)
    if cfg.scan_backend == "torch":
        print("Portable exact scan enabled; training throughput has not been benchmarked.", flush=True)
    for epoch in range(start_epoch, cfg.epochs):
        if stale >= cfg.patience:
            break
        if isinstance(train_loader.sampler, RecordingBalancedSampler):
            train_loader.sampler.set_epoch(epoch)
        train_metrics, global_step = train_epoch(model, train_loader, optimizer, ema, scaler, device, cfg, global_step)
        if train_metrics["optimizer_steps"] != len(train_loader):
            raise RuntimeError("Unexpected successful optimizer update count")
        val_metrics = evaluate(ema.model, val_loader, device)
        # Read-only online comparison on the same Val; selection still uses EMA.
        online_val_metrics = evaluate(model, val_loader, device) if cfg.compare_online_val else None
        score = selection_score(val_metrics, args.phase, cfg)
        improved = score > best
        best, stale = (score, 0) if improved else (best, stale + 1)
        row = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics,
               "val_online": online_val_metrics, "selection_metric": selection_name,
               "selection_score": score, "best": best, "lr": optimizer.param_groups[0]["lr"]}
        row["source_pool_coverage"] = train_loader.sampler.coverage
        if cfg.feature_mixup_prob > 0:
            print(f"Feature Mixup: {train_metrics['feature_mixup_batches']}/{len(train_loader)} batches; "
                  f"{train_metrics['feature_mixup_ships']} ship rows", flush=True)
        print(f"Source coverage: {train_loader.sampler.coverage}", flush=True)
        with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        payload = {"schema_version": 1, "config": cfg.to_dict(), "phase": args.phase,
                   "data_dir": str(args.data_dir.resolve()), "stats": stats,
                   "epoch": epoch + 1, "best": best, "stale": stale, "global_step": global_step,
                   "model": model.state_dict(), "ema": ema.model.state_dict(),
                   "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                   "rng": capture_rng(train_loader), "val": val_metrics, "val_online": online_val_metrics}
        payload["source_pool_seen"] = sorted(train_loader.sampler.seen)
        payload["source_pool_amp_policy"] = "retry_same_batch_restore_buffers_and_rng"
        save_checkpoint(args.output_dir / "last.pt", payload)
        if improved:
            save_checkpoint(args.output_dir / "best.pt", payload)
        recalls = val_metrics["single"]["recall"]
        print(f"Epoch {epoch+1}: EMA Val EMR={val_metrics['emr']:.6f}, "
              f"single EMR={val_metrics['single']['emr']:.6f}, "
              f"single Macro-F1={val_metrics['single']['macro_f1']:.6f}, "
              f"recall={recalls}; {format_noise_metrics(val_metrics)}", flush=True)
        if online_val_metrics is not None:
            print(f"  Online: Val EMR={online_val_metrics['emr']:.6f}, "
                  f"single EMR={online_val_metrics['single']['emr']:.6f}, "
                  f"single Macro-F1={online_val_metrics['single']['macro_f1']:.6f}, "
                  f"recall={online_val_metrics['single']['recall']}; "
                  f"{format_noise_metrics(online_val_metrics)}", flush=True)
    print(f"COMPLETE: best EMA {selection_name}={best:.6f}; Test NOT LOADED", flush=True)


if __name__ == "__main__":
    main()
