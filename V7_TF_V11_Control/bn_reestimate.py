"""Existing Train-only BN reestimation, without the old diagnostic CLI."""
import torch
from torch import nn

@torch.no_grad()
def reestimate_bn(model, loader, device):
    """Update only BN buffers; all weights and stochastic layers stay frozen."""
    model.eval()
    layers = [module for module in model.modules()
              if isinstance(module, nn.modules.batchnorm._BatchNorm)
              and module.track_running_stats]
    if not layers:
        raise ValueError("The model has no BatchNorm running statistics to reestimate")
    original_momenta = [module.momentum for module in layers]
    seen = 0
    try:
        for module in layers:
            module.reset_running_stats()
            module.train()
        for step, batch in enumerate(loader, 1):
            size = len(batch["labels"])
            # Sample-weighted cumulative batch statistics, including the last batch.
            for module in layers:
                module.momentum = size / (seen + size)
            outputs = model(batch["features"].to(device, non_blocking=True),
                            batch["demon"].to(device, non_blocking=True))
            if not bool(torch.isfinite(outputs["presence_logits"]).all()):
                raise FloatingPointError("Non-finite logits during Train-only BN reestimation")
            seen += size
            if step == 1 or step % 25 == 0 or step == len(loader):
                print(f"  BN Train: {step}/{len(loader)} batches, {seen} samples", flush=True)
        for module in layers:
            if not bool(torch.isfinite(module.running_mean).all()
                        and torch.isfinite(module.running_var).all()):
                raise FloatingPointError("Non-finite reestimated BN statistics")
    finally:
        for module, momentum in zip(layers, original_momenta):
            module.momentum = momentum
        model.eval()
    return {"split": "Train", "samples": seen, "bn_layers": len(layers),
            "augmentation": False, "dropout": False, "drop_path": False,
            "method": "sample-weighted cumulative batch statistics; one shuffled pass"}
