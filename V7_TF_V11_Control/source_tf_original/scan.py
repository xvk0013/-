"""Selective state-space recurrence, with portable and optional CUDA backends.

The torch backend implements the same recurrence, not an RNN approximation.
State accumulation is float32 under AMP. A sequential scan is intentionally
used for numerical stability; it is slower than the fused CUDA extension.
"""
from functools import lru_cache
import torch
import torch.nn.functional as F


@lru_cache(maxsize=1)
def cuda_scan():
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        import selective_scan_cuda  # noqa: F401 -- verify compiled extension
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "CUDA scan is unavailable. Use --scan-backend torch, or install a "
            "compatible mamba-ssm + selective_scan_cuda build in a separate environment."
        ) from exc
    return selective_scan_fn


def selective_scan(u, delta, A, B, C, D, z=None, delta_bias=None,
                   delta_softplus=True, return_last_state=False, backend="torch"):
    if backend == "cuda":
        if not u.is_cuda:
            raise ValueError("CUDA scan requires CUDA input")
        return cuda_scan()(u.contiguous(), delta.contiguous(), A.float(),
                           B.contiguous(), C.contiguous(), D.float(), z=z,
                           delta_bias=delta_bias, delta_softplus=delta_softplus,
                           return_last_state=return_last_state)
    if backend != "torch":
        raise ValueError(f"Unknown scan backend: {backend}")
    if u.ndim != 3 or delta.shape != u.shape:
        raise ValueError("u and delta must have the same [batch, channels, length] shape")
    n_batch, n_channels, length = u.shape
    if B.shape != C.shape or B.shape != (n_batch, A.shape[1], length):
        raise ValueError("B/C must have shape [batch, state, length]")
    dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
    original_dtype = u.dtype
    with torch.autocast(device_type=u.device.type, enabled=False):
        uf, dt, af, bf, cf = (v.to(dtype) for v in (u, delta, A, B, C))
        if delta_bias is not None:
            dt = dt + delta_bias.to(dtype)[None, :, None]
        if delta_softplus:
            dt = F.softplus(dt)
        state = uf.new_zeros(n_batch, n_channels, af.shape[1])
        outputs = []
        for t in range(length):
            step = dt[:, :, t, None]
            state = (torch.exp(step * af[None]) * state
                     + step * bf[:, None, :, t] * uf[:, :, t, None])
            outputs.append((state * cf[:, None, :, t]).sum(-1))
        result = torch.stack(outputs, dim=-1) + uf * D.to(dtype)[None, :, None]
        if z is not None:
            result = result * F.silu(z.to(dtype))
        result = result.to(original_dtype)
    return (result, state) if return_last_state else result
