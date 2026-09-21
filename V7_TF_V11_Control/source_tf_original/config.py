"""TF backbone control configuration; data paths are CLI arguments."""
from dataclasses import asdict, dataclass
import math


@dataclass
class Config:
    schema_version: int = 1
    architecture: str = "M2-Acoustic-TFConvNeXt-Small"
    class_names: tuple = ("Tanker", "Cargo", "Tug")
    sample_rate: int = 16000
    audio_len: int = 80000
    feature_type: str = "dual"
    normalization_mode: str = "per_channel"
    lofar_n_fft: int = 8192
    lofar_hop: int = 1024
    lofar_win: int = 8192
    lofar_n_bins: int = 513
    wideband_n_fft: int = 2048
    wideband_hop: int = 1024
    wideband_freq_lo_hz: float = 1000.0
    wideband_freq_max_hz: float = 4000.0
    demon_mode: str = "demon"
    demon_bp_low: int = 50
    demon_bp_high: int = 1000
    demon_lp_cutoff: int = 100
    demon_max_hz: int = 50
    demon_n_bins: int = 128
    demon_remove_dc: bool = True
    demon_encoder_dims: tuple = (32, 64, 128)
    demon_feat_dim: int = 128
    dims: tuple = (32, 64, 128, 256)
    depths: tuple = (1, 1, 4, 2)
    tf_time_kernel: int = 31
    tf_freq_kernel: int = 11
    tf_expand_ratio: int = 4
    # Retained for entry-point compatibility; TF does not use attention windows.
    heads: tuple = (1, 2, 4, 8)
    windows: tuple = (7, 7)
    mlp_ratio: float = 2.0
    drop_path: float = 0.10
    query_dim: int = 192
    query_heads: int = 4
    query_dropout: float = 0.20
    query_attn_dropout: float = 0.10
    # TF uses standard PyTorch convolutions on the selected device, without scan.
    scan_backend: str = "torch"
    batch_size: int = 32
    epochs: int = 60
    patience: int = 20
    lr: float = 3e-4
    min_lr: float = 1e-6
    warmup_epochs: int = 3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    ema_decay: float = 0.999
    amp: bool = True
    amp_init_scale: float = 1024.0
    workers: int = 0
    seed: int = 42
    presence_weight: float = 0.5
    single_ce_weight: float = 0.2
    ortho_weight: float = 0.1
    cr_weight: float = 0.0
    cr_tau: float = 0.1
    freq_masks: int = 2
    freq_mask_width: int = 24
    stats_samples: int = 500

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if self.architecture != "M2-Acoustic-TFConvNeXt-Small":
            raise ValueError("This directory requires a TF-control config/checkpoint")
        if (self.tf_time_kernel < 1 or self.tf_time_kernel % 2 != 1
                or self.tf_freq_kernel < 1 or self.tf_freq_kernel % 2 != 1
                or self.tf_expand_ratio < 1):
            raise ValueError("TF kernels must be positive and odd; expansion must be positive")
        if tuple(self.class_names) != ("Tanker", "Cargo", "Tug"):
            raise ValueError("Class order must remain Tanker/Cargo/Tug")
        if self.scan_backend not in ("torch", "cuda"):
            raise ValueError("scan_backend must be torch or cuda")
        if not (len(self.dims) == len(self.depths) == len(self.heads) == 4):
            raise ValueError("Exactly four backbone stages are required")
        if any(d < 1 for d in self.depths) or len(self.windows) != 2:
            raise ValueError("Invalid stage depths/windows")
        if not math.isfinite(self.amp_init_scale) or self.amp_init_scale <= 0:
            raise ValueError("amp_init_scale must be finite and positive")
        if not math.isfinite(self.single_ce_weight) or self.single_ce_weight < 0:
            raise ValueError("single_ce_weight must be finite and nonnegative")
        if self.epochs < 1 or self.batch_size < 1 or self.workers < 0:
            raise ValueError("Invalid training budget")
