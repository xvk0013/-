"""Copied V7 runtime configuration for the copied V11 TF model."""
from dataclasses import asdict, dataclass, field
import copy
from v7_common import TF_STRUCTURE
import math


@dataclass
class Config:
    schema_version: int = 1
    architecture: str = "M2-Acoustic-TFConvNeXt-Small"
    tf_structure: dict = field(default_factory=lambda: copy.deepcopy(TF_STRUCTURE))
    class_names: tuple = ("Tanker", "Cargo", "Tug")
    dataset_format: str = "deepship_v6_slices"
    noise_data_dir: str = ""
    mapping_dir: str = ""
    ptt_noise_fraction: float = 0.20
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
    use_demon_features: bool = True
    dims: tuple = (32, 64, 128, 256)
    depths: tuple = (1, 1, 4, 2)
    tf_time_kernel: int = 31
    tf_freq_kernel: int = 11
    tf_expand_ratio: int = 4
    heads: tuple = (1, 2, 4, 8)
    windows: tuple = (7, 7)
    mlp_ratio: float = 2.0
    drop_path: float = 0.10
    head_type: str = "query"
    query_dim: int = 192
    query_heads: int = 4
    query_dropout: float = 0.20
    query_attn_dropout: float = 0.10
    scan_backend: str = "torch"
    batch_size: int = 32
    epochs: int = 60
    patience: int = 20
    selection_metric: str = "single_emr"
    compare_online_val: bool = True
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
    single_train_sampler: str = "record_balanced"
    presence_weight: float = 0.5
    single_ce_weight: float = 0.2
    ortho_weight: float = 0.0
    cr_weight: float = 0.0
    cr_tau: float = 0.1
    freq_masks: int = 2
    freq_mask_width: int = 24
    spectral_aug_prob: float = 0.0
    spectral_aug_db: float = 3.0
    feature_mixup_prob: float = 0.0  # probability per training batch; old runs stay disabled
    feature_mixup_alpha: float = 0.2
    band_fusion: str = "early"  # late_shared: separate stems, shared trunk, GAP concatenation
    local_contrast_bins: int = 0  # 0 preserves the original two-channel frontend
    stats_samples: int = 500

    @property
    def input_channels(self):
        return 3 if self.local_contrast_bins else 2

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if self.architecture != "M2-Acoustic-TFConvNeXt-Small" or self.tf_structure != TF_STRUCTURE:
            raise ValueError("Use this copied original TF architecture/configuration")
        if (tuple(self.dims) != tuple(TF_STRUCTURE["backbone_dims"])
                or tuple(self.depths) != tuple(TF_STRUCTURE["backbone_blocks"])
                or self.query_dim != TF_STRUCTURE["query_dim"]
                or self.head_type != "query" or self.scan_backend != "torch"
                or self.query_heads != TF_STRUCTURE["query_num_heads"]
                or self.query_dropout != TF_STRUCTURE["query_dropout"]
                or self.query_attn_dropout != TF_STRUCTURE["query_attn_dropout"]
                or self.demon_feat_dim != TF_STRUCTURE["demon_feat_dim"]
                or tuple(self.demon_encoder_dims) != tuple(TF_STRUCTURE["demon_encoder_dims"])
                or self.tf_time_kernel != TF_STRUCTURE["tf_time_kernel"]
                or self.tf_freq_kernel != TF_STRUCTURE["tf_freq_kernel"]
                or self.tf_expand_ratio != TF_STRUCTURE["tf_expand_ratio"]
                or not self.use_demon_features):
            raise ValueError("TF/Query/DEMON structure must match the preserved original model")
        if not math.isfinite(self.feature_mixup_prob) or not 0 <= self.feature_mixup_prob <= 1:
            raise ValueError("feature_mixup_prob must be in [0, 1]")
        if not math.isfinite(self.feature_mixup_alpha) or self.feature_mixup_alpha <= 0:
            raise ValueError("feature_mixup_alpha must be finite and positive")
        if self.feature_mixup_prob > 0 and (self.head_type != "gap" or self.band_fusion != "early"
                                           or self.local_contrast_bins):
            raise ValueError("Feature Mixup requires early-fusion GAP and the original two-channel frontend")
        if self.band_fusion not in ("early", "late_shared"):
            raise ValueError("Unknown band_fusion")
        if self.band_fusion == "late_shared" and (self.head_type != "gap" or self.local_contrast_bins):
            raise ValueError("late_shared requires GAP and original two-channel frontend")
        k = self.local_contrast_bins
        if not isinstance(k, int) or (k != 0 and (k < 3 or k % 2 == 0 or k > self.lofar_n_bins)):
            raise ValueError("local_contrast_bins must be zero or an odd window in [3, lofar_n_bins]")
        if not math.isfinite(self.spectral_aug_prob) or not 0 <= self.spectral_aug_prob <= 1:
            raise ValueError("spectral_aug_prob must be in [0, 1]")
        if not math.isfinite(self.spectral_aug_db) or not 0 <= self.spectral_aug_db <= 6:
            raise ValueError("spectral_aug_db must be in [0, 6]")
        if self.head_type not in ("query", "gap", "freq4"):
            raise ValueError("Unknown classifier head")
        if self.head_type in ("gap", "freq4") and (self.ortho_weight != 0 or self.cr_weight != 0):
            raise ValueError("GAP ablation has no class queries; ortho/CR must be zero")
        if self.single_train_sampler not in ("scene_shuffle", "record_balanced"):
            raise ValueError("Unknown single-target training sampler")
        if not isinstance(self.use_demon_features, bool):
            raise ValueError("use_demon_features must be boolean")
        expected_classes = {
            "v11": ("Tanker", "Cargo", "Tug"),
            "deepship_ptt_slices": ("Tanker", "PassengerShip", "Tug"),
            "deepship_v6_slices": ("Tanker", "Cargo", "Tug"),
        }
        if self.dataset_format not in expected_classes:
            raise ValueError("Unknown dataset format")
        if tuple(self.class_names) != expected_classes[self.dataset_format]:
            raise ValueError("Class names/order do not match the dataset format")
        if not math.isfinite(self.ptt_noise_fraction) or not 0 < self.ptt_noise_fraction < 1:
            raise ValueError("ptt_noise_fraction must be between zero and one")
        if self.selection_metric not in ("single_macro_f1", "single_emr"):
            raise ValueError("Unknown single-target selection metric")
        if not isinstance(self.compare_online_val, bool):
            raise ValueError("compare_online_val must be boolean")
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


def checkpoint_config(saved_config):
    """Preserve old checkpoint behavior when experimental flags were absent."""
    config = dict(saved_config)
    config.setdefault("head_type", "query")
    config.setdefault("single_train_sampler", "scene_shuffle")
    config.setdefault("use_demon_features", True)
    config.setdefault("dataset_format", "v11")
    config.setdefault("noise_data_dir", "")
    config.setdefault("mapping_dir", "")
    config.setdefault("ptt_noise_fraction", 0.20)
    config.setdefault("selection_metric", "single_macro_f1")
    config.setdefault("compare_online_val", False)
    config.setdefault("feature_mixup_prob", 0.0)
    config.setdefault("feature_mixup_alpha", 0.2)
    return config


def from_baseline(saved_config):
    values = dict(saved_config)
    values.update(architecture="M2-Acoustic-TFConvNeXt-Small", head_type="query",
                  dims=tuple(TF_STRUCTURE["backbone_dims"]), depths=tuple(TF_STRUCTURE["backbone_blocks"]),
                  query_dim=TF_STRUCTURE["query_dim"], scan_backend="torch",
                  query_heads=TF_STRUCTURE["query_num_heads"], query_dropout=TF_STRUCTURE["query_dropout"],
                  query_attn_dropout=TF_STRUCTURE["query_attn_dropout"], ortho_weight=0.0, cr_weight=0.0)
    return Config(**values)
