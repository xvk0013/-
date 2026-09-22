"""M2 query/DEMON/head components copied from model.py and model_v6a.py."""
from typing import Dict, Tuple
import torch
from torch import nn
import torch.nn.functional as F

class DropPath(nn.Module):
    """Stochastic Depth (per-sample) — ConvNeXt 标配抗过拟合"""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x / keep_prob * random_tensor.floor_()


class DEMONEncoder(nn.Module):
    """DEMON 谱 1D CNN 编码器

    输入: (B, 1, demon_n_bins)  DEMON 谱 (去 DC, 标准化后)
    输出: (B, demon_feat_dim)   DEMON 特征向量

    结构: 3 层 Conv1d (k=7, s=2) → BN → GELU → AdaptiveAvgPool1d
    感受野: 7+14+28 = 49 bins, 覆盖几乎全谱
    """
    def __init__(self, config):
        super().__init__()
        dims = config.demon_encoder_dims  # [32, 64, 128]
        in_ch = 1
        layers = []
        for i, out_ch in enumerate(dims):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3, bias=False))
            layers.append(nn.GroupNorm(1, out_ch))  # GroupNorm(1) ≈ LayerNorm
            layers.append(nn.GELU())
            if i < len(dims) - 1:
                layers.append(nn.MaxPool1d(kernel_size=2))  # 降采样
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(dims[-1], config.demon_feat_dim)
        self.norm = nn.LayerNorm(config.demon_feat_dim)

    def forward(self, x):
        """x: (B, 1, demon_n_bins) → (B, demon_feat_dim)"""
        x = self.conv(x)           # (B, C, L')
        x = self.pool(x).squeeze(-1)  # (B, C)
        x = self.proj(x)           # (B, demon_feat_dim)
        return self.norm(x)


class SemanticQueryAttention(nn.Module):
    """语义查询交叉注意力 — 多目标解耦的核心模块

    机制:
      - N 个可学习查询向量 (N = num_classes), 每个查询"负责"一个类别
      - 查询主动扫描 LOFAR 特征图 (Cross-Attention), 从"被动看图"变"主动搜索"
      - Q_Tanker 聚焦低频强线谱区, Q_Tug 避开大船低频区去高频找细小线谱

    2D 位置编码 (关键改进):
      - Cross-Attention 是排列不变的, 展平后丢失了频时位置信息
      - 注入频率轴 PE + 时间轴 PE, 让 Query 知道"去哪里找"

    输入: feat_map (B, C, H, W)  — TF-ConvNeXt 最后一层特征图
    输出: query_feats (B, N, C)  — 每个查询聚合后的特征
    """
    def __init__(self, dim, num_queries, num_heads=4,
                 attn_dropout=0.1, proj_dropout=0.1,
                 max_h=64, max_w=32, num_bands=1):
        super().__init__()
        self.num_queries = num_queries
        self.dim = dim
        self.num_bands = num_bands

        # 可学习查询向量 (N, C)
        self.queries = nn.Parameter(torch.empty(num_queries, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # ========================================================
        # 2D 频时可学习位置编码 — 破解 Cross-Attention 排列不变性
        # ========================================================
        self.freq_pe = nn.Parameter(torch.zeros(1, max_h, dim))
        self.time_pe = nn.Parameter(torch.zeros(1, max_w, dim))
        nn.init.trunc_normal_(self.freq_pe, std=0.02)
        nn.init.trunc_normal_(self.time_pe, std=0.02)

        # E4 双流融合臂：两路频率坐标不同，各自编码和 token 化后再加入
        # 频带身份。单路基线不创建该参数，保持旧 checkpoint 键完全兼容。
        if num_bands > 1:
            self.band_pe = nn.Parameter(torch.zeros(num_bands, dim))
            nn.init.trunc_normal_(self.band_pe, std=0.02)
        else:
            self.register_parameter('band_pe', None)

        # ========================================================
        # Cross-Attention (单层; ModuleList 包装保持与旧 checkpoint
        # 的 cross_attn_layers.0 键名严格兼容)
        # ========================================================
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True,
            ) for _ in range(1)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(1)
        ])
        self.proj_drop = nn.Dropout(proj_dropout)

    def _tokenize(self, feat_map, band_idx=0):
        """单路特征图注入二维位置编码并转为 tokens。"""
        B, C, H, W = feat_map.shape
        if H > self.freq_pe.shape[1] or W > self.time_pe.shape[1]:
            raise ValueError(
                f'特征图尺寸 {(H, W)} 超出位置编码上限 '
                f'{(self.freq_pe.shape[1], self.time_pe.shape[1])}'
            )

        # 注入 2D 位置编码
        f_pe = self.freq_pe[:, :H, :].unsqueeze(2).expand(-1, -1, W, -1)
        t_pe = self.time_pe[:, :W, :].unsqueeze(1).expand(-1, H, -1, -1)
        feat_map_pe = feat_map.permute(0, 2, 3, 1)  # (B, H, W, C)
        feat_map_pe = feat_map_pe + f_pe + t_pe
        if self.band_pe is not None:
            feat_map_pe = feat_map_pe + self.band_pe[band_idx].view(1, 1, 1, C)
        return feat_map_pe.flatten(1, 2)  # (B, H*W, C)

    def forward(self, feat_map):
        """单路特征图或多路特征图 → query_feats: (B, N, C)。"""
        if isinstance(feat_map, (tuple, list)):
            if len(feat_map) != self.num_bands:
                raise ValueError(
                    f'期望 {self.num_bands} 路特征，实际得到 {len(feat_map)} 路'
                )
            tokens = torch.cat([
                self._tokenize(one_map, band_idx=i)
                for i, one_map in enumerate(feat_map)
            ], dim=1)
            B = feat_map[0].shape[0]
        else:
            if self.num_bands != 1:
                raise ValueError(f'双流模型期望 {self.num_bands} 路特征图')
            tokens = self._tokenize(feat_map)
            B = feat_map.shape[0]

        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, N, C)

        for attn, norm in zip(self.cross_attn_layers, self.norms):
            out = attn(q, tokens, tokens, need_weights=False)[0]
            q = norm(q + out)  # 残差 + LayerNorm

        return self.proj_drop(q)

LEGAL_SET_NAMES: Tuple[str, ...] = (
    "noise",
    "Tanker",
    "Cargo",
    "Tug",
    "Tanker+Cargo",
    "Tanker+Tug",
    "Cargo+Tug",
)


LEGAL_SET_MULTI_HOT = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ],
    dtype=torch.float32,
)


PAIR_CLASS_INDICES: Tuple[Tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))


def multihot_to_set_targets(labels: torch.Tensor) -> torch.Tensor:
    """Convert legal [Tanker, Cargo, Tug] multi-hot rows to set indices.

    Illegal soft, negative, or three-target rows are rejected rather than
    silently mapped into a legal class.
    """

    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError(f"labels must have shape (B, 3), got {tuple(labels.shape)}")
    if not torch.isfinite(labels).all():
        raise ValueError("labels contain NaN or Inf")
    binary = (labels == 0) | (labels == 1)
    if not bool(binary.all()):
        raise ValueError("labels must be exact binary multi-hot values")
    counts = labels.sum(dim=1)
    if bool((counts > 2).any()):
        raise ValueError("V6-A rejects triple-target labels")

    # bit code in class order: Tanker=1, Cargo=2, Tug=4.
    bit_weights = labels.new_tensor([1, 2, 4])
    codes = (labels * bit_weights).sum(dim=1).long()
    lookup = torch.full((8,), -1, dtype=torch.long, device=labels.device)
    lookup[labels.new_tensor([0, 1, 2, 4, 3, 5, 6], dtype=torch.long)] = \
        labels.new_tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long)
    targets = lookup[codes]
    if bool((targets < 0).any()):
        raise ValueError("labels contain an unsupported target set")
    return targets


def set_targets_to_multihot(targets: torch.Tensor) -> torch.Tensor:
    """Convert legal-set indices to [Tanker, Cargo, Tug] multi-hot rows."""

    if targets.ndim != 1:
        raise ValueError(f"targets must have shape (B,), got {tuple(targets.shape)}")
    if targets.numel() and (bool((targets < 0).any()) or bool((targets >= 7).any())):
        raise ValueError("legal-set target index must be in [0, 6]")
    table = LEGAL_SET_MULTI_HOT.to(device=targets.device)
    return table[targets.long()]


def conditional_joint_log_probs(
    tc_logits: torch.Tensor,
    single_logits: torch.Tensor,
    pair_logits: torch.Tensor,
) -> torch.Tensor:
    """Compose normalized log probabilities for all seven legal sets."""

    if tc_logits.ndim != 2 or tc_logits.shape[1] != 3:
        raise ValueError("tc_logits must have shape (B, 3)")
    if single_logits.shape != tc_logits.shape:
        raise ValueError("single_logits must have shape (B, 3)")
    if pair_logits.shape != tc_logits.shape:
        raise ValueError("pair_logits must have shape (B, 3)")

    log_count = F.log_softmax(tc_logits, dim=1)
    log_single = F.log_softmax(single_logits, dim=1)
    log_pair = F.log_softmax(pair_logits, dim=1)
    return torch.cat(
        [
            log_count[:, 0:1],
            log_count[:, 1:2] + log_single,
            log_count[:, 2:3] + log_pair,
        ],
        dim=1,
    )


class ConditionalSetHead(nn.Module):
    """Shared semantic representation with count/single/pair conditionals."""

    def __init__(
        self,
        feat_dim: int,
        demon_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        attn_dropout: float = 0.1,
        use_demon: bool = True,
        max_h: int = 64,
        max_w: int = 32,
        tc_hidden: int = 128,
        pair_hidden: int = 256,
        num_feature_maps: int = 1,
    ) -> None:
        super().__init__()
        self.use_demon = use_demon
        self.num_feature_maps = num_feature_maps
        self.query_attn = SemanticQueryAttention(
            dim=feat_dim,
            num_queries=3,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            max_h=max_h,
            max_w=max_w,
            num_bands=num_feature_maps,
        )

        query_dim = feat_dim + (demon_dim if use_demon else 0)
        pooled_dim = feat_dim * num_feature_maps + \
            (demon_dim if use_demon else 0)

        # One shared scorer applied to the three class-specific query tokens.
        self.single_scorer = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim, 1),
        )

        # A symmetric pair representation [sum, |difference|, product] keeps
        # pair scores independent of the arbitrary order of the two classes.
        self.pair_scorer = nn.Sequential(
            nn.LayerNorm(3 * query_dim),
            nn.Linear(3 * query_dim, pair_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden, 1),
        )

        self.tc_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, tc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tc_hidden, 3),
        )

    def _pool_feature_maps(self, feat_map) -> torch.Tensor:
        maps = list(feat_map) if isinstance(feat_map, (tuple, list)) else [feat_map]
        if len(maps) != self.num_feature_maps:
            raise ValueError(
                f"expected {self.num_feature_maps} feature maps, got {len(maps)}"
            )
        return torch.cat([one.mean(dim=(2, 3)) for one in maps], dim=1)

    def forward(self, feat_map, demon_feat=None) -> Dict[str, torch.Tensor]:
        query_features = self.query_attn(feat_map)  # (B, 3, C)
        pooled = self._pool_feature_maps(feat_map)

        if self.use_demon:
            if demon_feat is None:
                raise ValueError("DEMON is enabled but demon input/features are missing")
            expanded = demon_feat.unsqueeze(1).expand(-1, 3, -1)
            query_features = torch.cat([query_features, expanded], dim=2)
            pooled = torch.cat([pooled, demon_feat], dim=1)

        tc_logits = self.tc_head(pooled)
        single_logits = self.single_scorer(query_features).squeeze(-1)

        pair_features = []
        for left, right in PAIR_CLASS_INDICES:
            a = query_features[:, left]
            b = query_features[:, right]
            pair_features.append(torch.cat([a + b, (a - b).abs(), a * b], dim=1))
        pair_tensor = torch.stack(pair_features, dim=1)
        pair_logits = self.pair_scorer(pair_tensor).squeeze(-1)

        set_log_probs = conditional_joint_log_probs(
            tc_logits, single_logits, pair_logits
        )
        return {
            "tc_logits": tc_logits,
            "single_logits": single_logits,
            "pair_logits": pair_logits,
            "set_log_probs": set_log_probs,
            "query_features": query_features,
        }
