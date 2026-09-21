"""TF-ConvNeXt control with the V11 feature-map geometry and shared head.

TFConvNeXtBlock is copied verbatim from ../M2_V6/model.py. The stem,
transitions, output BN and stage widths/depths retain the Mamba V11 geometry.
Only the stage processing is replaced; no Mamba scan is used.
"""
import torch
from torch import nn
from m2_components import DropPath


class TFConvNeXtBlock(nn.Module):
    r"""
    频时解耦大核 ConvNeXt Block

    结构:
      x → [1x31 DWConv + 11x1 DWConv] → LN → 1x1 Expand(4x) → GELU → 1x1 Project → + x

    参数量 (dim=C):
      DWConv: 31*C + 11*C = 42*C (vs 7x7 = 49*C)
      MLP:    2 * 4*C*C = 8*C^2
      LN:     2*C

    感受野:
      时间方向: 31 帧 (vs 原 7x7 = 7 帧, 4.4x)
      频率方向: 11 bin (vs 原 7x7 = 7 bin, 1.6x)
    """
    def __init__(self, dim, time_kernel=31, freq_kernel=11,
                 expand_ratio=4, drop_path=0.0):
        super().__init__()
        hidden = dim * expand_ratio

        # 频时解耦并行 DWConv
        self.dwconv_time = nn.Conv2d(dim, dim, kernel_size=(1, time_kernel),
                                     padding=(0, time_kernel // 2),
                                     groups=dim, bias=False)
        self.dwconv_freq = nn.Conv2d(dim, dim, kernel_size=(freq_kernel, 1),
                                     padding=(freq_kernel // 2, 0),
                                     groups=dim, bias=False)

        # ConvNeXt 逆瓶颈 MLP
        self.norm = nn.GroupNorm(1, dim)  # GroupNorm(1) ≈ LayerNorm for 2D
        self.pw1 = nn.Conv2d(dim, hidden, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        # 频时并行相加
        x = self.dwconv_time(x) + self.dwconv_freq(x)
        # 逆瓶颈 MLP
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return residual + self.drop_path(x)


class AcousticTFBackbone(nn.Module):
    """Two-channel input -> [B, 256, 33, 20] for the default 513x79 features."""
    def __init__(self, cfg):
        super().__init__()
        dims = cfg.dims
        # Identical stem geometry, normalization and initialization to V11.
        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-4), nn.ReLU(),
            nn.Conv2d(32, dims[0], 3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(dims[0], eps=1e-4), nn.ReLU(),
        )
        self.stages = nn.ModuleList()
        rates = torch.linspace(0, cfg.drop_path, sum(cfg.depths)).tolist()
        offset = 0
        for dim, depth in zip(dims, cfg.depths):
            stage = nn.Sequential(*[
                TFConvNeXtBlock(dim, time_kernel=cfg.tf_time_kernel,
                                freq_kernel=cfg.tf_freq_kernel,
                                expand_ratio=cfg.tf_expand_ratio,
                                drop_path=rates[offset + index])
                for index in range(depth)
            ])
            # Preserve the old TF block initialization, scoped to TF stages.
            stage.apply(self._init_tf_weights)
            self.stages.append(stage)
            offset += depth
        self.transitions = nn.ModuleList([
            nn.Conv2d(dims[0], dims[1], 3, stride=(2, 2), padding=1, bias=False),
            nn.Conv2d(dims[1], dims[2], 3, stride=(2, 1), padding=1, bias=False),
            nn.Conv2d(dims[2], dims[3], 3, stride=1, padding=1, bias=False),
        ])
        self.norm = nn.BatchNorm2d(dims[-1])
        self.out_dim = dims[-1]

    @staticmethod
    def _init_tf_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index < len(self.transitions):
                x = self.transitions[index](x)
        return self.norm(x)
