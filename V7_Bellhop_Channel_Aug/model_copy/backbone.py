"""Small acoustic adaptation of MambaVision, frequency/time stride 16/4."""
import torch
from torch import nn
import torch.nn.functional as F
from mamba_blocks import Block, ConvBlock, MambaVisionMixer


class WindowStage(nn.Module):
    def __init__(self, dim, depth, heads, window, mlp_ratio, drop_rates, backend):
        super().__init__()
        self.window = window
        # Same stage recipe as MambaVision: first half SSM, second half attention.
        attention_blocks = list(range((depth + 1) // 2, depth))
        self.blocks = nn.ModuleList([
            Block(dim, heads, i, attention_blocks, mlp_ratio=mlp_ratio,
                  qkv_bias=True, drop_path=drop_rates[i]) for i in range(depth)
        ])
        for module in self.modules():
            if isinstance(module, MambaVisionMixer):
                module.scan_backend = backend

    def forward(self, x):
        batch, channels, height, width = x.shape
        w = self.window
        pad_h, pad_w = (-height) % w, (-width) % w
        x = F.pad(x, (0, pad_w, 0, pad_h))
        h, t = height + pad_h, width + pad_w
        # Local row-major windows, as in upstream; queries later aggregate all windows.
        x = x.reshape(batch, channels, h // w, w, t // w, w)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, w * w, channels)
        for block in self.blocks:
            x = block(x)
        x = x.reshape(batch, h // w, t // w, w, w, channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, h, t)
        return x[:, :, :height, :width].contiguous()


class AcousticMambaBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dims = cfg.dims
        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-4), nn.ReLU(),
            nn.Conv2d(32, dims[0], 3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(dims[0], eps=1e-4), nn.ReLU(),
        )
        self.stages = nn.ModuleList()
        rates = torch.linspace(0, cfg.drop_path, sum(cfg.depths)).tolist()
        offset = 0
        for i, (dim, depth) in enumerate(zip(dims, cfg.depths)):
            stage_rates = rates[offset:offset + depth]
            offset += depth
            if i < 2:
                stage = nn.Sequential(*[ConvBlock(dim, drop_path=r) for r in stage_rates])
            else:
                stage = WindowStage(dim, depth, cfg.heads[i], cfg.windows[i-2],
                                    cfg.mlp_ratio, stage_rates, cfg.scan_backend)
            self.stages.append(stage)
        self.transitions = nn.ModuleList([
            nn.Conv2d(dims[0], dims[1], 3, stride=(2, 2), padding=1, bias=False),
            nn.Conv2d(dims[1], dims[2], 3, stride=(2, 1), padding=1, bias=False),
            nn.Conv2d(dims[2], dims[3], 3, stride=1, padding=1, bias=False),
        ])
        self.norm = nn.BatchNorm2d(dims[-1])
        self.out_dim = dims[-1]
        # Keep the SSM timescale initialization; never zero dt_proj.bias.
        special = {id(m.dt_proj) for m in self.modules() if isinstance(m, MambaVisionMixer)}
        for module in self.modules():
            if isinstance(module, nn.Linear) and id(module) not in special:
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.band_fusion = cfg.band_fusion
        if cfg.band_fusion == "late_shared":
            import copy
            # Independent stems; the stages/transitions/norm below remain shared.
            original = self.stem[0]
            high_stem = copy.deepcopy(self.stem)
            with torch.random.fork_rng(devices=[]):
                low_conv = nn.Conv2d(1, 32, 3, stride=(2, 1), padding=1, bias=False)
                high_conv = nn.Conv2d(1, 32, 3, stride=(2, 1), padding=1, bias=False)
            with torch.no_grad():
                # Match single-input fan-in scale, without changing the trunk RNG.
                low_conv.weight.copy_(original.weight[:, :1] * (2 ** 0.5))
                high_conv.weight.copy_(original.weight[:, 1:2] * (2 ** 0.5))
            self.stem[0] = low_conv
            high_stem[0] = high_conv
            self.high_stem = high_stem
        if cfg.input_channels == 3:
            # Preserve original shared initialization and RNG stream. The new channel
            # starts at zero contribution but its convolution weights can learn.
            old = self.stem[0]
            with torch.random.fork_rng(devices=[]):
                expanded = nn.Conv2d(3, 32, 3, stride=(2, 1), padding=1, bias=False)
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, :2].copy_(old.weight)
            self.stem[0] = expanded

    def forward(self, x):
        if self.band_fusion == "late_shared":
            # [all low, all high] in one batch: shared BN updates once, without
            # branch-order effects. No convolution/attention mixes the two samples.
            x = torch.cat((self.stem(x[:, :1]), self.high_stem(x[:, 1:2])), dim=0)
        else:
            x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        return self.norm(x)
