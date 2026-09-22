"""One Mamba backbone; single-target preparation and joint-target training."""
import torch
from torch import nn
from backbone import AcousticMambaBackbone
from m2_components import (DEMONEncoder, ConditionalSetHead, PAIR_CLASS_INDICES,
                           LEGAL_SET_MULTI_HOT, conditional_joint_log_probs)


class SharedEvidenceHead(ConditionalSetHead):
    def __init__(self, cfg):
        super().__init__(cfg.query_dim, cfg.demon_feat_dim,
                         num_heads=cfg.query_heads, dropout=cfg.query_dropout,
                         attn_dropout=cfg.query_attn_dropout, max_h=64, max_w=32,
                         tc_hidden=128, pair_hidden=32, use_demon=True)
        self.interaction_bound = 0.5
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.pair_scorer[-1].weight)
        nn.init.zeros_(self.pair_scorer[-1].bias)

    def forward(self, feat_map, demon_feat, phase):
        query = self.query_attn(feat_map)
        query = torch.cat([query, demon_feat[:, None].expand(-1, 3, -1)], dim=-1)
        evidence = self.single_scorer(query).squeeze(-1)
        outputs = {"presence_logits": evidence, "query_features": query}
        if phase == "single":
            return outputs
        pooled = torch.cat([self._pool_feature_maps(feat_map), demon_feat], dim=-1)
        pair_features, additive = [], []
        for i, j in PAIR_CLASS_INDICES:
            a, b = query[:, i], query[:, j]
            pair_features.append(torch.cat([a + b, (a - b).abs(), a * b], dim=-1))
            additive.append(evidence[:, i] + evidence[:, j])
        interaction = self.interaction_bound * torch.tanh(
            self.pair_scorer(torch.stack(pair_features, dim=1)).squeeze(-1))
        pair = torch.stack(additive, dim=1) + interaction
        counts = self.tc_head(pooled)
        outputs.update(tc_logits=counts, pair_logits=pair, pair_interaction=interaction,
                       set_log_probs=conditional_joint_log_probs(
                           counts.float(), evidence.float(), pair.float()))
        return outputs


class GlobalPoolHead(nn.Module):
    """Single-phase mean pooling (global or four frequency bins) plus DEMON."""
    def __init__(self, cfg):
        super().__init__()
        self.frequency_bins = 4 if cfg.head_type == "freq4" else 1
        band_factor = 2 if cfg.band_fusion == "late_shared" else 1
        self.scorer = nn.Sequential(
            nn.Linear(band_factor * self.frequency_bins * cfg.query_dim + cfg.demon_feat_dim, cfg.query_dim),
            nn.GELU(),
            nn.Dropout(cfg.query_dropout),
            nn.Linear(cfg.query_dim, 3),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, feat_map, demon_feat, phase, mixup=None):
        if mixup is not None and (not self.training or self.frequency_bins != 1):
            raise ValueError("Feature Mixup is only supported by GAP during training")
        if phase != "single":
            raise ValueError("GAP head is a single-target diagnostic, not a joint head")
        if self.frequency_bins == 1:
            acoustic = feat_map.mean(dim=(-2, -1))
        else:
            # Input is [batch, channels, frequency, time].
            # Non-overlapping contiguous feature-frequency bins; no Hz claim.
            if feat_map.shape[-2] < self.frequency_bins:
                raise ValueError("Feature frequency axis is too small for four bins")
            time_mean = feat_map.mean(dim=-1)
            bins = torch.tensor_split(time_mean, self.frequency_bins, dim=-1)
            acoustic = torch.cat([band.mean(dim=-1) for band in bins], dim=-1)
        pooled = torch.cat([acoustic, demon_feat], dim=-1)
        if mixup is not None:
            pooled = mixup.apply(pooled)
        return {"presence_logits": self.scorer(pooled), "pooled_features": pooled}


class M2Mamba(nn.Module):
    def __init__(self, cfg, phase="single"):
        super().__init__()
        cfg.validate()
        self.band_fusion = cfg.band_fusion
        self.input_channels = cfg.input_channels
        self.backbone = AcousticMambaBackbone(cfg)
        self.projection = nn.Conv2d(cfg.dims[-1], cfg.query_dim, 1)
        self.demon_encoder = DEMONEncoder(cfg)
        # Retain module construction/order and checkpoint keys for the ablation.
        self.use_demon_features = cfg.use_demon_features
        self.demon_encoder.requires_grad_(self.use_demon_features)
        self.head_type = cfg.head_type
        self.classifier = GlobalPoolHead(cfg) if cfg.head_type in ("gap", "freq4") else SharedEvidenceHead(cfg)
        self.register_buffer("legal_sets", LEGAL_SET_MULTI_HOT.clone())
        self.set_phase(phase)

    def set_phase(self, phase):
        if phase not in ("single", "joint"):
            raise ValueError("phase must be single or joint")
        if self.head_type in ("gap", "freq4") and phase != "single":
            raise ValueError("GAP control supports single phase only")
        self.phase = phase
        if self.head_type in ("gap", "freq4"):
            return
        for module in (self.classifier.tc_head, self.classifier.pair_scorer):
            module.requires_grad_(phase == "joint")

    def forward(self, features, demon, mixup=None):
        if mixup is not None and (not self.training or self.phase != "single"
                                   or self.head_type != "gap" or self.band_fusion != "early"):
            raise ValueError("Feature Mixup is restricted to training the early-fusion GAP model")
        if features.ndim != 4 or features.shape[1] != self.input_channels:
            raise ValueError(f"Expected [batch, {self.input_channels}, frequency, time] features")
        feat = self.projection(self.backbone(features))
        if self.band_fusion == "late_shared":
            low, high = feat.chunk(2, dim=0)
            # GAP is channel-wise: this is equivalent to independently pooling
            # each band and concatenating the resulting vectors in the head.
            feat = torch.cat((low, high), dim=1)
        demon_feat = self.demon_encoder(demon)
        if not self.use_demon_features:
            # Zero AFTER encoding so learned biases cannot enter the classifier.
            # Keep the same feature width/dtype and the same shared head.
            demon_feat = torch.zeros_like(demon_feat)
        if mixup is not None:
            return self.classifier(feat, demon_feat, self.phase, mixup=mixup)
        return self.classifier(feat, demon_feat, self.phase)

    @torch.no_grad()
    def decode(self, outputs):
        if self.phase == "single":
            # Do not force argmax: zero/multiple predictions remain visible errors.
            return (outputs["presence_logits"] >= 0).float()
        return self.legal_sets[outputs["set_log_probs"].argmax(dim=1)]


def parameter_counts(model):
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
