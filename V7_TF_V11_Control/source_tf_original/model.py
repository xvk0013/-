"""TF backbone control with unchanged V11 evidence heads and decoding.

The historical M2Mamba class name is retained for the shared entry points.
"""
import torch
from torch import nn
from backbone import AcousticTFBackbone
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


class M2Mamba(nn.Module):
    def __init__(self, cfg, phase="single"):
        super().__init__()
        cfg.validate()
        self.backbone = AcousticTFBackbone(cfg)
        self.projection = nn.Conv2d(cfg.dims[-1], cfg.query_dim, 1)
        self.demon_encoder = DEMONEncoder(cfg)
        self.classifier = SharedEvidenceHead(cfg)
        self.register_buffer("legal_sets", LEGAL_SET_MULTI_HOT.clone())
        self.set_phase(phase)

    def set_phase(self, phase):
        if phase not in ("single", "joint"):
            raise ValueError("phase must be single or joint")
        self.phase = phase
        for module in (self.classifier.tc_head, self.classifier.pair_scorer):
            module.requires_grad_(phase == "joint")

    def forward(self, features, demon):
        if features.ndim != 4 or features.shape[1] != 2:
            raise ValueError("Expected [batch, 2, frequency, time] features")
        feat = self.projection(self.backbone(features))
        return self.classifier(feat, self.demon_encoder(demon), self.phase)

    @torch.no_grad()
    def decode(self, outputs):
        if self.phase == "single":
            # Do not force argmax: zero/multiple predictions remain visible errors.
            return (outputs["presence_logits"] >= 0).float()
        return self.legal_sets[outputs["set_log_probs"].argmax(dim=1)]


def parameter_counts(model):
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
