# Sources and local adaptations

## MambaVision

- Repository: https://github.com/NVlabs/MambaVision
- Pinned commit: `7860a506b2eb844eaaae676f08461ce8c3c26f43`
- Source: `mambavision/models/mamba_vision.py`
- Original source is retained as `third_party/mambavision_upstream.py` for provenance; it is not imported by the training code.
- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
- License: NVIDIA Source Code License-NC; complete text in `third_party/LICENSE_MambaVision.txt`.

`mamba_blocks.py` copies/adapts ConvBlock, MambaVisionMixer, Attention and Block from that revision. Native PyTorch tensor operations replace einops; a small equivalent MLP and the copied M2 DropPath replace timm dependencies. The pretrained classifier, registration and checkpoint-download code are omitted.

Intentional changes:

1. The selective scan backend is explicit: the installed mamba-ssm CUDA operator or an ordinary PyTorch implementation of the same state recurrence.
2. The delta projection is computed without its bias before scan, and the bias is applied exactly once inside scan. The pinned upstream implementation calls the biased Linear and also passes that bias to scan. The local code avoids that double addition.
3. The SSM delta weight and bias initialization is retained; generic Linear initialization does not overwrite it.
4. Attention dropout is explicitly zero in evaluation.
5. The acoustic wrapper changes channels, depths, MLP ratio, input channels, and frequency/time strides; it retains local window partitioning and the stage Mamba/attention order.

This is an acoustic adaptation, not a claim of exact reproduction of the official ImageNet model or pretrained results. No pretrained weights have been downloaded.

## Existing M2 code

Copied from the parent project into this independent folder:

- `dataset.py`: normalization, LOFARTransform, DualBandTransform, DEMONTransform, SpecAugment -> `frontend.py`.
- `model.py`: DropPath, DEMONEncoder, SemanticQueryAttention -> `m2_components.py`.
- `model_v6a.py`: legal-set definitions/conversion/composition and ConditionalSetHead -> `m2_components.py`.
- `model_v6a2.py`: shared evidence/additive pair interaction design -> adapted `model.py`.
- `losses/cross_recording_consistency.py` -> `cross_recording.py`, copied directly.
- The query orthogonality definition and EMA parameter/buffer update policy follow `train_v6a.py`.

The new implementation has no runtime import from the original M2 directory and provides no TF-backbone selection interface.
