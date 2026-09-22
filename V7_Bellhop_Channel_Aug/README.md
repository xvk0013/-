# V7 Bellhop channel augmentation: one controlled training run

The architecture is the original 2,090,723-parameter Mamba (GAP + dual-band + DEMON).
No teacher, new backbone, consistency loss, filtering, resplitting or new noise.
Complete original model and baseline copies are in model_copy/ and baseline_copy/.
generator_reference/ contains the unchanged V7 Bellhop builder and original ARR/template snapshot.

## Execute

First, in **Windows MATLAB**, run:

    cd('D:/LSJ/Code/UART/MUART V6.0/UART M2/V7_Bellhop_Channel_Aug')
    PREPARE_CHANNEL_BANK

This reuses the original 80-frequency Bellhop ARR files and reconstructs the original 39 causal filters.
It exports output/channel_bank.mat. It does not retrain a model or regenerate any dataset.

Then, in **WSL**:

    conda activate LSJ
    cd "/mnt/d/LSJ/Code/UART/MUART V6.0/UART M2/V7_Bellhop_Channel_Aug"
    bash run_channel_aug.sh

1. Generate exactly 17,118 additional Train ship WAVs (~5.10 GiB of audio).
2. Train the original Mamba from scratch ONCE, seed42, 60 epochs.
3. Produce the original full Train/Val best-EMA report and comparison.

Return only:
D:/LSJ/Code/UART/MUART V6.0/UART M2/V7_Bellhop_Channel_Aug/output/summary.json

## Fixed comparison

For each original source, choose one of the other 38 original distance/depth combinations
uniformly, using a separate fixed seed17042. The normalized source input MAT (3-second
context plus 5-second core) is propagated through that filter and the original core
window is extracted, then written as float32 WAV with the original output gain.
This is an alternative propagation of the SAME pre-Bellhop source; it is NOT
another convolution of an already propagated WAV. No additional normalization
is inserted into generation. The model keeps its original per-wave RMS normalization.

Beginning-of-recording sources remain: their original context includes zero padding.
The 3-second real-history eligibility rule is for future multi-target data only.
No history QC or multi_target_eligible flag is used to reject a Train source here.

Every source draw chooses original/alternate with probability 0.5/0.5, using a private
worker RNG. Sampler source identities and original order are preserved.
Train sources remain 17,118, noise remains 3,424. New stored ship variants do NOT
double the epoch budget: 1,800 ships/class + 1,080 noise = 6,480 draws, 203 updates.
60 epochs = 12,180 successful optimizer updates. Same BCE + CE0.2, AdamW LR3e-4,
WD1e-4, SpecAugment, EMA, and original Train feature statistics.
The two versions of one source are never passed together as an enlarged batch.

Val (3,668 ships + 734 noise) is unchanged. Test is not loaded.
Final Train reporting uses the original Train waves, so it matches the baseline.
This is one-seed development evidence, not a final independent test result.

## Outputs / interruption recovery

- output/channel_bank.mat: exact original Bellhop filter bank
- output/alternate_train/{Tanker,Cargo,Tug}/Train/*.wav: new Train views only
- output/alternate_train/manifest.json: original source identities and paired geometry
- output/alternate_train/summary.json: preparation counts and 3 original-wave reproduction checks
- output/run/: best.pt, last.pt, history.jsonl, config.json
- output/report/: original full Train/Val predictions and confusion matrices
- output/summary.json: combined result, baseline differences, actual alternate exposure,
  per-epoch time and peak CUDA allocation

Repeat the same commands after interruption. Completed channel export and committed
WAVs are reused. Training resumes at the last committed epoch, with matching worker
RNG and source sampler state. Completed training/report stages are not repeated.
Checkpoint replacement retains the bounded retry fix for WSL/Windows-mounted files.
Do not run two instances in this folder simultaneously.

Architecture and inference compute are unchanged. Training has one waveform/frontend/
model pass per sample, not two. Extra cost is one-time filter construction and alternate
waveform preparation/storage; actual training wall time is recorded, not promised equal.

Checks: synthetic convolution/cropping, deterministic distinct channel assignment,
source pairing/noise exclusion, private RNG/epoch resumption, input features and model
forward/backward. No full data generation, formal training or Test evaluation by Codex.
