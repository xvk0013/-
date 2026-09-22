"""Training-only smooth spectral coloration; no additive noise or pitch shift."""
import numpy as np


def smooth_frequency_response(wave, sample_rate, max_db, rng):
    if max_db == 0:
        return wave
    # Reflect padding limits boundary wraparound from the zero-phase FFT filter.
    pad = min(sample_rate // 2, len(wave) - 1)
    padded = np.pad(wave.astype(np.float64), (pad, pad), mode="reflect")
    freq = np.fft.rfftfreq(len(padded), 1.0 / sample_rate)
    anchors = np.array([0, 125, 250, 500, 1000, 2000, 4000, sample_rate / 2])
    values = rng.uniform(-max_db, max_db, len(anchors))
    position = np.log1p(freq / 125)
    knots = np.log1p(anchors / 125)
    indices = np.clip(np.searchsorted(knots, position, side="right") - 1, 0, len(knots) - 2)
    fraction = (position - knots[indices]) / (knots[indices + 1] - knots[indices])
    blend = (1 - np.cos(np.pi * fraction)) / 2
    db = values[indices] * (1 - blend) + values[indices + 1] * blend
    filtered = np.fft.irfft(np.fft.rfft(padded) * 10 ** (db / 20), n=len(padded))[pad:pad + len(wave)]
    old_rms = np.sqrt(np.mean(np.square(wave, dtype=np.float64)))
    new_rms = np.sqrt(np.mean(filtered ** 2))
    if not np.isfinite(new_rms) or new_rms <= 0:
        raise FloatingPointError("Invalid spectral augmentation RMS")
    return (filtered * (old_rms / new_rms)).astype(np.float32)
