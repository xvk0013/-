"""Copied M2 acoustic transforms; unchanged numerical feature definitions."""
import numpy as np
from scipy.signal import butter, filtfilt as sp_filtfilt, welch
import torch
import torchaudio

NORMALIZATION_MODES = {'global', 'per_channel'}


def normalize_feature(feature, mean, std, mode='global'):
    """用 Train 统计量归一化特征，不做逐样本标准化。

    ``global`` 要求单个标量 mean/std，严格复现 E1。``per_channel`` 要求
    每个输入通道各一个 mean/std，并仅沿频率和时间维广播。
    """
    if mean is None and std is None:
        return feature
    if mean is None or std is None:
        raise ValueError('normalization mean/std 必须同时提供')
    if mode not in NORMALIZATION_MODES:
        raise ValueError(
            f'未知 normalization_mode={mode!r}; '
            f'当前只支持 {sorted(NORMALIZATION_MODES)}'
        )

    mean_t = torch.as_tensor(mean, dtype=feature.dtype, device=feature.device).flatten()
    std_t = torch.as_tensor(std, dtype=feature.dtype, device=feature.device).flatten()
    if mean_t.numel() != std_t.numel():
        raise ValueError('normalization mean/std 元素数量不一致')
    if not torch.isfinite(mean_t).all() or not torch.isfinite(std_t).all():
        raise ValueError('normalization mean/std 包含 NaN 或 Inf')
    if torch.any(std_t <= 0):
        raise ValueError('normalization std 必须全部为有限正数')

    if mode == 'global':
        if mean_t.numel() != 1:
            raise ValueError('global normalization 要求标量 mean/std')
        return (feature - mean_t[0]) / std_t[0]

    n_channels = 1 if feature.dim() == 2 else feature.shape[0]
    if mean_t.numel() != n_channels:
        raise ValueError(
            f'per_channel normalization 需要 {n_channels} 组 mean/std，'
            f'实际得到 {mean_t.numel()} 组'
        )
    if feature.dim() == 2:
        return (feature - mean_t[0]) / std_t[0]
    broadcast_shape = (n_channels,) + (1,) * (feature.dim() - 1)
    return (feature - mean_t.reshape(broadcast_shape)) / std_t.reshape(broadcast_shape)


class LOFARTransform:
    """波形 → LOFAR 时频图 (全局归一化)

    LOFAR (Low Frequency Analysis and Recording):
      - 长窗 STFT (n_fft=8192) → 高频率分辨率 (1.95 Hz/bin)
      - 频率范围 0-1000Hz → 聚焦船舶线谱
      - log 压缩 → 动态范围压缩
    """
    def __init__(self, cfg, global_mean=None, global_std=None,
                 n_fft=None, hop=None, win=None, n_bins=None, frames=None):
        self.cfg = cfg
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft or cfg.lofar_n_fft,
            hop_length=hop or cfg.lofar_hop,
            win_length=win or cfg.lofar_win,
            power=2.0,
            normalized=False,
        )
        self.n_bins = n_bins or cfg.lofar_n_bins  # 513
        self.frames_target = frames   # 时间中心裁剪目标 (宽带短窗帧数 77 → 71)
        self.global_mean = global_mean
        self.global_std = global_std
        self.normalization_mode = getattr(cfg, 'normalization_mode', 'global')
        if self.normalization_mode not in NORMALIZATION_MODES:
            raise ValueError(f'未知 normalization_mode={self.normalization_mode!r}')

    def __call__(self, waveform):
        """waveform: (T,) → lofar: (1, n_bins, frames)"""
        spec = self.stft(waveform)
        spec = spec[:self.n_bins, :]
        spec_log = torch.log(spec + 1e-6)  # (F, T)

        # 时间中心裁剪 (宽带 n_fft=2048 → 77 帧 → 71, 与 LOFAR/CWT 帧数统一)
        if self.frames_target is not None and spec_log.shape[-1] > self.frames_target:
            start = (spec_log.shape[-1] - self.frames_target) // 2
            spec_log = spec_log[:, start:start + self.frames_target]

        # 全局归一化
        if self.global_mean is not None and self.global_std is not None:
            spec_log = normalize_feature(
                spec_log, self.global_mean, self.global_std,
                mode=self.normalization_mode)

        if spec_log.dim() == 2:
            spec_log = spec_log.unsqueeze(0)  # (1, F, T)
        return spec_log


class DualBandTransform:
    """双通道互补特征: ch0 = LOFAR 0-1kHz (1.95Hz/bin) + ch1 = 宽带高频 1-4kHz (7.8Hz/bin)

    依据 (单特征 1-seed 筛屏, no-anchor):
      lofar-alone 0.7475 > wideband-alone 0.7377 > cwt-alone 0.7353
      → 低频高分辨率 (8192 窗) 是命根子不可弃; 高频 1-4kHz 作增量通道
      (diag_feat.py: 高频带对 {0_1,0_2}-vs-1_2 混淆对 AUC 0.62, 低频仅 0.58)

    两个 STFT 从同一波形计算 (hop=1024, center=True → 79 帧天然对齐);
    高频 385 bins (1-4kHz @7.8Hz/bin) 线性插值到 513 与 ch0 堆叠 → (2, 513, 79)。
    """
    def __init__(self, cfg, global_mean=None, global_std=None):
        self.local_contrast_bins = cfg.local_contrast_bins
        self.lofar_tf = LOFARTransform(cfg)               # ch0, 无归一化
        high_n_fft = cfg.wideband_n_fft
        self.hz_per_bin = cfg.sample_rate / high_n_fft    # 2048 → 7.8 Hz/bin @16k
        self.f_lo = getattr(cfg, 'wideband_freq_lo_hz', 1000.0)
        self.f_hi = getattr(cfg, 'wideband_freq_max_hz', 4000.0)
        n_bins_high = int(round(self.f_hi / self.hz_per_bin)) + 1  # 覆盖到 f_hi 的 bin 数
        self.high_tf = LOFARTransform(cfg,                # ch1 高频源
                                      n_fft=high_n_fft,
                                      hop=cfg.wideband_hop,
                                      win=high_n_fft,
                                      n_bins=n_bins_high)
        self.global_mean = global_mean
        self.global_std = global_std
        self.normalization_mode = getattr(cfg, 'normalization_mode', 'global')
        if self.normalization_mode not in NORMALIZATION_MODES:
            raise ValueError(f'未知 normalization_mode={self.normalization_mode!r}')

    def __call__(self, waveform):
        lo = self.lofar_tf(waveform)[0]                   # (513, 79) raw log
        wb = self.high_tf(waveform)[0]                    # (513, 79) raw log 0-4kHz
        i0 = int(round(self.f_lo / self.hz_per_bin))      # 128
        i1 = min(int(round(self.f_hi / self.hz_per_bin)), wb.shape[0])
        hi = wb[i0:i1]                                    # (385, 79)
        hi = torch.nn.functional.interpolate(             # 沿频率轴插值 385 → 513
            hi.t().unsqueeze(0), size=lo.shape[0],
            mode='linear', align_corners=False)[0].t()    # (513, 79)
        channels = [lo, hi]
        if self.local_contrast_bins:
            # Each time frame independently: subtract its smooth frequency background.
            # Reflect padding avoids artificially low background at band boundaries.
            k = self.local_contrast_bins
            frames = lo.t().unsqueeze(1)  # [time, 1, frequency]
            padded = torch.nn.functional.pad(frames, (k // 2, k // 2), mode='reflect')
            background = torch.nn.functional.avg_pool1d(padded, k, stride=1)[:, 0].t()
            channels.append(lo - background)
        x = torch.stack(channels)  # original LOFAR, wideband, optional local contrast
        if self.global_mean is not None and self.global_std is not None:
            x = normalize_feature(
                x, self.global_mean, self.global_std,
                mode=self.normalization_mode)
        return x


class DEMONTransform:
    """波形 → DEMON 谱 (去 DC + 标准化)

    DEMON (Detection of Envelope Modulation On Noise):
      1. 带通滤波 (cfg.demon_bp_low–cfg.demon_bp_high Hz): 选择解调载波频带
      2. 平方检波 (envelope²): 提取包络
      3. 低通滤波 (cutoff=100 Hz): 去除高频残余
      4. FFT → DEMON 谱 (0-50 Hz): 展示螺旋桨叶片率/轴频峰值
      5. 去 DC (bin 0-1) + per-sample 标准化

    验证结论: 去 DC 后 Tanker-Tug 可分性 0.03→0.09, 互补性 0.16-0.31
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_bins = cfg.demon_n_bins
        self.max_hz = cfg.demon_max_hz
        self.remove_dc = cfg.demon_remove_dc
        self.sample_rate = cfg.sample_rate
        self.seg_len = cfg.audio_len
        # [demon-slice 臂 2026-08-27] 探针裁决 (_diag_demon_probe): 包络谱 P1 0.4493
        # 接近 chance (模型无视它是理性的); 原始谱切片 P2 0.5536 (+10.4pp)。
        # slice 模式 = 输出 ch0 0-58.5Hz Welch 谱切片 (30 bins @1.95Hz/bin, 与探针
        # P2 同口径) 插值到 126 维 → 模型/损失/协议零改动 (接口长度不变)。
        # 默认 'demon' = 原包络谱行为 (主线不变)。
        self.slice_mode = getattr(cfg, 'demon_mode', 'demon') == 'slice'

        # 预计算滤波器系数 (scipy)
        nyq = self.sample_rate / 2
        self._bp_b, self._bp_a = butter(
            4, [cfg.demon_bp_low / nyq, cfg.demon_bp_high / nyq], btype='band')
        self._lp_b, self._lp_a = butter(
            4, cfg.demon_lp_cutoff / nyq, btype='low')

    def __call__(self, waveform):
        """waveform: (T,) torch tensor → demon: (1, n_bins_out,) torch tensor

        去DC后 n_bins_out = n_bins - 2; slice 模式输出同长度 (接口不变)
        """
        sig = waveform.numpy().astype(np.float64)

        # [demon-slice 臂] 0-58.5Hz 原始谱切片替代包络谱 (log + per-sample
        # 标准化, 保持边缘分布恒定性质; 长度 = n_bins-2 → 模型零改动)
        if self.slice_mode:
            f, p = welch(sig, fs=self.sample_rate, nperseg=8192,
                         noverlap=4096, window='hann', scaling='density')
            m = f <= 58.5                     # 30 bins @1.95Hz/bin (= 探针 P2)
            sl = np.log10(p[m] + 1e-12)
            sl = np.interp(np.linspace(0, len(sl) - 1, self.n_bins - 2),
                           np.arange(len(sl)), sl)      # 30 → 126
            sl = (sl - sl.mean()) / (sl.std() + 1e-8)
            return torch.from_numpy(sl.astype(np.float32)).unsqueeze(0)

        # 1. 带通滤波，范围由 cfg.demon_bp_low/high 指定
        bp = sp_filtfilt(self._bp_b, self._bp_a, sig)

        # 2. 平方检波 (envelope²)
        env = bp ** 2

        # 3. 低通滤波 (cutoff=100 Hz)
        env_lp = sp_filtfilt(self._lp_b, self._lp_a, env)

        # 4. FFT → DEMON 谱
        spec = np.fft.rfft(env_lp)
        power = np.abs(spec) ** 2

        # 取 0-max_hz 范围, 重采样到 n_bins
        hz_per_bin = self.sample_rate / len(sig)
        n_bins_raw = int(self.max_hz / hz_per_bin)
        power = power[:n_bins_raw]
        power = np.interp(
            np.linspace(0, n_bins_raw - 1, self.n_bins),
            np.arange(n_bins_raw),
            power,
        )

        # 5. log 压缩
        demon = np.log(power + 1e-10)

        # 6. 去 DC + per-sample 标准化
        if self.remove_dc:
            demon = demon[2:]  # 去 bin 0-1 (DC 分量)
        demon = (demon - demon.mean()) / (demon.std() + 1e-8)

        return torch.from_numpy(demon.astype(np.float32)).unsqueeze(0)  # (1, n_bins_out)


class SpecAugment:
    """SpecAugment 数据增强 (频率掩码 + 时间掩码)"""
    def __init__(self, n_freq_masks=2, n_time_masks=2,
                 freq_mask_param=30, time_mask_param=20):
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def __call__(self, mel):
        """mel: (1, F, T) 或 (C, F, T)"""
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        c, n_freq, n_time = mel.shape

        for _ in range(self.n_freq_masks):
            f = min(self.freq_mask_param, n_freq)
            f_len = torch.randint(0, f, (1,)).item()
            f_start = torch.randint(0, n_freq - f_len + 1, (1,)).item()
            mel[:, f_start:f_start + f_len, :] = 0

        for _ in range(self.n_time_masks):
            t = min(self.time_mask_param, n_time)
            t_len = torch.randint(0, t, (1,)).item()
            t_start = torch.randint(0, n_time - t_len + 1, (1,)).item()
            mel[:, :, t_start:t_start + t_len] = 0

        return mel
