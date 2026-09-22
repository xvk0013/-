# 历史实验：录音均衡 + DEMON 50–4000 Hz

本实验已完成。当前默认配置已切换为指定目录中的 PTT 三类切片实验，请使用 [PTT 切片执行说明](当前实验_PTT切片_执行说明.md)。旧检查点仍按保存的类别、前端和数据配置加载。

继续使用 M2_Mamba_V11_CR0_CE 代码目录。M2_Mamba_V11_CR0_CE_RB 保留为对照，已有 runs 和检查点不覆盖。

## 本次改动

以 RB + DEMON 开启版为基准，只将 DEMON 载波频带从 50–1000 Hz 扩展到 50–4000 Hz。当前默认 use_demon_features=True，恢复 DEMON 特征和编码器训练。平方检波、100 Hz 低通、约 0–50 Hz 调制谱、去 DC 和标准化均沿用原设置。

LOFAR 仍参与识别：0–1000 Hz、8192 点 FFT、约 1.95 Hz 分辨率。它与约 1–4 kHz 高频谱组成双通道时频输入；DEMON 是额外的调制谱分支。本次没有改动这两个时频通道。

Mamba 主干、分类头、CR=0、CE=0.2、录音均衡采样、数据增强、EMA、AMP、阈值和选模方式保持原设置。按单目标 Macro-F1 选择最佳 EMA，patience=20。

## 1. 在 WSL 终端进入环境和代码目录

```bash
conda activate LSJ
cd "/mnt/d/LSJ/Code/UART/MUART V6.0/UART M2/M2_Mamba_V11_CR0_CE"
```

## 2. 从头训练

```bash
python train.py \
  --phase single \
  --data-dir "/mnt/d/LSJ/Data/data_gen_v6_no_noise/20260916_225548_748/dataset" \
  --output-dir runs/single_v11_cr0_ce02_rb_demon50_4000_s42 \
  --device cuda --scan-backend cuda \
  --batch-size 32 --workers 4 --seed 42 --epochs 60
```

本次从头训练，不添加 --resume 或 --init-checkpoint。启动日志应包含：

```text
DEMON features enabled=True; carrier=50-4000 Hz; modulation=0-50 Hz; single sampler=record_balanced
```

## 3. 训练结束后导出 Val 结果

```bash
python export_val_predictions.py \
  --checkpoint runs/single_v11_cr0_ce02_rb_demon50_4000_s42/best.pt \
  --data-dir "/mnt/d/LSJ/Data/data_gen_v6_no_noise/20260916_225548_748/dataset" \
  --device cuda --workers 4
```

结果在本次 runs 目录内的 val_report_时间戳 文件夹中。比较 summary.json 和 confusion_ema.csv，重点看单目标 Macro-F1、EMR、Cargo→Tanker 和 Tug 完全识别正确率。Test 不参与本实验。

旧检查点仍使用保存时的配置：关闭 DEMON 的检查点保持 False 和 50–1000 Hz；旧 RB 检查点保持 DEMON 开启和 50–1000 Hz。本次新检查点保存 True 和 50–4000 Hz。
