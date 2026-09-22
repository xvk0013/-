# 当前实验：新切片 Tanker / Cargo / Tug（无 Bellhop）

继续使用 M2_Mamba_V11_CR0_CE。保留旧 runs、PTT 结果和 M2_Mamba_V11_CR0_CE_RB 对照。

## 输入与训练设置

- 输入运行：D:/LSJ/Data/deepship_5s_four_classes/20260917_233441_628。
- 本轮选择 Tanker、Cargo、Tug；Passengership 不参与。按各类 Train/Val/all_info.txt 读取 FLOAT WAV 和 raw_relative_path，沿用新目录已有录音划分。
- 新切片已做单声道、16 kHz 重采样、整录音去直流、连续 5 秒切片、QC、RMS=1；无谱减、Bellhop、混合或额外 Wenz 加噪。模型仍使用已有输入归一化和前端。
- 新切片工程重新划分录音、导出全部有效片段，并包含首个片段；样本池与旧 v6 不完全一致。本轮是新数据基准，不是严格的 Bellhop 单变量实验。
- noise 来自已有 v6 数据中同 split 的全部纯噪声 WAV，每个 epoch 每条使用一次，不重复扩充到 20%。启动打印实际数量。
- 主干、LOFAR 0–1000 Hz、高频谱约 1–4 kHz、DEMON 载波 50–1000 Hz / 调制谱约 0–50 Hz、CR=0、CE=0.2、录音均衡、优化器、EMA=0.999 保持。
- 新实验按单目标 EMR 选择最佳 EMA，避免把几乎全报三类、但船舶 Macro-F1 接近 50% 的模型选为最佳。原 Macro-F1 和 recall 继续报告。
- 每轮额外在同一 Val 上评估在线模型，并打印、保存 val_online；不更新权重或 BN，不参与 EMA 的选模。用于分辨在线模型与 EMA 的表现差距。
- noise 打印 recall（正确率/正确数量）、Precision、F1、虚警率。仍使用独立 0.5 阈值，不强制 argmax。
- 旧检查点按保存的格式与类别加载；未记录选模字段的旧检查点保留 single_macro_f1 和不额外评估在线模型的行为。Test 不参与本次训练和选模。

## 1. 进入 WSL 环境和代码目录

```bash
conda activate LSJ
cd "/mnt/d/LSJ/Code/UART/MUART V6.0/UART M2/M2_Mamba_V11_CR0_CE"
```

## 2. 从头训练

```bash
python train.py \
  --phase single \
  --data-dir "/mnt/d/LSJ/Data/deepship_5s_four_classes/20260917_233441_628" \
  --noise-data-dir "/mnt/d/LSJ/Data/data_gen_v6_no_noise/20260916_225548_748/dataset" \
  --output-dir runs/single_ctt_v6slices_cr0_ce02_rb_emr_s42 \
  --device cuda --scan-backend cuda \
  --batch-size 32 --workers 4 --seed 42 --epochs 60
```

不要添加 --resume、--init-checkpoint 或 --mapping-dir。新 all_info.txt 已包含原始录音身份。

启动应显示 class order=('Tanker', 'Cargo', 'Tug')、Best checkpoint: EMA single_emr、online Val comparison=True。

## 3. 训练完成后导出 Val 结果

```bash
python export_val_predictions.py \
  --checkpoint runs/single_ctt_v6slices_cr0_ce02_rb_emr_s42/best.pt \
  --data-dir "/mnt/d/LSJ/Data/deepship_5s_four_classes/20260917_233441_628" \
  --device cuda --workers 4
```

结果在本次运行目录的 val_report_时间戳 文件夹。summary.json 包含 EMA 的 metrics 和同一轮在线模型的 saved_online_val；完整逐轮在线/EMA 曲线保存在 history.jsonl。
