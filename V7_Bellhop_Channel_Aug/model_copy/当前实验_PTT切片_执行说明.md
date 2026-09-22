# 历史实验：指定 DeepShip 切片目录的 PTT 三类识别

当前默认配置已经切换到新切片的 Tanker/Cargo/Tug 实验。请使用 [新切片执行说明](当前实验_新切片CTT_执行说明.md)。本文件保留历史配置，旧检查点仍使用保存时的 PTT 类别、数据读取和选模设置。

在 M2_Mamba_V11_CR0_CE 中继续修改，不复制模型文件夹。保留旧 runs 和 M2_Mamba_V11_CR0_CE_RB 对照。

## 数据和模型设置

- 直接读取 D:/LSJ/Data/deepship_16k_5s_split 内的 Train、Val WAV；1=Tanker、2=PassengerShip、3=Tug，没有 Cargo。
- 该目录属于历史切片流程，配套代码包含谱减降噪；本实验称为“现有 PTT 切片”，不称为未经处理的原始切片。不额外经过 Bellhop，不读取 source_contexts。
- 保持现有 Train/Val 归属，使用 deepship_16k_5s/{1,2,3}/segment_mapping.txt 获取源录音编号。仍检查录音不跨 Train/Val。
- 从已生成的 v6 数据集中读取同 split 的纯 noise WAV，固定种子选取约占总样本 20% 的子集；每个 epoch 使用同一子集。没有用 PassengerShip 冒充 noise。
- 没有类别专属损失权重、阈值或增强。录音均衡规则对三类相同，保留目录中的各类样本数。
- 沿用 RB + CR=0 + CE=0.2；DEMON 恢复较好基准的 50–1000 Hz，调制谱约 0–50 Hz。LOFAR 0–1000 Hz 和高频谱约 1–4 kHz 保持。
- 从头训练，不复用 Cargo 检查点或其归一化统计。旧检查点仍根据自身 config 使用旧类别和 v11 数据入口。
- 每轮及 Val 报告显示三类 recall、noise recall（正确率及正确数/样本数）、Precision、F1、虚警率。模型选择仍使用船舶单目标 Macro-F1。
- 该实验同时改变了类别组合、样本和预处理，不能把结果直接归因为 Bellhop 或 Cargo。Test 不参与训练和选模。

## 1. 进入环境和代码目录

```bash
conda activate LSJ
cd "/mnt/d/LSJ/Code/UART/MUART V6.0/UART M2/M2_Mamba_V11_CR0_CE"
```

## 2. 从头训练

```bash
python train.py \
  --phase single \
  --data-dir "/mnt/d/LSJ/Data/deepship_16k_5s_split" \
  --mapping-dir "/mnt/d/LSJ/Data/deepship_16k_5s" \
  --noise-data-dir "/mnt/d/LSJ/Data/data_gen_v6_no_noise/20260916_225548_748/dataset" \
  --output-dir runs/single_ptt_slices_cr0_ce02_rb_s42 \
  --device cuda --scan-backend cuda \
  --batch-size 32 --workers 4 --seed 42 --epochs 60
```

本次不加 --resume 或 --init-checkpoint。启动应打印 class order=('Tanker', 'PassengerShip', 'Tug') 和 Train/Val 各类数量。

## 3. 训练完成后导出 Val

```bash
python export_val_predictions.py \
  --checkpoint runs/single_ptt_slices_cr0_ce02_rb_s42/best.pt \
  --data-dir "/mnt/d/LSJ/Data/deepship_16k_5s_split" \
  --device cuda --workers 4
```

映射和 noise 路径从新检查点自动读取。结果在本次运行目录的 val_report_时间戳 中，包含 summary.json、confusion_ema.csv、predictions_ema.csv。报告的第二类应为 PassengerShip。
