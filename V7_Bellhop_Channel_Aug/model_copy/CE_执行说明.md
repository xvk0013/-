# CR=0 + 单目标三分类 CE 辅助监督

独立复制自 ../M2_Mamba_V11_CR0。原代码、结果和检查点均未改动；未复制历史 runs 或缓存。

唯一新增训练项：单目标阶段对舰船样本使用现有三个 presence_logits 计算三分类交叉熵，固定权重 0.2。noise 不参加 CE；没有舰船样本的 batch，其 CE 为 0。CR 仍为 0，BCE 和 0.1 查询正交项保持不变。joint 阶段不启用此辅助项。

history.jsonl 新增 train.single_ce，记录未乘 0.2 的辅助损失，沿用其他损失的 batch 样本数加权汇总方式。验证仍按单目标 Macro-F1 选择最佳 EMA，推理仍使用独立 0.5 阈值。

## 1. 进入 WSL 环境和新目录

```bash
conda activate LSJ
cd "/mnt/d/LSJ/Code/UART/MUART V6.0/UART M2/M2_Mamba_V11_CR0_CE"
```

## 2. 从头训练

```bash
python train.py \
  --phase single \
  --data-dir "/mnt/d/LSJ/Data/data_gen_v6_no_noise/20260916_225548_748/dataset" \
  --output-dir runs/single_v11_cr0_ce02_s42 \
  --device cuda --scan-backend cuda \
  --batch-size 32 --workers 4 --seed 42 --epochs 60
```

不传 --resume 或 --init-checkpoint。AMP、EMA、学习率、增强及 patience=20 均保持原配置；允许正常早停。

## 3. 训练结束后导出最佳 EMA 的 Val 结果

```bash
python export_val_predictions.py \
  --checkpoint runs/single_v11_cr0_ce02_s42/best.pt \
  --data-dir "/mnt/d/LSJ/Data/data_gen_v6_no_noise/20260916_225548_748/dataset" \
  --device cuda --workers 4
```

沿用原导出脚本，仅评估最佳 EMA，不重估 BN。结果保存在检查点同级 val_report_时间戳 文件夹，包含 summary.json、confusion_ema.csv 和 predictions_ema.csv。summary.json 的 config.single_ce_weight 为 0.2，config.cr_weight 为 0.0。

本次只修改代码并做静态检查，未执行训练或模型评估。Test 不参与本实验。
