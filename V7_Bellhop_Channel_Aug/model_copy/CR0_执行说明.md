# M2 Mamba V11：CR=0 单变量对照

独立复制自 ../M2_Mamba_V11，仅将 config.py 的 cr_weight 从 0.05 改为 0.0。其余已有代码逐字节保留。没有复制历史 runs、检查点或缓存。

新增 export_val_predictions.py 只复用原诊断脚本中的 Val 预测导出函数，评估保存的最佳 EMA，不进行 BN 重估。

## 1. 进入环境与新模型目录

```bash
conda activate LSJ
cd "/mnt/d/LSJ/Code/UART/MUART V6.0/UART M2/M2_Mamba_V11_CR0"
```

## 2. 从头训练

```bash
python train.py \
  --phase single \
  --data-dir "/mnt/d/LSJ/Data/data_gen_v6_no_noise/20260916_225548_748/dataset" \
  --output-dir runs/single_v11_cr0_s42 \
  --device cuda --scan-backend cuda \
  --batch-size 32 --workers 4 --seed 42 --epochs 60
```

不传 --resume 或 --init-checkpoint。和原实验一样使用 AMP、EMA、BCE、查询正交项、频率掩蔽、patience=20，仍按单目标 Macro-F1 选择最佳检查点。达到早停条件时可在第 60 轮之前结束。前端统计仍使用同样的 Train 确定性采样规则。

CR 原始损失仍会计算和记录，但乘以 0，不参与训练梯度；这不是 CR 仍然开启。该保留方式避免另行修改训练与损失代码。

## 3. 训练结束后，导出最佳 EMA 的 Val 指标与误判方向

```bash
python export_val_predictions.py \
  --checkpoint runs/single_v11_cr0_s42/best.pt \
  --data-dir "/mnt/d/LSJ/Data/data_gen_v6_no_noise/20260916_225548_748/dataset" \
  --device cuda --workers 4
```

结果在检查点同级新建的 val_report_时间戳 目录，包含 summary.json、confusion_ema.csv 和 predictions_ema.csv。沿用检查点 batch size、前端统计和独立 0.5 阈值，使用 FP32，与原诊断结果口径一致。

本实验不读取 Test，不改原实验目录。代码仅经静态检查，训练和评估由用户运行。
