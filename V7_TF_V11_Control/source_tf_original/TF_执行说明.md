# 当前 V11 数据与训练方案下的 TF 主干对照

本目录独立复制自 ../M2_Mamba_V11_CR0_CE。原代码和实验结果未改动；未复制历史 runs、检查点或缓存。

## 对照范围

- TFConvNeXtBlock 直接复制自 ../M2_V6/model.py，采用并行 1×31 时间核、11×1 频率核、GroupNorm 和 4 倍通道扩展，沿用旧 TF 模块的初始化方式。
- 保留当前 Mamba V11 的 stem、阶段通道数 (32,64,128,256)、阶段深度 (1,1,4,2)、下采样层及输出 BN，只替换各阶段处理模块。
- 默认输入 [B,2,513,79]，输出仍为 [B,256,33,20]，保持查询分类头输入尺寸一致。
- 按层定义估算总参数约 244 万，原 Mamba 为 227 万；实际数量由训练启动时打印并写入 config.json。
- 数据、前端、DEMON、分类头、CR=0、CE=0.2、查询正交项、增强、EMA、AMP、优化器、早停和选择指标保持原配置。
- 此对照比较的是 TF 模块与原 MambaVision 阶段模块，不是逐项复现旧 M2_V6 整套模型。TF 模块的归一化和初始化属于本次主干替换的一部分。
- TF 不调用扫描算子；scan_backend 的兼容字段设为 torch，并不表示使用 CPU。以下命令仍在 CUDA 上训练。

## 1. 进入 WSL 环境和新目录

```bash
conda activate LSJ
cd "/mnt/d/LSJ/Code/UART/MUART V6.0/UART M2/M2_TF_V11_CR0_CE"
```

## 2. 从头训练

```bash
python train.py \
  --phase single \
  --data-dir "/mnt/d/LSJ/Data/data_gen_v6_no_noise/20260916_225548_748/dataset" \
  --output-dir runs/single_v11_tf_cr0_ce02_s42 \
  --device cuda \
  --batch-size 32 --workers 4 --seed 42 --epochs 60
```

不加载 Mamba 检查点，不传 --resume 或 --init-checkpoint。沿用 patience=20，按单目标 Macro-F1 保存最佳 EMA。

## 3. 训练结束后导出最佳 EMA 的 Val 结果

```bash
python export_val_predictions.py \
  --checkpoint runs/single_v11_tf_cr0_ce02_s42/best.pt \
  --data-dir "/mnt/d/LSJ/Data/data_gen_v6_no_noise/20260916_225548_748/dataset" \
  --device cuda --workers 4
```

复用现有导出脚本。结果位于检查点同级 val_report_时间戳 文件夹，包括 summary.json、confusion_ema.csv 和 predictions_ema.csv。使用相同的 FP32 评估与独立 0.5 阈值，不重估 BN，不读取 Test。

本次只完成代码修改和静态检查，训练及模型评估由用户执行。
