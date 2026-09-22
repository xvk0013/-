# 历史实验：录音均衡 + DEMON 特征关闭

本文记录已完成的关闭 DEMON 实验，下面的从头训练指令不再对应当前默认配置。当前实验请使用 [DEMON 50–4000 Hz 执行说明](当前实验_DEMON宽频带_执行说明.md)。已保存的关闭 DEMON 检查点仍按其自身配置加载。

本次直接复用 M2_Mamba_V11_CR0_CE 目录，不再复制整套代码。M2_Mamba_V11_CR0_CE_RB 保留为对照，已有 runs、报告和检查点未修改。

## 单变量说明

以 RB 最佳方案为基线：Mamba、CR=0、CE=0.2、录音均衡和全部训练设置保持一致。唯一新增变化为 use_demon_features=False：DEMON 编码器仍按原顺序构建并保留所有检查点键，但冻结其参数，并在编码后将送入分类头的特征置零。分类头宽度、时频输入、EMA、AMP、阈值和选模口径保持不变。

当前 DEMON 前置带通为 50–1000 Hz，检波后低通为 100 Hz，输出为约 0–50 Hz 的调制谱并去 DC。这次不改频带；前端提取与编码路径仍保留，以保持接口和初始化方式一致。总参数结构保留，可训练参数数量会因冻结 DEMON 编码器而减少。

旧检查点兼容：未记录 use_demon_features 的旧检查点按 True 加载；未记录 single_train_sampler 的旧检查点按 scene_shuffle 加载。新检查点明确记录 False 和 record_balanced。现有预测、评估和恢复入口都经过同一加载函数，不会将旧模型误当成关闭 DEMON 的模型。

## 1. 进入原 CE 目录

```bash
conda activate LSJ
cd "/mnt/d/LSJ/Code/UART/MUART V6.0/UART M2/M2_Mamba_V11_CR0_CE"
```

## 2. 从头训练，使用新的输出目录

```bash
python train.py \
  --phase single \
  --data-dir "/mnt/d/LSJ/Data/data_gen_v6_no_noise/20260916_225548_748/dataset" \
  --output-dir runs/single_v11_cr0_ce02_rb_nodemon_s42 \
  --device cuda --scan-backend cuda \
  --batch-size 32 --workers 4 --seed 42 --epochs 60
```

不传 --resume 或 --init-checkpoint。启动时应显示 DEMON features enabled=False 和 single sampler=record_balanced。仍按单目标 Macro-F1 选择最佳 EMA，patience=20。

## 3. 训练结束后导出 Val 结果

```bash
python export_val_predictions.py \
  --checkpoint runs/single_v11_cr0_ce02_rb_nodemon_s42/best.pt \
  --data-dir "/mnt/d/LSJ/Data/data_gen_v6_no_noise/20260916_225548_748/dataset" \
  --device cuda --workers 4
```

结果位于新检查点同级 val_report_时间戳 文件夹，包含 summary.json、confusion_ema.csv 和 predictions_ema.csv。summary.json 中 config.use_demon_features 应为 false。

仅完成代码和配置兼容检查，未加载音频、实际检查点或运行模型。Test 不参与本实验。
