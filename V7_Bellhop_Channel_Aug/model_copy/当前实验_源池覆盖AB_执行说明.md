# 源池覆盖 A/B：固定船舶与 noise 为 5:1
原 50% 单目标 / 40% 多目标 / 10% noise 中去除多目标，将保留比例归一化：
船舶 83.33%，noise 16.67%。每轮三类各 1800、noise 1080，共 6480 条。
两组 batch=32、每轮 203 批、60 轮共 12180 次更新；不提前停止。
AMP 溢出后降低 scale，恢复 BN 等缓冲区和随机数状态，重试同一批已增强输入；仅成功更新推进 optimizer/EMA/学习率。每轮仍为 203 次成功更新，同批连续失败 8 次才终止。运行脚本自动从 last.pt 恢复最后完整轮次，未保存轮次重跑。可用 export_source_pool_A_original.sh 和 export_source_pool_B_expanded.sh 只导出现有 best.pt；未完成 60 轮的报告只是中途结果。
模型、损失、前端不变；单目标 EMR 选最佳 EMA，继续打印在线模型和 noise 指标。
A 每类 700 个不同源切片；B 在 A 已覆盖的同一批训练录音内扩充。
原核心缓存复用；新增片段沿用原历史/核心 QC、边界规则和 RMS 归一化。
不引入新录音、不重新划分、不读取 Test、不生成多目标、不运行 Bellhop。
B 的原子集音频与 A 字节一致，Val 两组都原样复制现有 paired/no_bellhop Val。
Val 保留原始分布（每类 394、noise 291），不为凑训练比例重采样验证集。
两组 noise 来自相同原 noise 池，每轮相同抽样；录音抽样顺序一致。
训练标准化统计各自由本组 Train 拟合；旧 checkpoint 不续训。
history.jsonl 的 source_pool_coverage 记录每轮及累计不同训练切片覆盖。

## 1. MATLAB 生成一次
```matlab
run('D:\LSJ\Code\UART\MUART V6.0\UART M2\M2_Mamba_V11_CR0_CE\EXPORT_SOURCE_POOL_AB.m')
```
输出 D:/LSJ/Data/deepship_source_pool_ab/20260916_225548_748，
包含 A_original、B_expanded、SUMMARY.tsv、COMPLETE.txt。
输出目录已存在时会停止，不覆盖已有数据。

## 2. WSL 中运行 A
```bash
conda activate LSJ
cd "/mnt/d/LSJ/Code/UART/MUART V6.0/UART M2/M2_Mamba_V11_CR0_CE"
bash run_source_pool_A_original.sh
```

## 3. A 完成后运行 B
```bash
bash run_source_pool_B_expanded.sh
```

每个脚本自动训练、导出 Val、汇总不同源切片与录音结果。
提交 SUMMARY.tsv，以及两组 runs/single_pool_<组名>_ship5_noise1_s42/val_report_*/ 内的：
summary.json、confusion_ema.csv、unique_source_summary.json、per_recording.csv。
旧训练入口与数据保持可用，不要用之前实验结果替代本次 A 基线。
