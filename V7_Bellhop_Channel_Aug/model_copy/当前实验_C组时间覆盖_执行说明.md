# C 组：固定 700 条、扩大时间覆盖

C 从 B 的合格训练切片中选择；每类 700 条，每条录音配额严格等于 A。
候选按 start_sample 排序，按排序分位点等间隔选取，包含两端；配额为 1 时取中间候选。
这是在合格候选序列上分散取样，不承诺实际秒数间隔完全相等。
不读取 Val 预测来筛选，不删困难录音，不改变 QC、标签或声学处理。
Val 从 A 原样保留，Test 不读取。没有新增/重新处理音频。
优先使用硬链接，不支持时复制；不要编辑链接后的 WAV 或元数据。
输入比例、训练预算、AMP 同批重试、EMA 选模和自动恢复与修复后的 A/B 相同。
原 A/B 结果保留，本次只训练 C。

## WSL 执行顺序
```bash
conda activate LSJ
cd "/mnt/d/LSJ/Code/UART/MUART V6.0/UART M2/M2_Mamba_V11_CR0_CE"
python prepare_source_pool_c.py
bash run_source_pool_C_time_spread.sh
```

生成目录：D:/LSJ/Data/deepship_source_pool_ab/20260916_225548_748/C_time_spread。
已存在时拒绝覆盖；如果先前已成功生成，仅执行训练命令即可。
脚本自动训练、导出 Val、按源切片/录音汇总，并输出 review_candidates.json。
两条 Cargo 和一条 Tug 仅是事后观察对象，状态为待复核；识别差不作为无效判据。
没有依据模型表现修改标签、删除验证样本或报告剔除后的主指标。

提供 C/selection_by_recording.csv 和
runs/single_pool_C_time_spread_ship5_noise1_s42/val_report_*/ 下的
summary.json、confusion_ema.csv、unique_source_summary.json、per_recording.csv、review_candidates.json。

判断：C 接近 B 支持原选择覆盖问题；C 接近 A 支持更大候选池的价值；
中间结果支持两者均有影响。单 seed 筛方向，稳定性仍需后续重复实验。
