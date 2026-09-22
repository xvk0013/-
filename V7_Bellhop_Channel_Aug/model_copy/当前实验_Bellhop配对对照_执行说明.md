# 当前实验：Bellhop 配对对照

数据：D:/LSJ/Data/deepship_5s_no_bellhop/20260916_225548_748_paired。
沿用 M2_Mamba_V11_CR0_CE 的 deepship_v6_slices 读取接口，无需更改主干或损失。
两组仅改变输入子目录与结果目录，从头训练；相同 seed=42、batch=32、epochs=60、workers=4、CR=0、CE=0.2、录音均衡、DEMON 50–1000 Hz、EMA 和单目标 EMR 选模。
Noise 均来自 /mnt/d/LSJ/Data/data_gen_v6_no_noise/20260916_225548_748/dataset，打印 noise recall/precision/F1/FA。
两组各 Train：1838/类 + 1386 noise = 6900；Val：394/类 + 291 noise = 1473。
不加载 Test。每个脚本训练成功后自动导出本组 Val 报告，训练失败则停止。
保留源切片重复次数。传播参考组保留原 WAV 量化影响；结论针对传播及原写出链路，不单凭结果认定 Bellhop 算法错误。

## 在 WSL 的 LSJ 环境中依次运行

```bash
conda activate LSJ
cd "/mnt/d/LSJ/Code/UART/MUART V6.0/UART M2/M2_Mamba_V11_CR0_CE"
bash run_paired_no_bellhop.sh
bash run_paired_bellhop_reference.sh
```

结果分别位于：
- runs/single_paired_no_bellhop_cr0_ce02_rb_emr_s42/val_report_时间戳/
- runs/single_paired_bellhop_reference_cr0_ce02_rb_emr_s42/val_report_时间戳/

比较这两次新运行的 summary.json 和 confusion_ema.csv；之前独立切片集结果只作背景参考。
