# V7 单目标 GAP 基线

仅执行一次训练：seed=42、60 epoch、batch=32、GAP、CE=0.2、
weight decay=1e-4，关闭 Feature Mixup、额外谱增强和 late_shared。
不修改模型结构；复用相邻 M2_Mamba_V11_CR0_CE 的模型、前端和损失实现。
train_v7.py 复制自原训练入口，仅适配 V7 路径与生成完成标记。
report_v7.py 复制自原 Train/Val 汇总程序，仅固定输出位置。
原模型目录和原实验文件均不修改。

固定数据：
D:/LSJ/Data/data_gen_v7_single/20260920_163131_193/dataset
船舶和 Noise 都读取本次 V7，首次训练从随机初始化开始，
仅从新 Train 重新估计前端标准化统计（沿用原来的 500 个统计样本）。

沿用原录音均衡采样预算：每 epoch 每船类抽取 1800 个、Noise 1080 个，
共 6480 次采样、203 次更新；60 epoch 共 12180 次更新。
Train 全部 17118 个船舶切片均在候选池内，没有每类 700 个的限制；
每个 epoch 并不是完整遍历全部训练集。累计覆盖数记录在训练历史中。
保留原来每 epoch 的 EMA/online Val 评估，仅 EMA 单目标 EMR 选择最佳模型。
训练结束后只对最佳 EMA 进行一次完整 Train/Val 推理；不读取 Test。

在 WSL 的 LSJ 环境运行：

    conda activate LSJ
    cd "/mnt/d/LSJ/Code/UART/MUART V6.0/UART M2/V7_Single_Baseline"
    bash run_baseline.sh

若中断，重复相同命令会从本次 run/last.pt 继续；绝不使用旧数据的检查点。
完整结束后重复命令直接退出，不会再训练或推理。
只有尚未产生第一份 last.pt 的中断需要单独处理，不自动覆盖目录。

输出均在此文件夹的 output 内：
- run/best.pt、last.pt、config.json、history.jsonl：训练结果。
- report/Train、report/Val：预测、混淆矩阵及录音级明细。
- summary.json：最终统一汇总，包含生成信息、训练历史和 Train/Val 指标。

完成后只需提供：
D:/LSJ/Code/UART/MUART V6.0/UART M2/V7_Single_Baseline/output/summary.json
