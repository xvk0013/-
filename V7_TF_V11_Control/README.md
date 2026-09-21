# V7 单目标：M2_TF_V11_CR0_CE 对照

本目录是独立副本。model.py、backbone.py、m2_components.py 直接复制自 M2_TF_V11_CR0_CE，未修改模型实现；source_tf_original 保存原有程序与说明副本。原目录和已有结果均未修改。

## 本次只运行一个配置

- 当前数据：D:/LSJ/Data/data_gen_v7_single/20260920_163131_193/dataset，保持完整 Bellhop 链、扩充源池和现有录音隔离划分。
- 复用 V7_Bellhop_Channel_Aug/output/alternate_train：训练船舶原/备选声道各50%；不生成新数据，不新增 Wenz 增强。
- TF通道32/64/128/256、深度1/1/4/2；原Query192+DEMON128及共享评分头。数量/双目标头保留冻结。
- 总参数2442469，可训练2367649；保留3层BN。
- 当前单目标损失BCE+0.2船类CE，CR/SupCon/Query正交0。原V11历史训练有Query正交项，本次与当前V7对照一致，不是旧训练协议完整复现。
- seed42从头训练一次，60轮、batch32；每轮各船类1800+noise1080，共6480抽样、203成功更新。原SpecAugment、AdamW、EMA、Train特征统计保持一致。
- 已修复的EMA BN流程：每轮验证前使用固定461条原Train，15个无梯度前向批次；不使用Val更新BN。校准后EMA随检查点保存，最终报告直接加载。
- 最高EMA Val单目标EMR选模，三logit独立0.5阈值。最终评估完整Train/Val，不读取Test。
- Mamba是GAP，本模型是Query；完整V6 TF无BN。比较整套结构，不宣称纯主干消融。

## WSL运行

~~~bash
conda activate LSJ
cd "/mnt/d/LSJ/Code/UART/MUART V6.0/UART M2/V7_TF_V11_Control"
bash run_control.sh
~~~

沿用现有LSJ环境，无须另装模型库。中断重跑同一命令，从last.pt完整轮次续训；完成后重跑只返回已有汇总，不追加实验。保留已有WSL检查点替换重试。

依赖现有V7数据及V7_Bellhop_Channel_Aug/output/alternate_train，请保留。参考报告已复制到reference，模型运行不依赖原M2_TF_V11_CR0_CE。

只回传：
D:/LSJ/Code/UART/MUART V6.0/UART M2/V7_TF_V11_Control/output/summary.json

汇总包含Train/Val、各类严格指标、录音等权结果、混淆、Mamba与完整V6 TF参考、训练历史、参数、耗时和显存。无需另传output/run或output/report。

准备阶段仅合成张量检查通过：原模型前向一致、CUDA AMP反向、BN校准、检查点恢复及入口导入。未读取真实音频、未启动正式训练或文件哈希审计。

