# M2 Mamba V11：独立模型框架

本目录是新建的独立模型，不引用父目录的 Python 模块，不修改原 M2 / M2_V6。当前任务仅搭建代码；真实数据尚未生成完，不启动训练。用户提出“先不跑代码”后，未继续执行模型或检查脚本。

## 1. 已搭建内容

- 固定 MambaVision 声学主干，没有 TF/Mamba 主干切换入口。
- 复制原 M2 的双通道声学前端、DEMON 编码器、类别查询、合法集合定义及跨录音 CR 损失。
- 主干通道 `[32,64,128,256]`，深度 `[1,1,4,2]`，前两阶段卷积，后两阶段混合 Mamba/注意力，MLP 扩展倍数 2。
- Mamba 状态维度 8，卷积宽度 3，扩展倍数 1；查询维度 192。
- 总参数 2,271,909；单目标阶段可训练参数 2,197,089，数量头和组合头冻结。
- 单目标训练、完整多目标训练、EMA 验证、保存 best/last、断点续训、独立评估及单 WAV 预测。
- 新版 `all_info.txt` + float32 WAV 数据接口；数据位置通过命令行传入。
- 提供合成输入检查脚本，但当前版本的完整检查尚未执行。

数据未完成前，没有识别准确率、收敛情况或训练吞吐方面的验证结论。

## 2. 结构与输入

```text
mix：16 kHz，5 秒，单声道 float32 WAV
  ├─ 整段 RMS 归一化 → 原 M2 双通道时频特征 [2,513,79]
  │     └─ 声学 MambaVision → [256,33,20]
  │          └─ 1×1 投影 → [192,33,20] → 三个类别查询 [3,192]
  └─ DEMON 特征 [1,126] → 原 M2 DEMON 编码器 [128]
        └─ 拼接查询特征 [3,320] → 三个共享类别证据
```

时频前端保留原 M2 定义：低频 0–1 kHz 长窗 STFT，高频约 1–4 kHz 短窗 STFT，频率插值后组成两个通道。没有将时频图缩放成 224×224 图像。

频率总下采样约 16 倍、时间约 4 倍。第三阶段有 2 个 Mamba 块和 2 个注意力块，第四阶段各 1 个；窗口尺寸 7×7。这是局部窗口混合，再通过类别查询聚合整幅特征，不宣称每个 Mamba 块直接扫描全时频图。

默认从头训练。通道数及输入结构已经改变，不能直接加载官方完整 ImageNet 预训练权重。

## 3. 数据接口

传入路径必须指向本次 v11 的 `dataset` 层级：

```text
dataset/
  noise/Train/mix/*.wav
  noise/Train/all_info.txt
  0/Train/mix/*.wav
  0/Train/all_info.txt
  ...
  1/  2/  0_1/  0_2/  1_2/
  每个组合内部都有 Train、Val、Test
```

`all_info.txt` 按制表符读取。标签由 `label_Tanker`、`label_Cargo`、`label_Tug` 明确构造，模型顺序固定为 **Tanker/Cargo/Tug**。生成器文件夹 `0/1/2` 对应 Cargo/Tanker/Tug；原始 DeepShip 类别 ID 不能直接用作模型下标。

仅 `mix` 进入网络。`s1/s2/s3` 不读取；这些字段对应的辅助监督留待后续研究。Wenz noise 的三个舰船标签均为 0。

float32 波形可以超过 ±1，读取时不截幅、不转 PCM16。整体 RMS 归一化是共同缩放，不改变同一混合内两目标的相对能量比。时频标准化参数只从 Train 抽样计算并存入检查点；Val/Test/预测复用它们。单目标检查点迁移到 joint 时沿用其 Train 统计量。

代码不要求旧版 `PROTOCOL.json`、`SCENES.tsv`，不套用旧 SNR 筛选，也不假设最终实际 SIR 严格满足目标牌面比例。只保留必要的格式、标签和 Train/Val 原录音不交叉检查，不增加生成器审计流程。

## 4. 已确认的 WSL 环境

用户提供的位置：`\\wsl.localhost\Ubuntu-22.04\home\lsj`。

已读取到的 Conda 环境：`/home/lsj/miniconda3/envs/LSJ`。

| 项目 | 已读取版本 |
|---|---|
| Python 环境 | LSJ |
| PyTorch / torchaudio | 2.10.0+cu128 / 2.10.0+cu128 |
| mamba-ssm | 2.3.1 |
| CUDA | 12.8 |
| GPU | RTX 5090 |
| selective_scan_cuda | 扩展可导入 |

默认 `scan_backend=cuda`，使用已安装的真实 selective scan 扩展，不自动安装依赖。环境可导入不等于完整模型前后向和 AMP 已验证，后者仍待运行。

另提供 `--scan-backend torch`：同一状态递推的纯 PyTorch 实现，权重结构一致，可用于 CPU/Windows 排错，预期慢于融合 CUDA 算子。CUDA 出错不会静默切换成另一种网络。

## 5. 后续执行顺序：现在不要执行训练

以下命令留给数据完成后运行。若当前终端是 Windows PowerShell，先进入 WSL：

```powershell
wsl -d Ubuntu-22.04
```

随后在 WSL 的 Bash 终端中执行：

```bash
source /home/lsj/miniconda3/etc/profile.d/conda.sh
conda activate LSJ
cd "/mnt/d/LSJ/Code/UART/MUART V6.0/UART M2/M2_Mamba_V11"
```

如果已经在 VSCode 的 WSL 窗口里，直接使用上述 Bash 命令即可。不要在 WSL 中传 Windows 的 `D:\...` 路径。

### 第一步：需要时检查框架，不需要数据

```bash
python check_environment.py
python smoke_check.py --device cuda --scan-backend cuda --amp
```

检查脚本会创建临时合成波形，验证 float32 读取、标签映射、扫描递推/梯度、两阶段前后向、EMA 与状态保存。它不读取真实数据；这个脚本当前尚未运行。

CPU 排错用：

```bash
python smoke_check.py --device cpu --scan-backend torch
```

### 第二步：数据完成后填写路径，先训练单目标

```bash
DATA_DIR="/mnt/d/LSJ/Data/data_gen_v6_no_noise/替换为本次run_id/dataset"

python train.py --phase single \
  --data-dir "$DATA_DIR" \
  --output-dir runs/single_s42 \
  --device cuda --scan-backend cuda
```

`single` 加载 noise、Cargo、Tanker、Tug；只训练 BCE 类别证据，加原 M2 的查询正交项和 CR。数量头及组合头不参与前向或梯度。

增强固定为两个温和频率掩码，最大宽度参数 24；不做时间掩码、额外白噪声、频移或波形拉伸。CR 权重 0.05、温度 0.1，仅使用同类、不同原录音的单目标正对。没有合适正对的 batch 不强行构造正对。

按 Val 舰船单目标子集 Macro-F1 保存 `best.pt`；同时报告单目标 EMR、各类召回和 noise 误报率。单目标预测用三个独立 sigmoid 的固定 0.5 阈值，不强制 argmax 掩盖多预测/漏预测。

### 第三步：检查单目标结果后，再训练完整多目标

```bash
python train.py --phase joint \
  --data-dir "$DATA_DIR" \
  --init-checkpoint runs/single_s42/best.pt \
  --output-dir runs/joint_from_single_s42 \
  --device cuda --scan-backend cuda
```

加载单目标 EMA 权重，启用数量头与有界类别对交互，训练 noise + 三个单目标 + 三个双目标。损失是七个合法集合 NLL + 0.5 BCE + 原查询正交项 + CR；按 Val EMR 保存最佳权重。主干容量保持不变。

这两个阶段属于同一个 Mamba 模型的训练阶段，不存在 TF 主干切换。

### 第四步：如需中断后继续

```bash
python train.py --phase single \
  --data-dir "$DATA_DIR" \
  --output-dir runs/single_s42 \
  --resume runs/single_s42/last.pt \
  --device cuda
```

joint 同理替换 phase 和输出目录。继续训练使用 `last.pt`；阶段迁移使用 `best.pt`。只加载自己生成或可信的检查点。

每次全新训练使用一个新的输出目录，代码不会覆盖已有实验。保存 `config.json`、`history.jsonl`、`best.pt`、`last.pt`，不生成额外审计目录。默认 batch size 32、60 epochs、patience 20；可通过 `--batch-size`、`--epochs`、`--workers`、`--seed` 调整新实验。

### 第五步：方案确定后，显式评估 Test

```bash
python evaluate.py \
  --checkpoint runs/joint_from_single_s42/best.pt \
  --data-dir "$DATA_DIR" --split Test --device cuda
```

训练只构建 Train/Val，Test 不参与归一化、早停或模型选择。评估会按保存的实际 SIR 统计双目标分组结果。

单条混合波形预测：

```bash
python predict.py \
  --checkpoint runs/joint_from_single_s42/best.pt \
  --wav "/mnt/d/请替换为实际mix文件.wav" --device cuda
```

## 6. 文件职责

| 文件 | 职责 |
|---|---|
| config.py | 网络、前端与训练默认参数；不硬编码数据路径 |
| frontend.py | 从原 M2 复制的双通道时频/DEMON/SpecAugment |
| m2_components.py | 从原 M2 复制的查询、DEMON 编码器和合法集合模块 |
| mamba_blocks.py / scan.py | 官方 MambaVision 模块适配及选择性状态递推 |
| backbone.py / model.py | 声学下采样主干、类别证据和两阶段分类头 |
| data.py | v11 数据接口、Train 统计量、标签及原录音映射 |
| objective.py / cross_recording.py | 阶段损失及复制的 CR |
| train.py | 训练、EMA、Val 选择、检查点与恢复 |
| evaluate.py / predict.py | 显式评估和单 WAV 预测 |
| check_environment.py / smoke_check.py | 用户后续可运行的环境及合成输入检查 |
| THIRD_PARTY.md / third_party/ | 官方固定版本、原始源码和许可证 |

后续的分量教师监督、类别条件扫描、不同容量搜索均未加入第一版，以便先确认该主干本身的效果。


## 7. 2026-09-17：AMP 首步检查修订

用户运行 CUDA/AMP 检查时，扫描算子的输出/梯度对照已经通过，随后在“首个优化器更新必须成功”的断言处停止。该日志说明发生了 AMP 跳步；仅凭这一次跳步，不能确定是初始缩放校准还是持续的梯度异常。

本次仅修改代码，未代运行：
- 训练和检查脚本统一通过 make_scaler 创建 GradScaler，默认初始缩放从 PyTorch 的 65536 调整为 1024，仍保留动态缩放。
- 合成检查对同一批输入最多允许 8 次缩放调整；每次恢复随机状态，使 dropout 等随机操作保持一致。
- 通过条件仍要求实际优化器更新、类别证据权重确实改变，且可训练参数和梯度均有限。
- 正式训练记录跳步次数、当前缩放和成功更新数；连续 8 次跳步会停止并报告，避免长期没有有效更新。
- 单目标与多目标阶段均使用相同的检查逻辑。没有关闭 AMP、替换主干或改动数据。

重新运行原命令：

```bash
python smoke_check.py --device cuda --scan-backend cuda --amp
```

若持续失败，脚本会给出明确错误。可再用下面的 FP32 命令定位精度相关性：

```bash
python smoke_check.py --device cuda --scan-backend cuda
```

FP32 单独通过不能当作 AMP 已通过；通过完整检查后再启动对应精度的正式训练。


## 8. 2026-09-17：检查点与推理数值比较修订

用户再次运行后，CUDA scan 对照及 AMP 单目标优化器更新已通过（初始 scale=1024，第一次即成功）。随后“single → joint”的类别证据比较因零容差失败：报告最大绝对误差为 2.60770320892334e-07。完整检查仍未完成。

修订仅涉及 smoke_check.py：
- 检查点中的所有参数和持久缓冲区逐项比较，仍严格要求 rtol=0、atol=0。
- 保存加载后的类别证据、重复执行的集合 log 概率，采用 FP32 数值容差 rtol=1e-5、atol=1e-6，并显示最大绝对误差。
- 随机数状态恢复仍要求零容差；保留有限值、梯度、真实参数更新和概率归一化检查。
- 不修改网络、训练损失、AMP 配置或正式推理精度。未代运行本次修订。

浮点运算数值相近不等于逐位相等，参见 PyTorch 2.10 数值说明：
https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html

后续仍运行同一条 CUDA/AMP 检查命令，完整通过后再训练。

## 9. 同一检查点的在线模型、EMA、重估 BN 后 EMA 对照

在 WSL 的 LSJ 环境、当前 M2_Mamba_V11 目录执行：

```bash
python diagnose_ema_bn.py \
  --checkpoint runs/single_v11_20260916_s42/best.pt \
  --data-dir "/mnt/d/LSJ/Data/data_gen_v6_no_noise/20260916_225548_748/dataset" \
  --device cuda --scan-backend cuda \
  --workers 4
```

脚本顺序比较同一轮保存的在线模型（online）、EMA（ema）和只重估 BN 统计量的 EMA（ema_bn_reestimated）。默认沿用检查点的 batch size，三者使用相同 Val、FP32、保存的前端归一化参数和独立 0.5 判定阈值。

BN 重估仅在内存中的 EMA 副本上进行：完整读取一遍打乱顺序的单目标阶段 Train（含 noise），按 batch 样本数累计 BN 统计量，关闭增强、Dropout 和 DropPath。没有优化器更新，不读取 Test，不保存或覆盖模型检查点，也不修改训练流程。

结果写入检查点同级新目录 `ema_bn_diagnostic_时间戳`：

- `summary.md`：三种状态的指标对照，以及各类实际误判方向、数量和占真实类别的比例。
- `summary.json`：完整指标、检查点原始 EMA 验证指标和诊断配置。
- `confusion_*.csv`：三份混淆矩阵，行是真实类别，列包含 noise、三个单类、三个双类和三类同时预测。
- `predictions_*.csv`：三份逐样本结果，包含 WAV 路径、原录音、真实/预测类别集合、各类概率和 logit。

注意区分 Cargo→Tug 与 Cargo→Cargo+Tug：前者是单类错认，后者是多报。此次只新增诊断入口，运行结果由用户执行后确认。
