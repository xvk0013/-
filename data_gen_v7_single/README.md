# data_gen_v7_single

唯一正式入口：RUN_DATA_GEN_V7_SINGLE.m。全部配置在入口顶部，MATLAB直接运行。
本版本从原始DeepShip生成经过Bellhop传播的三类单目标和独立noise。
不使用B_expanded或无Bellhop导出作为训练音频来源；不自动训练模型。

## 运行

MATLAB命令窗口：

~~~matlab
run('D:\LSJ\Code\UART\MUART V6.0\UART M2\data_gen_v7_single\RUN_DATA_GEN_V7_SINGLE.m')
~~~

需要MATLAB和Signal Processing Toolbox。项目已复制V6的Bellhop程序、许可证、
声学模板、ARR、来源资料和历史组件。声学参数相同且ARR完整时复用ARR，
仍对每个输出船舶样本实际执行传播；这不等于跳过Bellhop信道。

正式数据写到 D:\LSJ\Data\data_gen_v7_single\<运行时间>\dataset。
每个合格5秒源片段随机选择一个既有范围内的距离/深度，生成一次传播后的单目标。
不是每片段遍历39个几何。Noise约为船舶样本数的1/5，按各集合数量取整。
一轮只生成一个数据版本，不串联实验。

## 固定规则

- 先按 excluded_recordings.txt 跳过两条原始录音，不删除原文件。
  此名单来自无Bellhop条件下的规则1补评，不宣称已验证传播后仍不可识别。
- 遍历Cargo/Tanker/Tug全部原始录音，不依赖旧映射的入池范围。
- 整录音单声道、MATLAB有理数重采样至16kHz、去整录音直流。
- 每个完整5秒核心做现有QC，PASS/REVIEW保留。
  只拒绝空、非有限、全零、恒定核心；没有规则2或RMS均值80%过滤。
- 核心RMS归一化到1，同一增益作用于传播上下文，不做峰值缩放。
- 同一录音全部核心只进入一个集合；每类按合格单目标切片数量接近70/15/15，
  录音数允许小幅调整。只按类别、片段数量和固定种子分配，不按识别成绩选片。
- 首次结果固定到 sources/v7_single_split；以后复用，原V6划分不会控制本版本。
  数据范围、数量或划分设置改变时明确停止，需要指定新的source_split_root。
- 保留原几何、信道构造、时延窗口、传播/截窗及固定输出增益。
  单目标经过传播后不再重新归一化到RMS=1。
- Noise沿用Wenz谱形和Fix-D：仅用本次Train传播单目标的输出RMS拟合电平分布，
  Train/Val/Test独立合成波形、共享Train参考；不拟合Val/Test，不向船舶波形叠加noise。
- 16kHz、5秒、IEEE float32 WAV。沿用扩池导出的显式float WAV写法，
  保留可能超过正负1的幅值，不做PCM转换或峰值压缩。

## 真实前史与多目标冻结

单目标不因缺少完整3秒前史或前史QC不合格而丢弃合格核心。
因果传播仍需要信道记忆：有真实前史就使用，录音开始之前的未知部分按零初始状态处理。
不对每个5秒片段重新人为启停；内段使用同一录音的真实连续前史。
起始片段可能含卷积启动边界影响，metadata明确记录history_padding_samples，
不将零填充称为真实前史，也不宣称这消除了录音起始边界效应。

每个源片段保留8秒double输入MAT、归一化增益、源录音与片段位置。
multi_target_eligible仅在真实3秒前史齐全、前史QC合格、FIR边界合格时为true。
该资格标记不影响当前单目标入池数量。
未来多目标必须只使用eligible=true的来源，并沿用此录音级划分。

多目标的SIR配对、传播、混合、容量规则均保留。
主入口multi_target_enabled=false；调度仅执行noise与三类单目标；
generate_scenes对多目标调用提前报错。frozen_multi_target保存原V6完整主线源码。
本版本不支持仅切一个开关就直接产出新的多目标数据：
恢复时须接入eligible筛选和新配额，并显式解除冻结入口。

## 数据结构与模型读取

dataset/Cargo|Tanker|Tug/Train|Val|Test：每个传播WAV一份及all_info.txt。
dataset/noise/Train|Val|Test/mix：独立noise WAV；上一级all_info.txt。
inputs/<split>：每个源片段的8秒上下文一份，不额外输出未传播训练WAV。
splits/<split>/SOURCES.tsv：源片段、真实前史、输入增益和未来多目标资格。
RECORDING_PARTITION.tsv、EXCLUDED_RECORDINGS.txt随数据保存；不是独立审计。
audio_path相对dataset根目录；input_path、s1_path等来源路径相对运行根目录。
此布局兼容现有deepship_v6_slices读取器；以后训练时data-dir和noise-data-dir
都指向同一个新dataset目录。旧检查点不代表此新数据版本的性能。

## 只需回传一个文件

D:\LSJ\Code\UART\MUART V6.0\UART M2\data_gen_v7_single\output\summary.json

这是最新一轮运行汇总的快捷副本，包含实际数据路径、各类/各集合录音与切片数、
排除数量、前史资格、单目标和noise输出数、完成状态。
原运行目录保留同一份summary.json。正式生成仍由用户执行。
源片段预计约24455，最终以实际QC和输出汇总为准，noise另计。

程序会保留8秒double输入，全部数据预计占用约35GB，实际受片段数量影响。
中断或失败会保留已写文件和失败汇总；本版不自动续跑，也不覆盖已有运行目录。
再次运行会创建新时间目录，但复用固定录音划分。
