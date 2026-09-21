% V7：完整 Bellhop 单目标 + 独立 noise；多目标组件保留但冻结。
% 所有配置集中在本文件。真实3秒前史仅作为未来多目标资格条件。

%% 路径与运行规模
project_root=fileparts(mfilename('fullpath'));
addpath(fullfile(project_root,'src'),'-begin');
cfg=struct();
cfg.data_root='D:\LSJ\Data';
cfg.output_base=fullfile(cfg.data_root,'data_gen_v7_single');
cfg.profile='full_dataset';
cfg.multi_target_enabled=false;             % 本版本禁止多目标产出
cfg.single_use_each_source_once=true;       % 每个合格源片段只生成一次单目标
cfg.noise_per_ship=1/5;                     % 单目标:noise=5:1
cfg.exclusion_file=fullfile(project_root,'excluded_recordings.txt');
cfg.summary_copy=fullfile(project_root,'output','summary.json');
cfg.bellhop_executable=fullfile(project_root,'tools','bellhop','bellhop.exe');

%% 类别、来源划分与输入音频
cfg.class_ids=[0 1 3]; cfg.class_names={'Cargo','Tanker','Tug'};
cfg.raw_root=fullfile(cfg.data_root,'DeepShip');
cfg.split_types={'Train','Val','Test'};
% 无700/150限额：原始录音中的全部合格完整5秒核心均入单目标源池。
cfg.source_split=struct('seed',47,'recording_fractions',[0.70 0.15 0.15]);
cfg.source_pool.selection_seed=47;
cfg.source_split.balance_unit='QC_eligible_single_target_slices';
cfg.source_split.recording_slack_fraction=0.02; % 录音数允许小幅调整，优先接近切片比例
cfg.source_split.max_balance_steps=100;
cfg.single_boundary_policy='real_available_history_zero_initial_state_before_recording';
cfg.target_fs=16000; cfg.segment_len_s=5; cfg.source_history_s=3;
cfg.normalization=struct('method','mean_power','scope','effective_5s_input','target_rms',1);
cfg.resampler.method='MATLAB_rational_polyphase_resample';
cfg.quality=struct('near_zero_relative',1e-4,'dropout_min_s',0.25, ...
    'near_silence_peak',1e-5,'clipping_level',0.98,'flat_tolerance',1/32768, ...
    'clipping_min_samples',4,'transient_window_s',0.05,'transient_energy_fraction',0.50);

%% 独立 noise 类：Wenz 谱形合成，RMS 归一化后使用 Train Fix-D 电平
% 参数范围为合成实验设定；三个集合同分布、每场景重新生成随机波形。
cfg.noise=struct('model','wenz_three_component_shape', ...
    'wind_speed_range_m_s',[0 10],'shipping_factor_range',[0 1], ...
    'turbulence_min_hz',10); % 沿用旧生成器的湍流低频截断约定

%% 公共接收机与 Bellhop
cfg.ranges_km=1:0.25:4; cfg.single_ranges_km=cfg.ranges_km;
cfg.source_depths_m=[5 10 15]; cfg.receiver_x_km=0; cfg.receiver_depth_m=1000;
cfg.frequencies_hz=50:100:7950; cfg.max_delay_s=3;
cfg.geometry_method='reciprocal_fixed_receiver_same_side';
cfg.receiver_model='continuous_sources_full_5s_receiver_sum';

%% 配对：低使用次数小批搭档 -> 频域排序 -> 按需精算；保留目标牌堆和 L4
cfg.physical_seed=48;
cfg.pairing=struct('primary_fraction',0.80,'primary_bounds_db',[0 5], ...
    'supplement_bounds_db',[5 10],'deck_primary_draw_db',[0.3 4.7], ...
    'deck_supplement_draw_db',[5.3 9.7],'matcher_batch_size',8, ...
    'candidate_switches_max',2,'keeper_redraws_max',1, ...
    'max_pair_uses',2,'minimum_clip_cap',2,'clip_capacity_headroom',2);
cfg.source_reuse=struct('minimum_cap',10,'capacity_headroom',2); % 录音容量按入池片段量分配
cfg.attempts=struct('minimum_per_combination',2000,'factor_per_requested_scene',50);
cfg.cache=struct('waveforms_in_memory',128,'source_ffts_in_memory',8);
cfg.progress_every_scenes=100;

%% 输出配额：列为 noise / Cargo / Tanker / Tug / 三种双目标
cfg.combo_slots={[],1,2,3,[1 2],[1 3],[2 3]};
cfg.combo_names={'noise','Cargo','Tanker','Tug','Cargo_Tanker','Cargo_Tug','Tanker_Tug'};
cfg.combo_folders={'noise','Cargo','Tanker','Tug','0_1','0_2','1_2'};
% 单目标配额由实际源片段数确定；双目标配额固定为0。
% Noise 数量按各集合实际船舶样本量计算，不复制旧noise波形。
cfg.output_bits=32; cfg.output_encoding='IEEE_float32'; cfg.output_gain=1;
cfg.output_layout='single_named_indexed_v7';

%% 自动路径及执行（通常不需要修改）
cfg.revision='data_gen_v7_single_bellhop_expanded';
cfg.project_root=project_root;
cfg.mapping_root=fullfile(project_root,'sources','mapping');
cfg.source_split_root=fullfile(project_root,'sources','v7_single_split'); % 不复用V6旧划分
cfg.run_id=char(datetime('now','Format','yyyyMMdd_HHmmss_SSS'));
cfg.run_root=fullfile(cfg.output_base,cfg.run_id);
cfg.dataset_root=fullfile(cfg.run_root,'dataset');
v7_output_dir=generate_single_dataset(cfg);
