function out=generate_single_dataset(cfg)
% 唯一正式入口：全量源池 -> 固定录音划分 -> Bellhop单目标 -> Wenz/Fix-D。
assert(~cfg.multi_target_enabled,'V7:Frozen','多目标生成已冻结。');
assert(cfg.single_use_each_source_once,'V7必须一源片段对应一次单目标输出。');
assert(strcmp(cfg.split_types{1},'Train'),'Train必须先运行，以拟合noise参考电平。');
assert(strcmp(cfg.profile,'full_dataset'),'V7本轮只生成一个完整版本，不串联pilot或实验。');
assert(exist('resample','file')~=0,'需要Signal Processing Toolbox。');
out=cfg.run_root; assert(~isfolder(out),'输出已存在，请使用新的运行时间。');
mkdir(out); mkdir(cfg.dataset_root); save(fullfile(out,'RUN_CONFIG.mat'),'cfg');
started=tic;
report=struct('status','running','revision',cfg.revision,'run_root',out, ...
    'dataset_dir',cfg.dataset_root,'ship_audio_bellhop',true, ...
    'single_boundary_policy',cfg.single_boundary_policy, ...
    'training_runs',0,'multi_target_scenes',0,'failure','');
generation=struct([]);
try
    [channels,geometry,acoustics]=prepare_channels(cfg);
    save(fullfile(out,'ACOUSTICS.mat'),'geometry','acoustics');
    [source_sets,recording_sets,source_report]=prepare_sources_expanded(cfg);
    report.source=source_report;
    fprintf('V7 scope: %d recordings / %d unique ship slices; one propagated scene per slice; zero dual-target scenes.\n', ...
        source_report.retained_recordings,source_report.single_slices);
    cfg.scene_counts_by_split=zeros(numel(cfg.split_types),numel(cfg.combo_slots));
    for si=1:numel(cfg.split_types)
        sources=source_sets{si};
        for ci=1:numel(cfg.class_ids)
            cfg.scene_counts_by_split(si,ci+1)=nnz([sources.class_id]==cfg.class_ids(ci));
        end
        cfg.scene_counts_by_split(si,1)=round(numel(sources)*cfg.noise_per_ship);
    end
    save(fullfile(out,'RUN_CONFIG.mat'),'cfg');
    channel_power=cellfun(@(c) abs(c.H).^2,channels,'UniformOutput',false);
    for si=1:numel(cfg.split_types)
        current=cfg; current.active_split=cfg.split_types{si};
        current.scene_counts=cfg.scene_counts_by_split(si,:);
        current.max_attempts_per_combination=max(cfg.attempts.minimum_per_combination, ...
            cfg.attempts.factor_per_requested_scene*current.scene_counts);
        current.output_root=fullfile(out,'splits',current.active_split);
        current.input_root=fullfile(out,'inputs',current.active_split);
        sources=source_sets{si}; recordings=recording_sets{si};
        [caps,groups]=source_caps(current,recordings);
        shape=[numel(sources) numel(cfg.ranges_km) numel(cfg.source_depths_m)];
        cache=struct('sources',sources,'recordings',recordings,'recording_caps',caps, ...
            'groups',{groups},'channels',{channels},'channel_power',{channel_power}, ...
            'geometry',geometry,'source_ffts',{cell(numel(sources),1)}, ...
            'source_fft_ticks',zeros(numel(sources),1),'source_fft_count',0, ...
            'waves',{cell(shape)},'wave_ticks',zeros(shape),'wave_count',0,'clock',0, ...
            'energy_table',nan(shape),'energy_estimates',nan(shape), ...
            'total_clip_uses',zeros(1,numel(sources)));
        results=cell(size(cfg.combo_slots));
        % 严格只执行三类单目标及独立noise。多目标配对实现保留但不调用。
        for k=[2 3 4 1]
            if k==1
                reference_path=fullfile(out,'NOISE_LEVEL_REFERENCE.mat');
                if strcmp(current.active_split,'Train')
                    levels=[results{2}.mix_rms results{3}.mix_rms results{4}.mix_rms];
                    assert(numel(levels)>=2 && all(isfinite(levels) & levels>0),'无有效Train传播电平。');
                    reference=struct('method','train_single_bellhop_log_rms_empirical', ...
                        'source_split','Train','log_rms_sorted',sort(log(levels)));
                    save(reference_path,'reference');
                    report.noise_reference=struct('source_split','Train','ship_scenes',numel(levels), ...
                        'minimum_rms',min(levels),'median_rms',median(levels),'maximum_rms',max(levels));
                else
                    saved=load(reference_path,'reference'); reference=saved.reference;
                end
                cache.noise_level_reference=reference;
            end
            [result,cache]=generate_scenes(current,cache,k); results{k}=result;
            row=rmfield(result,'mix_rms'); row.split=current.active_split; row.combination=cfg.combo_names{k};
            if isempty(generation), generation=row; else, generation(end+1)=row; end %#ok<AGROW>
            assert(result.shortage==0,'V7:Shortage','%s/%s缺少%d个输出。', ...
                current.active_split,cfg.combo_names{k},result.shortage);
            if k>1
                assert(result.used_clips==result.requested,'单目标没有完整覆盖唯一源片段。');
            end
        end
        clear cache;
    end
    report.status='complete'; report.generation=generation;
    report.ship_samples=sum([generation(~strcmp({generation.combination},'noise')).accepted]);
    report.noise_samples=sum([generation(strcmp({generation.combination},'noise')).accepted]);
    report.total_samples=report.ship_samples+report.noise_samples;
    assert(report.ship_samples==source_report.single_slices,'源池与单目标输出数量不一致。');
    report.seconds=toc(started); report.files_to_return={cfg.summary_copy};
    write_report(cfg,report);
catch err
    report.status='failed'; report.failure=[err.identifier ': ' err.message];
    report.generation=generation; report.seconds=toc(started);
    write_report(cfg,report); rethrow(err);
end
fprintf('COMPLETE: %d single-target + %d noise; Bellhop ON; multi-target OFF.\nDataset: %s\nReturn only: %s\n', ...
    report.ship_samples,report.noise_samples,cfg.dataset_root,cfg.summary_copy);
end

function write_report(cfg,report)
body=jsonencode(report,'PrettyPrint',true);
path=fullfile(cfg.run_root,'summary.json'); fid=fopen(path,'wt','n','UTF-8');
assert(fid>=0,'无法写汇总。'); close_file=onCleanup(@() fclose(fid)); fprintf(fid,'%s\n',body);
clear close_file;
folder=fileparts(cfg.summary_copy); if ~isfolder(folder), mkdir(folder); end
copyfile(path,cfg.summary_copy);
end

function [caps,groups]=source_caps(cfg,recordings)
% Recording capacity follows its selected clip count, not an equal recording share.
groups=cell(1,numel(cfg.class_ids)); caps=zeros(numel(recordings),numel(cfg.combo_slots));
for ci=1:numel(groups)
    groups{ci}=find([recordings.class_id]==cfg.class_ids(ci));
    assert(~isempty(groups{ci}),'V6:SourcePool','No recording for one class.');
end
for k=1:numel(cfg.combo_slots)
    for ci=cfg.combo_slots{k}
        group=groups{ci}; q=cfg.scene_counts(k);
        counts=arrayfun(@(r) numel(r.source_indices),recordings(group));
        cap=max(cfg.source_reuse.minimum_cap, ...
            ceil(cfg.source_reuse.capacity_headroom*q*counts/sum(counts)));
        caps(group,k)=cap(:);
    end
end
end

