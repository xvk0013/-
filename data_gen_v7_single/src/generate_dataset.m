function out=generate_dataset(cfg)
% 一次运行：固定来源划分 -> 公共声学信道 -> Train/Test/Val -> 最终汇总。
out=cfg.run_root;
assert(~exist(out,'dir'),'Output exists; use a new run ID: %s.',out);
assert(exist('resample','file')~=0,'Signal Processing Toolbox is required.');
partition_path=prepare_source_split(cfg);
plans=source_plans(cfg,partition_path);
mkdir(out); save(fullfile(out,'RUN_CONFIG.mat'),'cfg');
fprintf('V6: Train %d / Test %d / Val %d scenes.\n',sum(cfg.scene_counts_by_split,2));
summary=cell(0,18); complete=false; failure='none';
source_rows=cell(0,6);
try
    [channels,geometry,acoustics]=prepare_channels(cfg);
    channel_power=cellfun(@(c) abs(c.H).^2,channels,'UniformOutput',false);
    save(fullfile(out,'ACOUSTICS.mat'),'geometry','acoustics');
    for si=1:numel(cfg.split_types)
        current=cfg; current.active_split=cfg.split_types{si};
        current.scene_counts=cfg.scene_counts_by_split(si,:);
        current.max_attempts_per_combination=max(cfg.attempts.minimum_per_combination, ...
            cfg.attempts.factor_per_requested_scene*current.scene_counts);
        current.output_root=fullfile(out,'splits',current.active_split);
        current.input_root=fullfile(out,'inputs',current.active_split);
        mkdir(current.output_root); mkdir(current.input_root);
        for slot=1:numel(cfg.class_ids)
            mkdir(fullfile(out,'components',current.active_split,cfg.combo_folders{slot+1}));
        end
        [sources,recordings]=prepare_sources(current,plans{si});
        source_rows=[source_rows;vertcat(sources.row)]; %#ok<AGROW>
        [caps,groups]=source_caps(current,recordings);
        shape=[numel(sources) numel(cfg.ranges_km) numel(cfg.source_depths_m)];
        cache=struct('sources',sources,'recordings',recordings,'recording_caps',caps, ...
            'groups',{groups},'channels',{channels},'channel_power',{channel_power}, ...
            'geometry',geometry,'source_ffts',{cell(numel(sources),1)}, ...
            'source_fft_ticks',zeros(numel(sources),1),'source_fft_count',0, ...
            'waves',{cell(shape)},'wave_ticks',zeros(shape),'wave_count',0,'clock',0, ...
            'energy_table',nan(shape),'energy_estimates',nan(shape), ...
            'total_clip_uses',zeros(1,numel(sources)));
        fprintf('Sources ready: %s; receiver energies will be computed on demand.\n',current.active_split);
        results=cell(size(cfg.combo_slots));
        % Fix-D 只用 Train 的船舶场景拟合，Test/Val 共用该参考。
        for k=[2:numel(cfg.combo_slots) 1]
            if k==1
                path=fullfile(out,'NOISE_LEVEL_REFERENCE.mat');
                if strcmp(current.active_split,'Train')
                    levels=[];
                    for j=2:numel(results), levels=[levels results{j}.mix_rms]; end %#ok<AGROW>
                    assert(numel(levels)>=2 && all(isfinite(levels) & levels>0), ...
                        'Insufficient valid Train ship scenes for Fix-D.');
                    reference=struct('method','train_ship_mix_log_rms_empirical', ...
                        'source_split','Train','log_rms_sorted',sort(log(levels)));
                    save(path,'reference');
                else
                    saved=load(path,'reference'); reference=saved.reference;
                end
                cache.noise_level_reference=reference;
            end
            [result,cache]=generate_scenes(current,cache,k); results{k}=result;
            summary(end+1,:)={current.active_split,cfg.combo_names{k},result.requested, ...
                result.accepted,result.attempted,result.shortage, ...
                result.primary_sir_scenes,result.supplement_sir_scenes,result.outside_sir_scenes, ...
                result.primary_fraction_actual,result.supplement_fraction_actual, ...
                result.card_band_mismatch_scenes,result.used_recordings,result.used_clips, ...
                result.unique_pairs,result.numeric_rejected,result.pairing_rejected,result.stop_reason}; %#ok<AGROW>
        end
    end
    complete=all(cell2mat(summary(:,6))==0);
catch err
    failure=[err.identifier ': ' err.message];
    write_summary(out,summary,source_rows);
    save(fullfile(out,'RUN_RESULT.mat'),'complete','failure');
    rethrow(err);
end
write_summary(out,summary,source_rows);
save(fullfile(out,'RUN_RESULT.mat'),'complete','failure');
fprintf('Saved %d/%d scenes. Complete=%d\n%s\n',sum(cell2mat(summary(:,4))), ...
    sum(cfg.scene_counts_by_split,'all'),complete,out);
end

function write_summary(out,rows,source_rows)
write_dataset_table(fullfile(out,'SUMMARY.tsv'), ...
    {'split','combination','requested','saved','attempted','shortage', ...
    'primary_SIR_count','supplement_SIR_count','outside_SIR_count', ...
    'primary_SIR_fraction','supplement_SIR_fraction','card_band_mismatch_count', ...
    'used_recordings','used_clips','unique_pairs','numeric_rejected','pairing_rejected','stop_reason'},rows);
write_dataset_table(fullfile(out,'recording_split_manifest.tsv'), ...
    {'class_id','class_name','split','recording_key','segment_file','segment_index'},source_rows);
end

function plans=source_plans(cfg,partition_path)
% 三个集合都从固定录音归属读取全部映射候选，供 QC 后按比例补足。
partition=read_source_table(partition_path,5);
assert(all(ismember(partition(:,3),cfg.split_types)) && ...
    all(ismember(str2double(partition(:,1)),cfg.class_ids)), 'Unknown split/class in recording partition.');
plans=cell(size(cfg.split_types));
for si=1:numel(plans), plans{si}=struct('pools',{cell(size(cfg.class_ids))}); end
for ci=1:numel(cfg.class_ids)
    id=cfg.class_ids(ci); assigned=partition(str2double(partition(:,1))==id,:);
    assert(all(strcmp(assigned(:,2),cfg.class_names{ci})) && ...
        numel(unique(lower(string(assigned(:,4)))))==size(assigned,1), ...
        'Invalid class, duplicate recording or cross-split recording in partition.');
    mapping=read_source_table(fullfile(cfg.mapping_root,num2str(id),'segment_mapping.txt'),4);
    mapping_keys=lower(string(mapping(:,2))+"/"+string(mapping(:,3)));
    for si=1:numel(plans)
        split=cfg.split_types{si}; keys=assigned(strcmp(assigned(:,3),split),4);
        eligible=ismember(mapping_keys,lower(string(keys))) & str2double(mapping(:,4))>=2;
        candidates=mapping(eligible,:); count=size(candidates,1);
        selected=[repmat({num2str(id),cfg.class_names{ci},split},count,1), ...
            cellstr(mapping_keys(eligible)),candidates(:,1),candidates(:,4)];
        keys=unique(lower(string(selected(:,4)))); ranks=zeros(numel(keys),1);
        for j=1:numel(keys), ranks(j)=v6_seed(sprintf('%d|%s',id,keys(j)),cfg.source_pool.selection_seed); end
        [~,order]=sort(ranks); keys=keys(order);
        groups=struct('key',{},'rows',{});
        for j=1:numel(keys)
            clips=selected(lower(string(selected(:,4)))==keys(j),:);
            clip_ranks=zeros(size(clips,1),1);
            for k=1:size(clips,1)
                clip_ranks(k)=v6_seed(sprintf('clip|%d|%s|%s',id,keys(j),clips{k,6}), ...
                    cfg.source_pool.selection_seed);
            end
            [~,order]=sort(clip_ranks);
            groups(j)=struct('key',char(keys(j)),'rows',{clips(order,:)});
        end
        plans{si}.pools{ci}=groups;
    end
end
end

function partition_path=prepare_source_split(cfg)
% Build once from provenance metadata; subsequent runs reuse the frozen split.
% No audio decoding, denoising or reuse of a clip-level Train/Test/Val folder.
partition_path=fullfile(cfg.source_split_root,'RECORDING_PARTITION.tsv');
if exist(partition_path,'file')==2
    fprintf('Reusing frozen recording split: %s\n',partition_path); return;
end
partition=cell(0,5);
recording_counts=zeros(numel(cfg.class_ids),numel(cfg.split_types));
for ci=1:numel(cfg.class_ids)
    id=cfg.class_ids(ci); name=cfg.class_names{ci};
    mapping_path=fullfile(cfg.mapping_root,num2str(id),'segment_mapping.txt');
    mapping=read_source_table(mapping_path,4);
    indices=str2double(mapping(:,4));
    assert(all(isfinite(indices) & indices>=1 & indices==floor(indices)), ...
        'V6:Mapping','Invalid original segment index: %s',mapping_path);
    mapping=mapping(indices>=2,:); % Segment 1 has no real three-second history.
    map_keys=lower(string(mapping(:,2))+"/"+string(mapping(:,3)));
    keys=unique(map_keys); ranks=zeros(numel(keys),1);
    wavs=dir(fullfile(cfg.raw_root,name,'**','*.wav'));
    raw_keys=strings(numel(wavs),1);
    for j=1:numel(wavs)
        [~,parent]=fileparts(wavs(j).folder); [~,stem]=fileparts(wavs(j).name);
        raw_keys(j)=lower(string(parent)+"/"+string(stem));
    end
    assert(all(ismember(keys,raw_keys)),'V6:RawSource', ...
        '来源映射中的原始录音不完整：%s。',fullfile(cfg.raw_root,name));
    for j=1:numel(keys)
        ranks(j)=v6_seed(sprintf('split|%d|%s',id,keys(j)),cfg.source_split.seed);
    end
    [~,order]=sort(ranks); keys=keys(order);
    exact=numel(keys)*cfg.source_split.recording_fractions;
    counts=floor(exact);
    [~,remainder_order]=sort(exact-counts,'descend');
    for j=1:numel(keys)-sum(counts)
        counts(remainder_order(j))=counts(remainder_order(j))+1;
    end
    recording_counts(ci,:)=counts;
    available=zeros(numel(keys),1);
    for j=1:numel(keys), available(j)=nnz(map_keys==keys(j)); end
    offset=0;
    for si=1:numel(cfg.split_types)
        assigned=offset+(1:counts(si)); offset=offset+counts(si);
        target=cfg.split_target_segments(si);
        assert(sum(available(assigned))>=target,'V6:SplitCapacity', ...
            '%s/%s 的录音划分不足 %d 个候选片段。',cfg.split_types{si},name,target);
        for j=assigned
            partition(end+1,:)={id,name,cfg.split_types{si},char(keys(j)),available(j)}; %#ok<AGROW>
        end
        fprintf('New source split %s/%s: %d recordings, %d mapped candidates.\n', ...
            cfg.split_types{si},name,counts(si),sum(available(assigned)));
    end
end
if ~exist(cfg.source_split_root,'dir'), mkdir(cfg.source_split_root); end
protocol=cfg.source_split;
protocol.recording_order='seeded_SHA256_rank_within_class';
protocol.clip_selection='proportional_mapped_candidate_counts_with_QC_backfill_at_each_run';
protocol.created_at=char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
protocol.raw_root=cfg.raw_root; protocol.mapping_root=cfg.mapping_root;
protocol.class_ids=cfg.class_ids; protocol.class_names=cfg.class_names;
protocol.splits=cfg.split_types; protocol.recording_counts=recording_counts;
protocol.QC_eligible_clips_per_class=cfg.split_target_segments;
protocol.old_deleted_split_restored=false;
protocol.QC='raw_resampled_history_and_core_before_normalization_at_each_run';
save(fullfile(cfg.source_split_root,'SOURCE_SPLIT.mat'),'protocol');
write_dataset_table(partition_path, ...
    {'class_id','class_name','split','recording_key','available_clips_with_history'},partition);
fprintf('New recording split saved for reuse: %s\n',partition_path);
end

function rows=read_source_table(path,ncols)
% Preserve numeric-looking filenames and IDs as text (no readtable inference).
lines=readlines(path,'Encoding','UTF-8');
lines=lines(strlength(strtrim(lines))>0);
assert(~isempty(lines),'Empty manifest: %s',path);
header=reshape(cellstr(split(lines(1),sprintf('\t'))),1,[]);
assert(numel(header)==ncols,'Wrong header width: %s',path);
rows=cell(numel(lines)-1,ncols);
for k=2:numel(lines)
    fields=reshape(cellstr(strtrim(split(lines(k),sprintf('\t')))),1,[]);
    assert(numel(fields)==ncols,'Wrong column count, row %d: %s',k,path);
    rows(k-1,:)=fields;
end
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

