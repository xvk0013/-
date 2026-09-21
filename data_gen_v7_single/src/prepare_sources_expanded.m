function [source_sets,recording_sets,report]=prepare_sources_expanded(cfg)
% 全部原始录音 -> 核心QC -> 单目标全量源池 -> 按录音固定划分。
% 真实前史/FIR边界仅决定 multi_target_eligible，不筛除单目标核心。
N=round(cfg.segment_len_s*cfg.target_fs); P=round(cfg.source_history_s*cfg.target_fs);
assert(N==80000 && P==48000,'V7使用16 kHz、5秒核心、3秒信道记忆。');
excluded=cellstr(strtrim(readlines(cfg.exclusion_file,'Encoding','UTF-8')));
excluded=excluded(~cellfun(@isempty,excluded));
excluded=strrep(excluded,'\','/');
assert(numel(unique(lower(string(excluded))))==numel(excluded),'排除名单有重复路径。');
for j=1:numel(excluded)
    pieces=strsplit(excluded{j},'/');
    assert(numel(pieces)>=2 && ismember(pieces{1},cfg.class_names) && ...
        ~any(ismember(pieces,{'','..','.'})) && ~contains(excluded{j},':'),'非法排除路径。');
end
found=false(size(excluded));
staging=fullfile(cfg.run_root,'inputs','preparing'); mkdir(staging);
records=struct('class_id',{},'class_name',{},'raw_relative_path',{},'clips',{});
class_report=struct([]); serial=0;
for ci=1:numel(cfg.class_names)
    name=cfg.class_names{ci}; folder=fullfile(cfg.raw_root,name);
    files=dir(fullfile(folder,'**','*.wav'));
    assert(~isempty(files),'类别没有原始WAV：%s',folder);
    paths=cell(numel(files),1);
    for j=1:numel(files)
        full=fullfile(files(j).folder,files(j).name);
        paths{j}=strrep(full(numel(cfg.raw_root)+2:end),'\','/');
    end
    [~,order]=sort(lower(string(paths))); paths=paths(order);
    counts=struct('class_name',name,'original_recordings',numel(paths), ...
        'excluded_recordings',0,'excluded_complete_slices',0,'invalid_recordings',0, ...
        'recordings_without_valid_core',0,'retained_recordings',0,'single_slices',0, ...
        'multi_target_eligible_slices',0,'padded_start_slices',0,'rejected_core_slices',0, ...
        'review_core_slices',0,'discarded_tail_seconds',0);
    for ri=1:numel(paths)
        relative=paths{ri}; raw_path=fullfile(cfg.raw_root,relative);
        match=find(strcmpi(excluded,relative));
        if ~isempty(match)
            found(match)=true; counts.excluded_recordings=counts.excluded_recordings+1;
            info=audioinfo(raw_path);
            length16=ceil(info.TotalSamples*cfg.target_fs/info.SampleRate);
            counts.excluded_complete_slices=counts.excluded_complete_slices+floor(length16/N);
            continue;
        end
        try
            [raw,original_fs]=audioread(raw_path);
        catch err
            warning('V7:Decode','跳过无法解码的录音 %s：%s',relative,err.message);
            counts.invalid_recordings=counts.invalid_recordings+1; continue;
        end
        if isempty(raw) || any(~isfinite(raw(:)))
            counts.invalid_recordings=counts.invalid_recordings+1; continue;
        end
        wave=mean(double(raw),2); delay=0;
        if original_fs~=cfg.target_fs
            factor=gcd(original_fs,cfg.target_fs); p=cfg.target_fs/factor; q=original_fs/factor;
            [wave,b]=resample(wave,p,q); delay=(numel(b)-1)/(2*q);
        end
        wave=wave-mean(wave);
        total=floor(numel(wave)/N);
        counts.discarded_tail_seconds=counts.discarded_tail_seconds+(numel(wave)-total*N)/cfg.target_fs;
        clips=struct([]);
        for k=1:total
            first=(k-1)*N+1; last=k*N;
            core=wave(first:last); core_q=v7_waveform_quality(core,cfg.target_fs,cfg,N);
            if strcmp(core_q.status,'REJECT')
                counts.rejected_core_slices=counts.rejected_core_slices+1; continue;
            end
            assert(~strcmp(core_q.status,'ERROR'),'核心格式错误：%s',relative);
            left=first-P; padding=max(0,1-left);
            x=[zeros(padding,1);wave(max(1,left):last)];
            assert(numel(x)==P+N,'上下文长度错误。');
            if padding==0
                history_q=v7_waveform_quality(x(1:P),cfg.target_fs,cfg,P);
            else
                history_q=struct('status','REJECT','reasons','MISSING_REAL_PREHISTORY');
            end
            fir_supported=left>delay && last<=numel(wave)-delay;
            eligible=padding==0 && ismember(history_q.status,{'PASS','REVIEW'}) && fir_supported;
            before=sqrt(mean(core.^2)); gain=cfg.normalization.target_rms/before;
            x=x*gain; fs=cfg.target_fs;
            real_history_samples=P-padding; history_padding_samples=padding;
            multi_target_eligible=eligible;
            serial=serial+1; temporary=fullfile(staging,sprintf('source_%08d.mat',serial));
            save(temporary,'x','fs','real_history_samples','history_padding_samples', ...
                'multi_target_eligible','-v6');
            item=struct('temporary',temporary,'segment_index',k,'original_fs',original_fs, ...
                'core_start_sample',first,'core_stop_sample',last,'resampler_delay_samples',delay, ...
                'QC_status',core_q.status,'QC_reasons',core_q.reasons, ...
                'history_QC_status',history_q.status,'history_QC_reasons',history_q.reasons, ...
                'history_padding_samples',padding,'fir_supported',fir_supported, ...
                'multi_target_eligible',eligible,'input_gain',gain, ...
                'input_rms_before',before,'input_rms_after',sqrt(mean(x(P+1:end).^2)));
            if isempty(clips), clips=item; else, clips(end+1)=item; end %#ok<AGROW>
            counts.multi_target_eligible_slices=counts.multi_target_eligible_slices+eligible;
            counts.padded_start_slices=counts.padded_start_slices+(padding>0);
            counts.review_core_slices=counts.review_core_slices+strcmp(core_q.status,'REVIEW');
        end
        if isempty(clips)
            counts.recordings_without_valid_core=counts.recordings_without_valid_core+1;
        else
            records(end+1)=struct('class_id',cfg.class_ids(ci),'class_name',name, ...
                'raw_relative_path',relative,'clips',clips); %#ok<AGROW>
            counts.retained_recordings=counts.retained_recordings+1;
            counts.single_slices=counts.single_slices+numel(clips);
        end
        if mod(ri,10)==0 || ri==numel(paths)
            fprintf('Prepare %s: %d/%d recordings, %d single-target slices.\n', ...
                name,ri,numel(paths),counts.single_slices);
        end
    end
    if isempty(class_report), class_report=counts; else, class_report(end+1)=counts; end %#ok<AGROW>
end
assert(all(found),'排除名单中的录音未全部找到；请核对原始数据根目录。');
[assignment,partition]=frozen_partition(cfg,records,excluded);
write_dataset_table(fullfile(cfg.run_root,'RECORDING_PARTITION.tsv'),partition);
copyfile(cfg.exclusion_file,fullfile(cfg.run_root,'EXCLUDED_RECORDINGS.txt'));
source_sets=cell(size(cfg.split_types)); recording_sets=source_sets;
header={'split','class_id','class_name','recording_key','segment_file','segment_index', ...
    'QC_status','QC_reasons','raw_path','original_fs','context_start_sample','core_start_sample', ...
    'core_stop_sample','resampler_delay_samples','input_gain','input_rms_before','input_rms_after', ...
    'source_index','input_path','raw_relative_path','history_padding_samples', ...
    'history_QC_status','history_QC_reasons','fir_supported','multi_target_eligible'};
split_report=struct([]);
for si=1:numel(cfg.split_types)
    split=cfg.split_types{si}; input_root=fullfile(cfg.run_root,'inputs',split);
    table_root=fullfile(cfg.run_root,'splits',split); mkdir(input_root); mkdir(table_root);
    sources=struct([]); recording_info=struct([]); inventory=cell(0,numel(header));
    for ri=find(assignment(:).'==si)
        record=records(ri); recording_index=numel(recording_info)+1; indices=[];
        for j=1:numel(record.clips)
            clip=record.clips(j); index=numel(sources)+1;
            input_path=sprintf('inputs/%s/input_%05d.mat',split,index);
            destination=fullfile(cfg.run_root,input_path);
            assert(startsWith(destination,[cfg.run_root filesep]),'输入目标越界。');
            movefile(clip.temporary,destination);
            segment_file=sprintf('rec_%05d_seg_%06d.wav',ri,clip.segment_index);
            row={num2str(record.class_id),record.class_name,split,record.raw_relative_path, ...
                segment_file,num2str(clip.segment_index)};
            source=rmfield(clip,'temporary');
            source.row=row; source.context_path=destination; source.input_path=input_path;
            source.recording_index=recording_index; source.raw_relative_path=record.raw_relative_path;
            source.class_name=record.class_name; source.class_id=record.class_id;
            if isempty(sources), sources=source; else, sources(end+1)=source; end %#ok<AGROW>
            indices(end+1)=index; %#ok<AGROW>
            inventory(end+1,:)={split,record.class_id,record.class_name,record.raw_relative_path, ...
                segment_file,clip.segment_index,clip.QC_status,clip.QC_reasons, ...
                fullfile(cfg.raw_root,record.raw_relative_path),clip.original_fs, ...
                clip.core_start_sample-P,clip.core_start_sample,clip.core_stop_sample, ...
                clip.resampler_delay_samples,clip.input_gain,clip.input_rms_before,clip.input_rms_after, ...
                index,input_path,record.raw_relative_path,clip.history_padding_samples, ...
                clip.history_QC_status,clip.history_QC_reasons,clip.fir_supported,clip.multi_target_eligible}; %#ok<AGROW>
        end
        item=struct('class_id',record.class_id,'class_name',record.class_name, ...
            'recording_key',record.raw_relative_path,'source_indices',indices);
        if isempty(recording_info), recording_info=item; else, recording_info(end+1)=item; end %#ok<AGROW>
    end
    source_sets{si}=sources; recording_sets{si}=recording_info;
    write_dataset_table(fullfile(table_root,'SOURCES.tsv'),header,inventory);
    for ci=1:numel(cfg.class_ids)
        group=find([sources.class_id]==cfg.class_ids(ci));
        total=class_report(ci).single_slices;
        item=struct('split',split,'class_name',cfg.class_names{ci}, ...
            'recordings',nnz([recording_info.class_id]==cfg.class_ids(ci)), ...
            'single_slices',numel(group),'target_slice_fraction',cfg.source_split.recording_fractions(si), ...
            'actual_slice_fraction',numel(group)/total, ...
            'multi_target_eligible_slices',nnz([sources(group).multi_target_eligible]));
        if isempty(split_report), split_report=item; else, split_report(end+1)=item; end %#ok<AGROW>
    end
end
rmdir(staging); % 仅删除已移空的本次临时输入目录，不递归删除任何音频。
report=struct('classes',class_report,'splits',split_report,'excluded_recordings',{excluded}, ...
    'original_recordings',sum([class_report.original_recordings]), ...
    'excluded_recording_count',sum([class_report.excluded_recordings]), ...
    'excluded_complete_slices',sum([class_report.excluded_complete_slices]), ...
    'retained_recordings',sum([class_report.retained_recordings]), ...
    'single_slices',sum([class_report.single_slices]), ...
    'multi_target_eligible_slices',sum([class_report.multi_target_eligible_slices]), ...
    'padded_start_slices',sum([class_report.padded_start_slices]));
end

function [assignment,partition]=frozen_partition(cfg,records,excluded)
root=cfg.source_split_root;
path=fullfile(root,'RECORDING_PARTITION.tsv'); setting_path=fullfile(root,'SPLIT_SETTINGS.mat');
settings=struct('seed',cfg.source_split.seed,'fractions',cfg.source_split.recording_fractions, ...
    'split_types',{cfg.split_types},'class_names',{cfg.class_names}, ...
    'excluded_recordings',{excluded},'boundary_policy',cfg.single_boundary_policy, ...
    'recording_slack_fraction',cfg.source_split.recording_slack_fraction, ...
    'max_balance_steps',cfg.source_split.max_balance_steps);
assignment=zeros(numel(records),1);
if isfile(path) || isfile(setting_path)
    assert(isfile(path) && isfile(setting_path),'冻结划分不完整，请指定新的source_split_root。');
    saved=load(setting_path,'settings');
    assert(isequaln(settings,saved.settings),'划分配置改变，请指定新的source_split_root。');
    old=readtable(path,'FileType','text','Delimiter','\t','TextType','string');
    keys=lower(string({records.raw_relative_path}));
    [present,positions]=ismember(keys,lower(old.raw_relative_path));
    assert(height(old)==numel(records) && all(present) && ...
        numel(unique(positions))==numel(records),'原始录音范围改变，请创建新的划分版本。');
    for j=1:numel(records)
        row=positions(j);
        assert(old.single_slices(row)==numel(records(j).clips) && ...
            old.multi_target_eligible_slices(row)==nnz([records(j).clips.multi_target_eligible]), ...
            '录音切片数量改变，请创建新的划分版本。');
        s=find(strcmp(cfg.split_types,old.split(row)));
        assert(isscalar(s),'未知集合。'); assignment(j)=s;
    end
    fprintf('Reusing frozen V7 recording partition.\n');
else
    for ci=1:numel(cfg.class_ids)
        group=find([records.class_id]==cfg.class_ids(ci));
        counts=arrayfun(@(r) numel(r.clips),records(group));
        assignment(group)=balance_recordings(counts,cfg.source_split.recording_fractions, ...
            cfg.source_split.seed+cfg.class_ids(ci),cfg.source_split.recording_slack_fraction, ...
            cfg.source_split.max_balance_steps);
    end
end
partition=struct([]);
for j=1:numel(records)
    item=struct('class_id',records(j).class_id,'class_name',records(j).class_name, ...
        'raw_relative_path',records(j).raw_relative_path,'split',cfg.split_types{assignment(j)}, ...
        'single_slices',numel(records(j).clips), ...
        'multi_target_eligible_slices',nnz([records(j).clips.multi_target_eligible]));
    if isempty(partition), partition=item; else, partition(end+1)=item; end %#ok<AGROW>
end
if ~isfile(path)
    if ~isfolder(root), mkdir(root); end
    write_dataset_table(path,partition); save(setting_path,'settings');
end
end
