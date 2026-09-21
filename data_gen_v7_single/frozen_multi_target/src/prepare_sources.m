function [sources,recordings]=prepare_sources(cfg,plan)
% 原始录音 -> 单声道/重采样/去直流 -> 真实前史及核心 QC -> 核心 RMS=1。
N=round(cfg.segment_len_s*cfg.target_fs); P=round(cfg.source_history_s*cfg.target_fs);
context_root=cfg.input_root;
sources=struct('row',{},'context_path',{},'input_gain',{}, ...
    'input_rms_before',{},'input_rms_after',{},'recording_index',{});
recordings=struct('class_id',{},'class_name',{},'recording_key',{},'source_indices',{});
inventory=cell(0,19);
target=cfg.split_target_segments(strcmp(cfg.active_split,cfg.split_types));
for ci=1:numel(cfg.class_ids)
    raw_map=index_recordings(fullfile(cfg.raw_root,cfg.class_names{ci}));
    pool=plan.pools{ci};
    row_counts=arrayfun(@(group) size(group.rows,1),pool);
    cursor=ones(size(row_counts)); kept=zeros(size(row_counts)); recording_ids=zeros(size(row_counts));
    while sum(kept)<target && any(cursor<=row_counts)
        remaining=max(0,row_counts-cursor+1);
        % 按剩余映射候选量分配名额，最大余数取整；拒绝后先在本录音补选。
        needed=min(target-sum(kept),sum(remaining));
        exact=needed*remaining/sum(remaining); extra=floor(exact);
        [~,order]=sort(exact-extra,'descend');
        left=needed-sum(extra);
        extra(order(1:left))=extra(order(1:left))+1;
        goals=kept+extra;
        for j=find(goals>kept)
            rows=pool(j).rows; key=pool(j).key;
            assert(isKey(raw_map,key),'Missing original recording: %s/%s',cfg.class_names{ci},key);
            raw_path=raw_map(key); reason='';
            try
                [raw,original_fs]=audioread(raw_path);
            catch
                raw=[]; original_fs=NaN; reason='RAW_DECODE_FAILED';
            end
            if isempty(reason) && (isempty(raw) || any(~isfinite(raw(:))))
                reason='RAW_EMPTY_OR_NONFINITE';
            end
            if isempty(reason), [wave,filter_delay]=prepare_recording(raw,original_fs,cfg); end
            while cursor(j)<=row_counts(j) && kept(j)<goals(j)
                row=rows(cursor(j),:); cursor(j)=cursor(j)+1;
                k=str2double(row{6}); first=(k-1)*N+1; last=k*N;
                status='PASS'; reasons='none'; gain=NaN; before=NaN; after=NaN; source_index=0; delay=NaN;
                input_path='';
                if ~isempty(reason)
                    status='REJECT'; reasons=reason;
                elseif first-P<1
                    status='REJECT'; reasons='MISSING_PREHISTORY';
                else
                    assert(last<=numel(wave),'Original recording span mismatch: %s segment %d.',raw_path,k);
                    x=wave(first-P:last); delay=filter_delay;
                    history_q=waveform_quality(x(1:P),cfg.target_fs,cfg,P);
                    core_q=waveform_quality(x(P+1:end),cfg.target_fs,cfg,N);
                    checks={history_q,core_q}; parts={'RESAMPLED_HISTORY','RESAMPLED_CORE'};
                    for part=1:2
                        q=checks{part};
                        if ~strcmp(q.status,'PASS')
                            detail=[parts{part} ':' q.reasons];
                            if strcmp(reasons,'none'), reasons=detail; else, reasons=[reasons ';' detail]; end
                            if ismember(q.status,{'REJECT','ERROR'}) || strcmp(status,'PASS'), status=q.status; end
                            if ismember(status,{'REJECT','ERROR'}), break; end
                        end
                    end
                    if ismember(status,{'PASS','REVIEW'}) && ...
                            ~(first-P>delay && last<=numel(wave)-delay)
                        status='REJECT'; reasons='FIR_SUPPORT_CROSSES_RECORD_BOUNDARY';
                    end
                    assert(~strcmp(status,'ERROR'),'Invalid source format: %s segment %d.',raw_path,k);
                    if ismember(status,{'PASS','REVIEW'})
                        before=sqrt(mean(x(P+1:end).^2)); gain=cfg.normalization.target_rms/before;
                        x=x*gain; after=sqrt(mean(x(P+1:end).^2));
                        source_index=numel(sources)+1;
                        input_path=sprintf('inputs/%s/input_%05d.mat',cfg.active_split,source_index);
                        context_path=fullfile(context_root,sprintf('input_%05d.mat',source_index)); fs=cfg.target_fs;
                        save(context_path,'x','fs','-v6');
                        if recording_ids(j)==0
                            recording_ids(j)=numel(recordings)+1;
                            recordings(end+1)=struct('class_id',cfg.class_ids(ci), ...
                                'class_name',cfg.class_names{ci},'recording_key',key,'source_indices',[]); %#ok<AGROW>
                        end
                        recording_index=recording_ids(j);
                        sources(end+1)=struct('row',{row},'context_path',context_path, ...
                            'input_gain',gain,'input_rms_before',before,'input_rms_after',after, ...
                            'recording_index',recording_index); %#ok<AGROW>
                        recordings(recording_index).source_indices(end+1)=numel(sources);
                        kept(j)=kept(j)+1;
                    end
                end
                inventory(end+1,:)={cfg.active_split,row{1},row{2},row{4},row{5},k, ...
                    status,reasons,raw_path,original_fs,first-P,first,last,delay,gain,before,after,source_index,input_path}; %#ok<AGROW>
            end
            if mod(j,10)==0 || j==numel(pool)
                fprintf('Sources %s/%s: %d/%d recordings, %d clips kept.\n', ...
                    cfg.active_split,cfg.class_names{ci},j,numel(pool),sum(kept));
            end
        end
    end
    assert(sum(kept)==target, '%s/%s only has %d QC-eligible clips; %d required.', ...
        cfg.active_split,cfg.class_names{ci},sum(kept),target);
end
write_dataset_table(fullfile(cfg.output_root,'SOURCES.tsv'), ...
    {'split','class_id','class_name','recording_key','segment_file','segment_index', ...
    'QC_status','QC_reasons','raw_path','original_fs','context_start_sample','core_start_sample', ...
    'core_stop_sample','resampler_delay_samples','input_gain','input_rms_before','input_rms_after','source_index','input_path'},inventory);

fprintf('Prepared %s: %d ship recordings / %d clips.\n', ...
    cfg.active_split,numel(recordings),numel(sources));
end

function map=index_recordings(root)
wavs=dir(fullfile(root,'**','*.wav'));
map=containers.Map('KeyType','char','ValueType','char');
for j=1:numel(wavs)
    [~,parent]=fileparts(wavs(j).folder); [~,stem]=fileparts(wavs(j).name);
    key=lower([parent '/' stem]);
    assert(~isKey(map,key),'Ambiguous original recording: %s.',key);
    map(key)=fullfile(wavs(j).folder,wavs(j).name);
end
end

function [x,filter_delay]=prepare_recording(raw,fs_in,cfg)
% 整录音前端；片段 RMS 归一化在切片和 QC 后执行。
x=mean(double(raw),2);
factor=gcd(fs_in,cfg.target_fs); p=cfg.target_fs/factor; q=fs_in/factor;
if fs_in==cfg.target_fs, b=1;
else, [x,b]=resample(x,p,q); end
x=x-mean(x);
filter_delay=(numel(b)-1)/(2*q);
end

function q=waveform_quality(x,fs,cfg,expected_samples)
% 只返回主线使用的判定及原因，保留原有 QC 阈值。
q=struct('status','PASS','reasons','none');
if isempty(x), q.status='REJECT'; q.reasons='EMPTY'; return; end
if any(~isfinite(x(:))), q.status='REJECT'; q.reasons='NONFINITE'; return; end
if all(x(:)==0), q.status='REJECT'; q.reasons='ALL_ZERO'; return; end
if fs~=cfg.target_fs || size(x,1)~=expected_samples || size(x,2)~=1
    q.status='ERROR'; q.reasons='FORMAT_MISMATCH'; return;
end
x=double(x(:));
if all(x==x(1)), q.status='REJECT'; q.reasons='CONSTANT_DC'; return; end
peak=max(abs(x));
reasons={};
n=longest_run(abs(x)<=cfg.quality.near_zero_relative*peak);
if n/fs>=cfg.quality.dropout_min_s, reasons{end+1}='SUSPECT_DROPOUT'; end
if peak<cfg.quality.near_silence_peak, reasons{end+1}='NEAR_SILENCE'; end
flat=abs(x(2:end))>=cfg.quality.clipping_level & abs(diff(x))<=cfg.quality.flat_tolerance;
n=longest_run(flat);
if n+(n>0)>=cfg.quality.clipping_min_samples, reasons{end+1}='SUSPECT_CLIPPING'; end
width=max(1,round(cfg.quality.transient_window_s*fs));
energy=movsum(x.^2,[0 width-1],'Endpoints','shrink');
fraction=max(energy)/max(sum(x.^2),realmin);
if fraction>=cfg.quality.transient_energy_fraction
    reasons{end+1}='SUSPECT_TRANSIENT';
end
if ~isempty(reasons), q.status='REVIEW'; q.reasons=strjoin(reasons,';'); end
end

function n=longest_run(mask)
d=diff([false;mask(:);false]); starts=find(d==1); stops=find(d==-1)-1;
if isempty(starts), n=0; return; end
n=max(stops-starts+1);
end
