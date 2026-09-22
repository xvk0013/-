% 单目标源池 A/B；只运行生成，不训练、不运行 Bellhop、不读取 Test。
run_root='D:\LSJ\Data\data_gen_v6_no_noise\20260916_225548_748';
paired_root='D:\LSJ\Data\deepship_5s_no_bellhop\20260916_225548_748_paired\no_bellhop';
output_root='D:\LSJ\Data\deepship_source_pool_ab\20260916_225548_748';
saved=load(fullfile(run_root,'RUN_CONFIG.mat'),'cfg'); cfg=saved.cfg;
assert(~isfolder(output_root),'输出已存在，请修改 output_root：%s',output_root);
assert(cfg.target_fs==16000 && cfg.segment_len_s==5,'要求 16 kHz / 5 秒');
N=round(cfg.target_fs*cfg.segment_len_s); P=round(cfg.target_fs*cfg.source_history_s);
assert(P>0,'要求保留历史上下文');
[sources,sc]=read_manifest(fullfile(run_root,'splits','Train','SOURCES.tsv'));
sources=sources(ismember(sources(:,sc('QC_status')),{'PASS','REVIEW'}),:);
mkdir(output_root);
header={'file_name','audio_path','class_id','class_name','split','raw_relative_path', ...
    'fs','duration_s','QC_status','QC_reasons','segment_index','start_sample','stop_sample', ...
    'original_fs','input_rms_before','normalization_gain','output_rms','in_original_pool'};
names={'Cargo','Tanker','Tug'}; ids=[0 1 3]; arms={'A_original','B_expanded'};
summary=cell(0,5);
for ci=1:3
    name=names{ci}; class_rows=find(strcmp(sources(:,sc('class_name')),name));
    assert(numel(class_rows)==700,'原源池每类应为 700 条：%s',name);
    raw_paths=unique(sources(class_rows,sc('raw_path')),'stable');
    rowsA=cell(0,numel(header)); rowsB=cell(0,numel(header));
    for ai=1:2
        mkdir(fullfile(output_root,arms{ai},name,'Train'));
        copyfile(fullfile(paired_root,name,'Val'),fullfile(output_root,arms{ai},name,'Val'));
    end
    for ri=1:numel(raw_paths)
        raw_path=raw_paths{ri};
        selected=class_rows(strcmp(sources(class_rows,sc('raw_path')),raw_path));
        old_segments=str2double(sources(selected,sc('segment_index')));
        assert(numel(unique(old_segments))==numel(old_segments),'重复源切片');
        relative=relative_recording(raw_path,cfg.raw_root,name);
        [raw,original_fs]=audioread(raw_path);
        assert(~isempty(raw) && all(isfinite(raw(:))),'原录音无效：%s',raw_path);
        [wave,delay]=prepare_recording(raw,original_fs,cfg);
        old_saved=0;
        for k=1:floor(numel(wave)/N)
            first=(k-1)*N+1; last=k*N;
            old=find(old_segments==k);
            if ~isempty(old)
                i=selected(old);
                if isKey(sc,'input_path')
                    path=resolve_path(run_root,sources{i,sc('input_path')});
                else
                    path=fullfile(run_root,'splits','Train','source_contexts',sources{i,sc('context_file')});
                end
                cached=load(path,'x','fs');
                assert(cached.fs==cfg.target_fs && numel(cached.x)==P+N,'缓存格式错误');
                audio=single(cached.x(P+(1:N))); audio=audio(:);
                status=sources{i,sc('QC_status')}; reasons=sources{i,sc('QC_reasons')};
                before=str2double(sources{i,sc('input_rms_before')});
                gain=str2double(sources{i,sc('input_gain')});
                old_saved=old_saved+1;
            else
                if first-P<=delay || last>numel(wave)-delay, continue; end
                x=wave(first-P:last);
                hq=waveform_quality(x(1:P),cfg.target_fs,cfg,P);
                cq=waveform_quality(x(P+1:end),cfg.target_fs,cfg,N);
                assert(~strcmp(hq.status,'ERROR') && ~strcmp(cq.status,'ERROR'),'QC 格式错误');
                if strcmp(hq.status,'REJECT') || strcmp(cq.status,'REJECT'), continue; end
                status='PASS'; reasons='none';
                if strcmp(hq.status,'REVIEW') || strcmp(cq.status,'REVIEW')
                    status='REVIEW';
                    reasons=['RESAMPLED_HISTORY:' hq.reasons ';RESAMPLED_CORE:' cq.reasons];
                end
                core=x(P+1:end); before=sqrt(mean(core.^2));
                gain=cfg.normalization.target_rms/before; audio=single(core*gain);
            end
            assert(all(isfinite(audio)) && any(audio~=0),'输出音频无效');
            file=sprintf('rec_%05d_seg_%06d.wav',ri,k);
            rel=sprintf('%s/Train/%s',name,file);
            row={file,rel,ids(ci),name,'Train',relative,cfg.target_fs,5,status,reasons, ...
                k,first,last,original_fs,before,gain,sqrt(mean(double(audio).^2)),~isempty(old)};
            write_float_wav(fullfile(output_root,'B_expanded',name,'Train',file),audio,cfg.target_fs);
            rowsB(end+1,:)=row; %#ok<SAGROW>
            if ~isempty(old)
                copyfile(fullfile(output_root,'B_expanded',name,'Train',file), ...
                    fullfile(output_root,'A_original',name,'Train',file));
                rowsA(end+1,:)=row; %#ok<SAGROW>
            end
        end
        assert(old_saved==numel(selected),'原切片未完整保留：%s',raw_path);
        if mod(ri,10)==0 || ri==numel(raw_paths)
            fprintf('%s %d/%d recordings: A=%d B=%d\n',name,ri,numel(raw_paths),size(rowsA,1),size(rowsB,1));
        end
    end
    assert(size(rowsA,1)==700 && size(rowsB,1)>=700,'源池数量错误');
    write_table(fullfile(output_root,'A_original',name,'Train','all_info.txt'),header,rowsA);
    write_table(fullfile(output_root,'B_expanded',name,'Train','all_info.txt'),header,rowsB);
    summary(end+1,:)={name,numel(raw_paths),size(rowsA,1),size(rowsB,1),size(rowsB,1)-size(rowsA,1)}; %#ok<SAGROW>
end
write_table(fullfile(output_root,'SUMMARY.tsv'), ...
    {'class_name','same_recordings','A_unique_clips','B_unique_clips','added_clips'},summary);
save(fullfile(output_root,'SOURCE_RUN_CONFIG.mat'),'cfg','run_root','paired_root');
fid=fopen(fullfile(output_root,'COMPLETE.txt'),'wt');
assert(fid>=0,'无法保存完成标记'); fprintf(fid,'Source-pool A/B export complete.\n'); fclose(fid);
fprintf('完成。训练 noise 仍引用原 dataset；Val 原样复制；Test 未读取。\n%s\n',output_root);

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

function [rows,columns]=read_manifest(path)
lines=readlines(path,'Encoding','UTF-8'); lines=lines(strlength(lines)>0);
header=reshape(cellstr(split(lines(1),sprintf('\t'))),1,[]);
columns=containers.Map(header,num2cell(1:numel(header)));
rows=cell(numel(lines)-1,numel(header));
for j=2:numel(lines)
    rows(j-1,:)=reshape(cellstr(split(lines(j),sprintf('\t'))),1,[]);
end
end

function path=resolve_path(root,value)
if ~isempty(regexp(value,'^([A-Za-z]:[\\/]|[\\/])','once'))
    path=value;
else
    path=fullfile(root,value);
end
end

function relative=relative_recording(path,raw_root,class_name)
path=strrep(path,'\','/'); root=strrep(raw_root,'\','/');
root=regexprep(root,'/+$','');
assert(startsWith(path,[root '/'],'IgnoreCase',true),'原录音路径不属于 cfg.raw_root：%s',path);
relative=path(numel(root)+2:end);
parts=strsplit(relative,'/');
assert(numel(parts)>=2 && strcmp(parts{1},class_name) && ...
    ~any(ismember(parts,{'','..','.'})) && endsWith(relative,'.wav','IgnoreCase',true), ...
    '原录音相对路径无效：%s',relative);
end

function write_float_wav(path,x,fs)
% IEEE float32 WAV：保留 RMS=1 后可能超过 +/-1 的幅值，不做峰值缩放。
% WAVEFORMATEX: https://learn.microsoft.com/en-us/windows/win32/api/mmreg/ns-mmreg-waveformatex
n=numel(x); data_bytes=4*n;
fid=fopen(path,'wb','ieee-le');
assert(fid>=0,'Cannot write %s.',path);
close_file=onCleanup(@() fclose(fid));
fwrite(fid,'RIFF','char'); fwrite(fid,50+data_bytes,'uint32'); fwrite(fid,'WAVE','char');
fwrite(fid,'fmt ','char'); fwrite(fid,18,'uint32');
fwrite(fid,[3 1],'uint16'); fwrite(fid,[fs 4*fs],'uint32'); fwrite(fid,[4 32 0],'uint16');
fwrite(fid,'fact','char'); fwrite(fid,4,'uint32'); fwrite(fid,n,'uint32');
fwrite(fid,'data','char'); fwrite(fid,data_bytes,'uint32'); fwrite(fid,x,'single');
end

function write_table(path,header,rows)
% 单份来源索引，数值保留 17 位有效数字。
fid=fopen(path,'wt','n','UTF-8');
assert(fid>=0,'Cannot write %s.',path);
close_file=onCleanup(@() fclose(fid));
data=[header;rows];
for r=1:size(data,1)
    fields=cell(1,size(data,2));
    for c=1:size(data,2)
        value=data{r,c};
        if isnumeric(value) || islogical(value), value=sprintf('%.17g',value); end
        fields{c}=regexprep(char(value),'[\t\r\n]',' ');
    end
    fprintf(fid,'%s\n',strjoin(fields,sprintf('\t')));
end
end
