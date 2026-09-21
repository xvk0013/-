% 严格配对的单目标对照；参照 RUN_DEEPSHIP_CUT.m 的目录、元数据和 FLOAT32 格式。
% 使用原场景清单，不重新划分、不重新切片/QC、不重新抽样、不重新归一化。
% no_bellhop: 原源缓存核心 * 原场景 common_scale。
% bellhop_reference: 原传播 WAV 解码后原幅值转存；不重新运行 Bellhop。
% 两边逐场景同名同标签，保留源重复次数。训练两边均使用原 dataset 的 noise。
% 已有传播 WAV 的量化/削波无法通过转存撤销，因此比较包含原传播写出链路的影响。
run_root='D:\LSJ\Data\data_gen_v6_no_noise\20260916_225548_748';
output_base='D:\LSJ\Data\deepship_5s_no_bellhop';

saved=load(fullfile(run_root,'RUN_CONFIG.mat'),'cfg'); cfg=saved.cfg;
[~,source_run_id]=fileparts(run_root);
export_output_dir=fullfile(output_base,[source_run_id '_paired']);
assert(~isfolder(export_output_dir),'输出目录已存在，请修改 output_base：%s',export_output_dir);
assert(cfg.segment_len_s==5 && cfg.target_fs==16000,'要求原运行是 16 kHz / 5 秒。');
assert(isequal(cfg.class_names,{'Cargo','Tanker','Tug'}),'要求原类别顺序 Cargo/Tanker/Tug。');
P=round(cfg.source_history_s*cfg.target_fs); N=round(cfg.segment_len_s*cfg.target_fs);
variants={'no_bellhop','bellhop_reference'};
named_ids=[0 1 3]; % 与 RUN_DEEPSHIP_CUT.m 及当前 named-class 模型读取器一致
header={'file_name','audio_path','class_id','class_name','split','recording_index', ...
    'raw_relative_path','segment_index','original_fs','fs','duration_s', ...
    'start_sample','stop_sample','QC_status','QC_reasons','input_rms_before', ...
    'normalization_gain','output_rms','source_index','source_class_id','recording_key', ...
    'candidate_id','original_scene_table','original_audio_path','input_context_path', ...
    'common_scale','variant','range_km','source_depth_m'};
mkdir(export_output_dir);
save(fullfile(export_output_dir,'SOURCE_RUN_CONFIG.mat'),'cfg','run_root','variants');
summary=cell(0,5);
record_splits=containers.Map('KeyType','char','ValueType','char');
total=0;
for si=1:numel(cfg.split_types)
    split_name=cfg.split_types{si};
    [sources,sc]=read_manifest(fullfile(run_root,'splits',split_name,'SOURCES.tsv'));
    sources=sources(ismember(sources(:,sc('QC_status')),{'PASS','REVIEW'}),:);
    if isKey(sc,'source_index')
        source_indices=str2double(sources(:,sc('source_index')));
    else
        source_indices=(1:size(sources,1)).';
    end
    assert(all(isfinite(source_indices)) && numel(unique(source_indices))==numel(source_indices), ...
        '源编号无效或重复：%s',split_name);
    for ci=1:numel(cfg.class_names)
        class_name=cfg.class_names{ci};
        scene_dir=fullfile(run_root,'dataset',cfg.combo_folders{ci+1},split_name);
        scene_table=fullfile(scene_dir,'all_info.txt');
        [scenes,tc]=read_manifest(scene_table);
        assert(~isempty(scenes),'没有单目标场景：%s',scene_table);
        out_rows=cell(1,2);
        for vi=1:2
            mkdir(fullfile(export_output_dir,variants{vi},class_name,split_name));
            out_rows{vi}=cell(size(scenes,1),numel(header));
        end
        used_sources=zeros(size(scenes,1),1);
        for j=1:size(scenes,1)
            assert(str2double(scenes{j,tc('n_targets')})==1 && ...
                strcmp(scenes{j,tc('split')},split_name) && ...
                strcmp(scenes{j,tc('class1')},class_name),'场景类别/划分不匹配。');
            source_index=str2double(scenes{j,tc('source1_index')});
            i=find(source_indices==source_index);
            assert(isscalar(i),'找不到唯一源编号：%g',source_index);
            assert(strcmp(sources{i,sc('class_name')},class_name) && ...
                strcmp(sources{i,sc('split')},split_name) && ...
                strcmp(sources{i,sc('recording_key')},scenes{j,tc('recording1')}), ...
                '场景与源缓存不匹配：%s / %d',scene_table,j);
            used_sources(j)=source_index;
            if isKey(sc,'input_path')
                input_path=resolve_path(run_root,sources{i,sc('input_path')});
            else
                input_path=fullfile(run_root,'splits',split_name,'source_contexts', ...
                    sources{i,sc('context_file')});
            end
            context=load(input_path,'x','fs');
            assert(context.fs==cfg.target_fs && numel(context.x)==P+N,'源缓存长度/采样率错误。');
            scale=str2double(scenes{j,tc('common_scale')});
            assert(isfinite(scale) && scale>0,'场景 common_scale 无效。');
            dry=single(double(context.x(P+(1:N)))*scale); dry=dry(:);
            if isKey(tc,'audio_path')
                original_audio_path=resolve_path(run_root,scenes{j,tc('audio_path')});
            else
                original_audio_path=fullfile(scene_dir,'mix',scenes{j,tc('file_name')});
                if ~isfile(original_audio_path)
                    original_audio_path=fullfile(scene_dir,scenes{j,tc('file_name')});
                end
            end
            [wet,fs]=audioread(original_audio_path);
            assert(fs==cfg.target_fs && isequal(size(wet),[N 1]),'传播 WAV 长度/通道/采样率错误。');
            wet=single(wet);
            assert(all(isfinite(dry)) && any(dry~=0) && all(isfinite(wet)) && any(wet~=0), ...
                '配对音频非有限或全零。');
            raw_relative=relative_recording(sources{i,sc('raw_path')},cfg.raw_root,class_name);
            record_key=lower(raw_relative);
            if isKey(record_splits,record_key)
                assert(strcmp(record_splits(record_key),split_name),'录音跨集合：%s',raw_relative);
            else
                record_splits(record_key)=split_name;
            end
            filename=sprintf('scene_%06d.wav',j);
            relative=sprintf('%s/%s/%s',class_name,split_name,filename);
            waves={dry,wet};
            for vi=1:2
                audio=waves{vi};
                write_float_wav(fullfile(export_output_dir,variants{vi},class_name,split_name,filename), ...
                    audio,cfg.target_fs);
                out_rows{vi}(j,:)={filename,relative,named_ids(ci),class_name,split_name, ...
                    scenes{j,tc('recording1_index')},raw_relative,sources{i,sc('segment_index')}, ...
                    sources{i,sc('original_fs')},cfg.target_fs,5,sources{i,sc('core_start_sample')}, ...
                    sources{i,sc('core_stop_sample')},sources{i,sc('QC_status')},sources{i,sc('QC_reasons')}, ...
                    sources{i,sc('input_rms_before')},sources{i,sc('input_gain')}, ...
                    sqrt(mean(double(audio).^2)),source_index,sources{i,sc('class_id')}, ...
                    sources{i,sc('recording_key')},scenes{j,tc('candidate_id')},scene_table, ...
                    original_audio_path,input_path,scale,variants{vi}, ...
                    scenes{j,tc('range1_km')},scenes{j,tc('source_depth1_m')}};
            end
        end
        for vi=1:2
            write_table(fullfile(export_output_dir,variants{vi},class_name,split_name,'all_info.txt'), ...
                header,out_rows{vi});
        end
        summary(end+1,:)={class_name,split_name,size(scenes,1),numel(unique(used_sources)), ...
            fullfile(run_root,'dataset')}; %#ok<SAGROW>
        total=total+size(scenes,1);
        fprintf('%s/%s: %d paired scenes, %d unique sources.\n', ...
            split_name,class_name,size(scenes,1),numel(unique(used_sources)));
    end
end
write_table(fullfile(export_output_dir,'SUMMARY.tsv'), ...
    {'class_name','split','paired_scenes','unique_sources','shared_noise_data_dir'},summary);
fprintf('完成：每组 %d 条单目标场景，FLOAT32，保留原幅值及重复次数。\n%s\n',total,export_output_dir);
fprintf('两组训练均使用原 noise-data-dir：%s\n',fullfile(run_root,'dataset'));

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
