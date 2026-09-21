% 整理已完成的旧四路输出：保留每个源分量一份，场景改用相对路径索引。
% 只移动已有 WAV、更新索引并删除冗余副本/缓存，不重算或重编码音频。
% 8 秒归一化输入 MAT 同样保留一份，移到 inputs 并记录路径。
% 在 MATLAB 确认下面的目录后点击“运行”；不能对正在生成的数据运行。
run_root='D:\LSJ\Data\data_gen_v6_no_noise\20260916_225548_748';

project_root=fileparts(mfilename('fullpath'));
addpath(fullfile(project_root,'src'),'-begin');
saved=load(fullfile(run_root,'RUN_RESULT.mat'),'complete');
assert(saved.complete,'请等待这批数据完整生成后再整理。');
saved=load(fullfile(run_root,'RUN_CONFIG.mat'),'cfg'); cfg=saved.cfg;

for si=1:numel(cfg.split_types)
    split_name=cfg.split_types{si};
    for slot=1:numel(cfg.class_ids)
        path=fullfile(run_root,'components',split_name,cfg.combo_folders{slot+1});
        if ~isfolder(path), mkdir(path); end
    end
    for k=1:numel(cfg.combo_slots)
        slots=cfg.combo_slots{k};
        folder=fullfile(run_root,'dataset',cfg.combo_folders{k},split_name);
        manifest=fullfile(folder,'all_info.txt');
        [header,rows]=read_manifest(manifest);
        for field={'audio_path','s1_path','s2_path','s3_path'}
            if ~ismember(field{1},header)
                header{end+1}=field{1}; rows(:,end+1)={''};
            end
        end
        columns=containers.Map(header,num2cell(1:numel(header)));
        for j=1:size(rows,1)
            % 旧版按组合内保存序号命名；重新运行整理脚本时仍可定位残留副本。
            name=sprintf('combined_%05d.wav',str2double(rows{j,columns('combIdx')}));
            component_paths={'','',''};
            for t=1:numel(slots)
                slot=slots(t);
                source=str2double(rows{j,columns(sprintf('source%d_index',t))});
                range=str2double(rows{j,columns(sprintf('range%d_km',t))});
                depth=str2double(rows{j,columns(sprintf('source_depth%d_m',t))});
                [has_range,r]=ismember(range,cfg.ranges_km);
                [has_depth,d]=ismember(depth,cfg.source_depths_m);
                assert(has_range && has_depth,'场景几何不在本次配置中：%s。',manifest);
                relative=sprintf('components/%s/%s/src_%05d_r%02d_d%02d.wav', ...
                    split_name,cfg.combo_folders{slot+1},source,r,d);
                destination=fullfile(run_root,relative);
                if ~isfile(destination)
                    movefile(fullfile(folder,sprintf('s%d',slot),name),destination);
                end
                component_paths{slot}=relative;
            end
            if numel(slots)==1
                audio_path=component_paths{slots};
            else
                audio_path=sprintf('dataset/%s/%s/%s',cfg.combo_folders{k},split_name,name);
                destination=fullfile(run_root,audio_path);
                if ~isfile(destination), movefile(fullfile(folder,'mix',name),destination); end
            end
            [~,stem,extension]=fileparts(audio_path);
            rows{j,columns('file_name')}=[stem extension];
            rows{j,columns('audio_path')}=audio_path;
            for slot=1:3, rows{j,columns(sprintf('s%d_path',slot))}=component_paths{slot}; end
            % 所有有效音频已有唯一归档；只删该场景的旧副本和全零占位。
            for legacy={'mix','s1','s2','s3'}
                path=fullfile(folder,legacy{1},name);
                if isfile(path), delete(path); end
            end
        end
        write_dataset_table(manifest,header,rows);
        for legacy={'mix','s1','s2','s3'}
            path=fullfile(folder,legacy{1});
            if isfolder(path), rmdir(path); end % 仅移除空目录，不递归删除未知内容。
        end
        fprintf('Indexed %s/%s: %d scenes.\n',split_name,cfg.combo_names{k},size(rows,1));
    end

    % 保留 8 秒输入，将旧上下文移到 inputs，来源表记录编号和相对路径。
    source_manifest=fullfile(run_root,'splits',split_name,'SOURCES.tsv');
    [header,rows]=read_manifest(source_manifest);
    context_column=find(strcmp(header,'context_file'),1);
    if ~isempty(context_column)
        input_root=fullfile(run_root,'inputs',split_name);
        if ~isfolder(input_root), mkdir(input_root); end
        status_column=find(strcmp(header,'QC_status'),1);
        header{end+1}='source_index'; rows(:,end+1)={''};
        index_column=numel(header);
        index=0;
        for j=1:size(rows,1)
            source_index=0; input_path='';
            if ismember(rows{j,status_column},{'PASS','REVIEW'})
                index=index+1; source_index=index;
                input_path=sprintf('inputs/%s/input_%05d.mat',split_name,source_index);
                destination=fullfile(run_root,input_path);
                if ~isfile(destination)
                    old_path=fullfile(run_root,'splits',split_name,'source_contexts',rows{j,context_column});
                    movefile(old_path,destination);
                end
            end
            rows{j,context_column}=input_path;
            rows{j,index_column}=num2str(source_index);
        end
        header{context_column}='input_path';
        write_dataset_table(source_manifest,header,rows);
    end

    % 只清理重复的传播 MAT；输入 MAT 已移动保留，只移除其旧空目录。
    cache_root=fullfile(run_root,'splits',split_name,'propagated_cache');
    remove_mat_files(cache_root,'ref_*.mat');
    context_root=fullfile(run_root,'splits',split_name,'source_contexts');
    for id=cfg.class_ids
        path=fullfile(context_root,num2str(id));
        if isfolder(path), rmdir(path); end
    end
    if isfolder(context_root), rmdir(context_root); end
end
cfg.output_layout='indexed_components_v12';
if isfield(cfg,'output_folders'), cfg=rmfield(cfg,'output_folders'); end
save(fullfile(run_root,'RUN_CONFIG.mat'),'cfg');
fprintf('整理完成：训练按 all_info.txt 的 audio_path 读取，相对根目录 %s\n',run_root);

function [header,rows]=read_manifest(path)
% 保留原始数值字符串与文件名，不让类型推断改写场景记录。
lines=readlines(path,'Encoding','UTF-8'); lines=lines(strlength(lines)>0);
header=reshape(cellstr(split(lines(1),sprintf('\t'))),1,[]);
rows=cell(numel(lines)-1,numel(header));
for j=2:numel(lines)
    rows(j-1,:)=reshape(cellstr(split(lines(j),sprintf('\t'))),1,[]);
end
end

function remove_mat_files(folder,pattern)
if ~isfolder(folder), return; end
files=dir(fullfile(folder,pattern));
for j=1:numel(files), delete(fullfile(folder,files(j).name)); end
rmdir(folder); % 非空则停止，绝不递归清理额外文件。
end
