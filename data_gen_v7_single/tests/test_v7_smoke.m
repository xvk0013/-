function test_v7_smoke
% 仅合成小样本的实现检查：不读DeepShip，不训练，不运行参数搜索。
root=fileparts(fileparts(mfilename('fullpath')));
entry=fileread(fullfile(root,'RUN_DATA_GEN_V7_SINGLE.m'));
entry=strrep(entry,'project_root=fileparts(mfilename(''fullpath''));','project_root=root;');
position=strfind(entry,'v7_output_dir=generate_single_dataset(cfg);');
assert(isscalar(position)); eval(entry(1:position-1));
project_root=root; cfg.project_root=root;
cfg.bellhop_executable=fullfile(root,'tools','bellhop','bellhop.exe');
addpath(fullfile(root,'src'),'-begin');
work=fullfile(root,'tests','smoke_work');
assert(~isfolder(work),'合成测试目录已存在。'); mkdir(work);
cfg.raw_root=fullfile(work,'raw'); cfg.source_split_root=fullfile(work,'split');
cfg.run_root=fullfile(work,'run'); cfg.dataset_root=fullfile(cfg.run_root,'dataset');
cfg.summary_copy=fullfile(work,'output','summary.json');
cfg.exclusion_file=fullfile(work,'excluded_recordings.txt');
cfg.progress_every_scenes=100;
excluded={'Cargo/excluded/a.wav','Tanker/excluded/b.wav'};
write_dataset_table(cfg.exclusion_file,excluded(1),excluded(2));
lengths=[11 17 23 29];
for ci=1:3
    for j=1:4
        folder=fullfile(cfg.raw_root,cfg.class_names{ci},sprintf('record_%02d',j)); mkdir(folder);
        fs=16000; if j==2, fs=8000; end
        t=(0:fs*lengths(j)-1).'/fs;
        wave=.2*sin(2*pi*(50+ci*30+j*7)*t)+.05*sin(2*pi*(217+j*19)*t);
        v7_write_float_wav(fullfile(folder,'record.wav'),single(wave),fs);
    end
end
for j=1:2
    path=fullfile(cfg.raw_root,excluded{j}); mkdir(fileparts(path));
    t=(0:6*16000-1).'/16000;
    v7_write_float_wav(path,single(.2*sin(2*pi*90*t)),16000);
end
% 长度悬殊时仍整录音分配，且固定种子可复现。
weights=[2 3 4 5 9 15 18 20 31 45 60 80 100 130 170 200];
a=balance_recordings(weights,[.7 .15 .15],47,.02,100);
b=balance_recordings(weights,[.7 .15 .15],47,.02,100);
assert(isequal(a,b) && all(ismember(1:3,a)));
loads=accumarray(a,weights.',[3 1]).'/sum(weights);
assert(max(abs(loads-[.7 .15 .15]))<.03,'切片平衡误差过大。');
generate_single_dataset(cfg);
report=jsondecode(fileread(cfg.summary_copy));
assert(strcmp(report.status,'complete') && report.ship_audio_bellhop);
assert(report.source.retained_recordings==12 && report.source.single_slices==42);
assert(report.source.excluded_recording_count==2 && report.source.excluded_complete_slices==2);
assert(report.ship_samples==42 && report.multi_target_scenes==0);
assert(report.source.padded_start_slices==12);
assert(report.source.multi_target_eligible_slices<report.source.single_slices);
partition=readtable(fullfile(cfg.run_root,'RECORDING_PARTITION.tsv'),'FileType','text', ...
    'Delimiter','\t','TextType','string');
assert(numel(unique(partition.raw_relative_path))==height(partition));
for si=1:3
    split=cfg.split_types{si};
    for ci=1:3
        folder=fullfile(cfg.dataset_root,cfg.class_names{ci},split);
        table=readtable(fullfile(folder,'all_info.txt'),'FileType','text','Delimiter','\t','TextType','string');
        assert(height(table)==numel(unique(table.source1_index)));
        row=table(1,:);
        [wave,fs]=audioread(fullfile(cfg.dataset_root,row.audio_path));
        assert(fs==16000 && isequal(size(wave),[80000 1]) && any(wave~=0));
        context=load(fullfile(cfg.run_root,row.input_path),'x');
        assert(abs(sqrt(mean(context.x(48001:end).^2))-1)<1e-10);
        assert(norm(wave-double(single(context.x(48001:end))))>1e-4, ...
            '单目标输出错误地仍然是未传播核心。');
        fid=fopen(fullfile(cfg.dataset_root,row.audio_path),'rb'); fseek(fid,20,'bof');
        format=fread(fid,1,'uint16'); fclose(fid); assert(format==3,'输出不是IEEE浮点WAV。');
        assert(all(table.mixing_stage=="receiver_after_independent_propagation"));
    end
    n=readtable(fullfile(cfg.dataset_root,'noise',split,'all_info.txt'), ...
        'FileType','text','Delimiter','\t','TextType','string');
    assert(all(n.label_Cargo==0 & n.label_Tanker==0 & n.label_Tug==0));
    assert(all(n.noise_receiver_target_rms>=report.noise_reference.minimum_rms*(1-1e-6)) && ...
        all(n.noise_receiver_target_rms<=report.noise_reference.maximum_rms*(1+1e-6)));
    assert(isfile(fullfile(cfg.dataset_root,'noise',split,'mix',n.file_name(1))));
end
assert(~isfolder(fullfile(cfg.dataset_root,'0_1')));
reference=load(fullfile(cfg.run_root,'NOISE_LEVEL_REFERENCE.mat'),'reference');
assert(strcmp(reference.reference.source_split,'Train'));
% 冻结保护在调用传播/配对前生效。
blocked=false;
try
    current=cfg; current.scene_counts=zeros(1,7);
    generate_scenes(current,struct(),5);
catch err
    blocked=strcmp(err.identifier,'V7:Frozen');
end
assert(blocked,'多目标冻结保护没有生效。');
fprintf('V7_SMOKE_PASS: 42 propagated ship samples; independent noise; record split; source exclusion; real-history eligibility; float WAV; multi-target frozen.\n');
end
