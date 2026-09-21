function test_v7_frozen_split
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'src'),'-begin');
work=fullfile(root,'tests','smoke_work');
saved=load(fullfile(work,'run','RUN_CONFIG.mat'),'cfg'); cfg=saved.cfg;
original=readtable(fullfile(cfg.run_root,'RECORDING_PARTITION.tsv'), ...
    'FileType','text','Delimiter','\t','TextType','string');
cfg.run_root=fullfile(work,'repeat_source_preparation');
assert(~isfolder(cfg.run_root)); mkdir(cfg.run_root);
[~,~,report]=prepare_sources_expanded(cfg);
repeated=readtable(fullfile(cfg.run_root,'RECORDING_PARTITION.tsv'), ...
    'FileType','text','Delimiter','\t','TextType','string');
assert(isequal(original,repeated) && report.single_slices==42);
fprintf('V7_FROZEN_SPLIT_PASS: repeat source preparation reused the exact recording assignment; no scene generation or model training.\n');
end
