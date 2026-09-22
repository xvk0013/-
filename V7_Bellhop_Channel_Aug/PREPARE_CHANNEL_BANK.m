% Export the ORIGINAL V7 Bellhop filters. No audio generation/training here.
here=fileparts(mfilename('fullpath'));
out=fullfile(here,'output');
if ~isfolder(out), mkdir(out); end
destination=fullfile(out,'channel_bank.mat');
if isfile(destination)
    fprintf('Channel bank already exists: %s\n',destination);
    return;
end
generation_root='D:/LSJ/Data/data_gen_v7_single/20260920_163131_193';
saved=load(fullfile(generation_root,'RUN_CONFIG.mat'),'cfg'); cfg=saved.cfg;
assert(strcmp(cfg.revision,'data_gen_v7_single_bellhop_expanded'));
assert(cfg.target_fs==16000 && cfg.segment_len_s==5 && cfg.source_history_s==3);
assert(cfg.output_gain==1 && strcmp(cfg.output_encoding,'IEEE_float32'));
reference=fullfile(here,'generator_reference');
a=load(fullfile(generation_root,'ACOUSTICS.mat'),'geometry','acoustics');
cached=load(fullfile(reference,'acoustics','arr','ARR_SETTINGS.mat'),'settings');
assert(isequaln(a.acoustics.settings,cached.settings), ...
    'The ARR snapshot must match the completed original V7 generation.');
for i=1:size(cached.settings.templates,1)
    extension=cached.settings.templates{i,1};
    assert(strcmp(fileread(fullfile(reference,'acoustics',['template_pos1' extension])), ...
        cached.settings.templates{i,2}),'Acoustic template changed.');
end
for f=cfg.frequencies_hz
    assert(isfile(fullfile(reference,'acoustics','arr',sprintf('ReceiverAzi1freq%dHz.arr',f))), ...
        'Missing original ARR; do not silently rebuild a different environment.');
end
cfg.project_root=reference;
addpath(fullfile(reference,'src'),'-begin');
[channels,geometry,~]=prepare_channels(cfg);
assert(isequaln(geometry,a.geometry),'Original channel geometry does not match.');
nr=numel(cfg.ranges_km); nd=numel(cfg.source_depths_m);
H=zeros(floor(geometry.nfft/2)+1,nr*nd);
ranges_km=zeros(1,nr*nd); depths_m=ranges_km;
range_indices=ranges_km; depth_indices=ranges_km;
for d=1:nd
    for r=1:nr
        k=(d-1)*nr+r;
        H(:,k)=channels{r,d}.H;
        ranges_km(k)=cfg.ranges_km(r); depths_m(k)=cfg.source_depths_m(d);
        range_indices(k)=r; depth_indices(k)=d;
    end
end
nfft=geometry.nfft; fs=cfg.target_fs;
history_samples=round(cfg.source_history_s*fs);
core_samples=round(cfg.segment_len_s*fs);
memory_samples=round(cfg.max_delay_s*fs);
output_gain=cfg.output_gain; schema_version=1;
temporary=fullfile(out,'channel_bank.tmp.mat');
save(temporary,'H','nfft','fs','history_samples','core_samples','memory_samples', ...
    'output_gain','ranges_km','depths_m','range_indices','depth_indices', ...
    'generation_root','schema_version','-v7');
movefile(temporary,destination,'f');
fprintf('READY: %d original Bellhop filters exported. No Train/Val/Test audio generated.\n',nr*nd);
fprintf('Next, run bash run_channel_aug.sh in WSL.\n');
