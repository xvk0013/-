function [channels,geometry,acoustics]=prepare_channels(cfg)
% 公共接收机互易 ARR -> 39 个宽带信道。内部辅助函数集中在本文件。
template=fullfile(cfg.project_root,'acoustics','template_pos1');
templates=cell(0,2);
for extension={'.env','.ssp','.bty','.brc','.ati','.trc','.sbp'}
    path=[template extension{1}];
    if isfile(path), templates(end+1,:)={extension{1},fileread(path)}; end %#ok<AGROW>
end
settings=struct('geometry_method',cfg.geometry_method,'receiver_x_km',cfg.receiver_x_km, ...
    'receiver_depth_m',cfg.receiver_depth_m,'source_depths_m',cfg.source_depths_m, ...
    'ranges_km',cfg.ranges_km,'frequencies_hz',cfg.frequencies_hz, ...
    'templates',{templates});
relative=fullfile('acoustics','arr'); folder=fullfile(cfg.project_root,relative);
generate_arr(cfg,template,folder,settings);
acoustics=struct('settings',settings,'arr_relative_path',relative);
fc=cfg.frequencies_hz; arrays=cell(size(fc)); Pos=[];
for k=1:numel(fc)
    path=fullfile(folder,sprintf('ReceiverAzi1freq%dHz.arr',fc(k)));
    [arrays{k},position]=read_arrivals_asc(path);
    if isempty(Pos), Pos=position;
    else, assert(isequaln(Pos,position),'ARR frequency files have different geometry.'); end
end
assert(isequal(double(Pos.r.r(:).')/1000,cfg.ranges_km) && ...
    isequal(double(Pos.r.z(:).'),cfg.source_depths_m) && ...
    isequal(double(Pos.s.z(:).'),cfg.receiver_depth_m),'ARR geometry differs from the common receiver model.');
N=round(cfg.segment_len_s*cfg.target_fs); P=round(cfg.source_history_s*cfg.target_fs);
nfft=2^nextpow2(N+P+round(cfg.max_delay_s*cfg.target_fs));
channels=cell(numel(cfg.ranges_km),numel(cfg.source_depths_m));
for d=1:size(channels,2)
    for r=1:size(channels,1)
        amps=cell(size(fc)); taus=amps;
        for k=1:numel(fc)
            amps{k}=double(arrays{k}(r,d,1).A); taus{k}=double(arrays{k}(r,d,1).delay);
        end
        H=build_channel_H(amps,taus,fc,cfg.target_fs,nfft,cfg.max_delay_s);
        channels{r,d}=causal_channel(H,cfg.target_fs,nfft,cfg.max_delay_s);
    end
end
geometry=struct('ranges_km',cfg.ranges_km,'source_depths_m',cfg.source_depths_m, ...
    'source_x_km',cfg.receiver_x_km+cfg.ranges_km,'receiver_x_km',cfg.receiver_x_km, ...
    'receiver_depth_m',cfg.receiver_depth_m,'method',cfg.geometry_method,'nfft',nfft,'f_centers',fc);
fprintf('Channels ready: %d ranges x %d source depths, receiver %.0f m.\n', ...
    size(channels,1),size(channels,2),cfg.receiver_depth_m);
end

function generate_arr(cfg,template,folder,settings)
settings_path=fullfile(folder,'ARR_SETTINGS.mat');
files=arrayfun(@(f) fullfile(folder,sprintf('ReceiverAzi1freq%dHz.arr',f)), ...
    cfg.frequencies_hz,'UniformOutput',false);
if isfile(settings_path)
    previous_settings=load(settings_path,'settings');
    if isequaln(previous_settings.settings,settings) && all(cellfun(@isfile,files))
        fprintf('Reusing %d ARR files: %s\n',numel(files),folder); return;
    end
    % 重建期间不能把新旧混合的一组 ARR 标为完成。
    delete(settings_path);
end
lines=regexp(fileread([template '.env']),'\r?\n','split');
source_line=find(contains(lines,'! NSz'),1);
lines{source_line}='1 ! NSz: computational source at physical receiver';
lines{source_line+1}=sprintf('%.10g / ! Sz (m)',cfg.receiver_depth_m);
depth_line=find(contains(lines,'! NRz'),1);
lines{depth_line}=sprintf('%d ! NRz: physical source depths',numel(cfg.source_depths_m));
lines{depth_line+1}=[sprintf('%.10g ',cfg.source_depths_m) '/ ! Rz (m)'];
range_line=find(contains(lines,'! NRr'),1);
lines{range_line}=sprintf('%d ! NRr',numel(cfg.ranges_km));
lines{range_line+1}=[sprintf('%.10g ',cfg.ranges_km) '/ ! Rr (km)'];
fid=fopen([template '.bty'],'rt'); fgetl(fid); fgetl(fid);
bottom=fscanf(fid,'%f',[2 Inf]).'; fclose(fid);
box_line=find(contains(lines,'Box.r'),1); box=sscanf(lines{box_line},'%f',3);
lines{box_line}=sprintf('%.10g %.10g %.10g ! deltas Box.z Box.r',box(1),box(2),bottom(end,1));
run_line=find(contains(lines,'! Run Type'),1);
lines{run_line}='''AB R'' ! Run Type: arrivals, Gaussian beams, omnidirectional point source';
if ~exist(folder,'dir'), mkdir(folder); end
work=fullfile(cfg.project_root,'acoustics','bellhop_work');
if ~exist(work,'dir'), mkdir(work); end
prefix='channel';
for extension={'.ssp','.bty','.brc','.ati','.trc','.sbp'}
    input=[template extension{1}]; output=fullfile(work,[prefix extension{1}]);
    if isfile(input), copyfile(input,output);
    elseif isfile(output), delete(output); end
end
previous=pwd; restore_folder=onCleanup(@() cd(previous)); cd(work);
for k=1:numel(cfg.frequencies_hz)
    lines{2}=sprintf('%.2f ! Frequency (Hz)',cfg.frequencies_hz(k));
    fid=fopen([prefix '.env'],'wt'); fprintf(fid,'%s\n',lines{:}); fclose(fid);
    if isfile([prefix '.arr']), delete([prefix '.arr']); end
    fprintf('Bellhop: %d/%d frequencies.\n',k,numel(cfg.frequencies_hz));
    [status,message]=system(sprintf('"%s" %s',cfg.bellhop_executable,prefix));
    assert(status==0 && isfile([prefix '.arr']), ...
        'Bellhop failed at %g Hz: %s',cfg.frequencies_hz(k),message);
    movefile([prefix '.arr'],files{k},'f');
end
save(settings_path,'settings');
cd(previous);
delete(fullfile(work,[prefix '.*']));
rmdir(work);
end

function channel=causal_channel(H,fs,nfft,max_delay_s)
% 保留原 0～3 s 因果投影，不补偿被截去的能量。
P=round(max_delay_s*fs); half=floor(nfft/2)+1;
H=H(:); H(1)=real(H(1));
if mod(nfft,2)==0
    H(end)=real(H(end)); full=[H;conj(H(end-1:-1:2))];
else
    full=[H;conj(H(end:-1:2))];
end
raw=real(ifft(full)); h=raw(1:P+1);
assert(all(isfinite(h)) && sum(h.^2)>0,'Invalid or empty causal channel.');
kept=fft(h,nfft);
channel=struct('H',kept(1:half),'nfft',nfft,'memory_samples',P,'fs',fs);
end

function H=build_channel_H(cell_Amp,cell_tau,f_centers,fs,N_pad,max_delay)
% ARR phase is exp(+1i*phase); Fourier propagation is exp(-1i*2*pi*f*tau).
% t0 is REAL and common to all frequency anchors at this geometry. Only the
% real delays determine ray retention; imaginary seconds remain in tau.
% exp(-1i*2*pi*f*(tau-t0)) =
% exp(-1i*2*pi*f*(real(tau)-t0)) * exp(2*pi*f*imag(tau)).
% Do not fold absorption into A at fc and then apply it a second time here.
for k=1:numel(cell_tau)
    assert(numel(cell_Amp{k})==numel(cell_tau{k}) && ...
        all(isfinite(cell_Amp{k}(:))) && all(isfinite(cell_tau{k}(:))), ...
        'Invalid ARR amplitude/complex delay.');
end
fk=(0:N_pad-1)'/N_pad*fs;
half_N=floor(N_pad/2)+1;
fk_half=fk(1:half_N);
global_min_tau=inf;
for i=1:numel(cell_tau)
    if ~isempty(cell_tau{i}), global_min_tau=min(global_min_tau,min(real(cell_tau{i}))); end
end
if ~isfinite(global_min_tau), error('Bellhop 信道没有有效到达。'); end
H=zeros(half_N,1);
for i=1:numel(f_centers)-1
    fL=f_centers(i); fR=f_centers(i+1);
    if numel(f_centers)==2
        idx=(1:half_N)'; % the same interval owns both extrapolation ends
    elseif i==1
        idx=find(fk_half>=0 & fk_half<=fR);
    elseif i==numel(f_centers)-1
        idx=find(fk_half>fL & fk_half<=fs/2);
    else
        idx=find(fk_half>fL & fk_half<=fR);
    end
    if isempty(idx), continue; end
    f_eval=fk_half(idx);
    tauL_all=real(cell_tau{i})-global_min_tau;
    tauR_all=real(cell_tau{i+1})-global_min_tau;
    maskL=tauL_all<=max_delay; maskR=tauR_all<=max_delay;
    % Subtract only the common real origin: subtracting a complex origin
    % would incorrectly remove the earliest path's absorption.
    tauL=cell_tau{i}(maskL)-global_min_tau;
    tauR=cell_tau{i+1}(maskR)-global_min_tau;
    ampL=cell_Amp{i}(maskL); ampR=cell_Amp{i+1}(maskR);
    if isempty(ampL)
        HL=zeros(1,numel(idx));
    else
        % 使用绝对频率 f 和完整复延迟，同时计入相位与虚延迟衰减。
        HL=(ampL(:).')*exp(-1j*2*pi*tauL(:)*f_eval(:).');
    end
    if isempty(ampR)
        HR=zeros(1,numel(idx));
    else
        HR=(ampR(:).')*exp(-1j*2*pi*tauR(:)*f_eval(:).');
    end
    wR=(f_eval(:).'-fL)/(fR-fL);
    if i==1, wR(f_eval(:).'<fL)=0; end
    if i==numel(f_centers)-1, wR(f_eval(:).'>fR)=1; end
    H(idx)=(1-wR).*HL+wR.*HR;
end
assert(all(isfinite(H)),'Nonfinite complex-delay channel response.');
end

function [Arr, Pos] = read_arrivals_asc(fname)
% Parse Bellhop ASCII arrivals, retaining complex delay and amplitude.
    txt = fileread(fname);
    lines = regexp(txt, '\r?\n', 'split');
    keep = ~cellfun(@(l) isempty(strtrim(l)), lines);
    lines = lines(keep);
    lines = cellfun(@strtrim, lines, 'UniformOutput', false);

    % ===== 头部解析 =====
    tok_s  = strsplit(lines{3});          % [NSz, z_src...]
    tok_r  = strsplit(lines{4});          % [NRz, z_recv...]
    tok_rr = strsplit(lines{5});          % [NRr, r...]
    NSz = str2double(tok_s{1});
    NRz = str2double(tok_r{1});
    NRr = str2double(tok_rr{1});
    Pos.s.z = str2double(tok_s(2:end));      % 源深 (m)
    Pos.r.z = str2double(tok_r(2:end));      % 接收深 (m)
    Pos.r.r = str2double(tok_rr(2:end));     % 距离 (m)

    % ===== 主体: section (源深) → 接收深 × 距离 块 =====
    Arr = repmat(struct('A', [], 'delay', []), NRr, NRz, NSz);
    pos = 6;
    for n_d = 1:NSz
        pos = pos + 1;                    % section 额外整数 (最大到达数), 跳过
        for nz = 1:NRz
            for k_r = 1:NRr
                n = str2double(lines{pos});
                pos = pos + 1;
                if n > 0
                    rows = zeros(n, 8);
                    for k = 1:n
                        rows(k, :) = str2double(strsplit(lines{pos}));
                        pos = pos + 1;
                    end
                    Arr(k_r, nz, n_d).A = rows(:, 1) .* exp(1j * rows(:, 2) * pi / 180);
                    Arr(k_r, nz, n_d).delay = complex(rows(:, 3), rows(:, 4));
                end
            end
        end
    end
    assert(pos == numel(lines) + 1, ...
        '%s: 解析未耗尽 (已读 %d/%d 行) — .arr 格式与预期不符', fname, pos - 1, numel(lines));
end
