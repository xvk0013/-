function [result,cache]=generate_scenes(cfg,cache,k)
% Generate one split/combination; preserve the accepted pairing and reuse rules.
slots=cfg.combo_slots{k}; nt=numel(slots); requested=cfg.scene_counts(k);
assert(nt<=1 || (isfield(cfg,'multi_target_enabled') && cfg.multi_target_enabled), ...
    'V7:Frozen','多目标组件已冻结；本轮不能生成双目标场景。');
name=cfg.combo_names{k}; folder=fullfile(cfg.dataset_root,cfg.combo_folders{k},cfg.active_split);
assert(~exist(folder,'dir'),'V6:OutputExists','Combination output already exists.'); mkdir(folder);
uses=zeros(1,numel(cache.recordings));
clip_uses=zeros(1,numel(cache.sources)); caps=cache.recording_caps(:,k).';
used_scenes=containers.Map('KeyType','char','ValueType','logical');
max_attempts=cfg.max_attempts_per_combination(k);
attempted=0; accepted=0; selected=struct([]); numeric_rejected=0; pairing_rejected=0;
primary_count=0; supplement_count=0; outside_count=0; card_mismatch_count=0;
deck=[];
[~,single_ranges]=ismember(cfg.single_ranges_km,cfg.ranges_km);
pair_state=struct();
if nt==2
    clip_caps=zeros(size(clip_uses));
    for slot=slots
        group=[cache.recordings(cache.groups{slot}).source_indices];
        clip_caps(group)=max(cfg.pairing.minimum_clip_cap, ...
            ceil(cfg.pairing.clip_capacity_headroom*cfg.source_reuse.capacity_headroom*requested/numel(group)));
    end
    pair_state=struct('recording_uses',uses,'recording_caps',caps, ...
        'clip_uses',clip_uses,'clip_caps',clip_caps, ...
        'sampling_uses',clip_uses, ...
        'pair_uses',containers.Map('KeyType','char','ValueType','double'), ...
        'range_uses',zeros(1,numel(cfg.ranges_km)));
    deck=target_deck(cfg,requested,name);
end
template=struct('source1_index',0,'source2_index',0,'recording1_index',0,'recording2_index',0, ...
    'recording1','none','recording2','none', ...
    'range1_km',NaN,'range2_km',NaN,'source_depth1_m',NaN,'source_depth2_m',NaN, ...
    'source1_x_km',NaN,'source2_x_km',NaN, ...
    'keeper_slot',NaN,'signed_offset_m',NaN,'separation_m',NaN,'direction','not_applicable', ...
    'card_target_db',NaN,'card_band','none', ...
    'predicted_signed_db',NaN,'stronger_scheduled',NaN, ...
    'candidate_switches',NaN,'keeper_redraws',NaN, ...
    'ladder_level',NaN,'shortlist_size',NaN, ...
    'pair_search_reason','not_applicable', ...
    'noise_model','none','noise_wind_speed_m_s',NaN,'noise_shipping_factor',NaN, ...
    'noise_input_gain',NaN, ...
    'noise_input_rms_before',NaN,'noise_input_rms_after',NaN, ...
    'noise_receiver_target_rms',NaN,'noise_receiver_gain',NaN);
old=rng; guard=onCleanup(@() rng(old));
try
    while accepted<requested && attempted<max_attempts
        exhausted=false;
        for t=1:nt
            group=cache.groups{slots(t)};
            exhausted=exhausted || ~any(uses(group)<caps(group));
        end
        if exhausted, break; end
        attempted=attempted+1; id=sprintf('%s_%s_c%06d',cfg.active_split,name,attempted);
        c=template;
        rng(v6_seed(id,cfg.physical_seed),'twister');
        indices=zeros(1,nt); recording_indices=zeros(1,nt); ranges=zeros(1,nt);
        depths=randi(numel(cfg.source_depths_m),1,nt);
        targets=zeros(cfg.target_fs*cfg.segment_len_s,nt);
        component_paths={'','',''};
        noise_clip=[];
        sampling_uses=cache.total_clip_uses;
        rng(v6_seed([id '|sources'],cfg.physical_seed),'twister');
        if nt==0
            rng(v6_seed([id '|wenz'],cfg.physical_seed),'twister');
            wind=cfg.noise.wind_speed_range_m_s; shipping=cfg.noise.shipping_factor_range;
            c.noise_model=cfg.noise.model;
            c.noise_wind_speed_m_s=wind(1)+diff(wind)*rand;
            c.noise_shipping_factor=shipping(1)+diff(shipping)*rand;
            noise_clip=generate_wenz_noise(cfg,c.noise_wind_speed_m_s,c.noise_shipping_factor);
            [noise_clip,normalization]=normalize_noise_input(noise_clip,cfg);
            c.noise_input_gain=normalization.input_gain;
            c.noise_input_rms_before=normalization.input_rms_before;
            c.noise_input_rms_after=normalization.input_rms_after;
            levels=cache.noise_level_reference.log_rms_sorted;
            position=1+(numel(levels)-1)*rand;
            c.noise_receiver_target_rms=exp(interp1(1:numel(levels),levels,position,'linear'));
            c.noise_receiver_gain=c.noise_receiver_target_rms / ...
                (normalization.input_rms_after*cfg.output_gain);
            noise_clip=noise_clip*c.noise_receiver_gain;
        end
        if nt==1
            group=cache.groups{slots}; group=group(uses(group)<caps(group));
            candidates=[cache.recordings(group).source_indices];
            if isfield(cfg,'single_use_each_source_once') && cfg.single_use_each_source_once
                candidates=candidates(clip_uses(candidates)==0);
                if isempty(candidates), break; end
            end
            indices=draw_clip(candidates,sampling_uses);
            recording_indices=cache.sources(indices).recording_index;
            ranges=single_ranges(randi(numel(single_ranges)));
        elseif nt==2
            card_target=deck.target_db(accepted+1);
            card_band=deck.band{accepted+1};
            rng(v6_seed([id '|pairing'],cfg.physical_seed),'twister');
            stronger=randi(2); keeper=randi(2); matcher=3-keeper;
            target_signed=card_target*(3-2*stronger);
            c.card_target_db=card_target; c.card_band=card_band;
            c.stronger_scheduled=stronger; c.keeper_slot=keeper;
            pair_state.recording_uses=uses; pair_state.clip_uses=clip_uses;
            pair_state.sampling_uses=sampling_uses;
            [scheduled,cache]=schedule_pair(cfg,cache,slots,pair_state, ...
                depths,keeper,target_signed,card_band,used_scenes);
            if ~scheduled.found
                pairing_rejected=pairing_rejected+1; continue;
            end
            indices=scheduled.indices; ranges=scheduled.ranges;
            card_mismatch=scheduled.card_mismatch;
            recording_indices=[cache.sources(indices).recording_index];
            c.signed_offset_m=round((cfg.ranges_km(ranges(matcher))- ...
                cfg.ranges_km(ranges(keeper)))*1000);
            c.separation_m=abs(c.signed_offset_m);
            c.direction='same_range';
            if c.signed_offset_m<0, c.direction='matcher_closer'; end
            if c.signed_offset_m>0, c.direction='matcher_farther'; end
            c.predicted_signed_db=scheduled.predicted_signed_db;
            c.candidate_switches=scheduled.candidate_switches;
            c.keeper_redraws=scheduled.keeper_redraws;
            c.ladder_level=scheduled.ladder_level;
            c.shortlist_size=scheduled.shortlist_size;
            c.pair_search_reason=scheduled.reason;
        end
        assert(nt~=2 || ranges(1)~=ranges(2),'Two sources cannot share a horizontal position.');
        if nt>0
            % 类别槽位次序固定：每个分量的片段/距离/源深索引标记完整场景。
            % 重复则重新提议片段和位置，不传播、不写盘、不提交使用次数。
            scene_key=sprintf('%d_%d_%d;', [indices;ranges;depths]);
            if isKey(used_scenes,scene_key), continue; end
        end
        for t=1:nt
            row=cache.sources(indices(t)).row;
            c.(sprintf('source%d_index',t))=indices(t);
            c.(sprintf('recording%d_index',t))=recording_indices(t);
            c.(sprintf('recording%d',t))=row{4};
            c.(sprintf('range%d_km',t))=cfg.ranges_km(ranges(t));
            c.(sprintf('source%d_x_km',t))=cache.geometry.source_x_km(ranges(t));
            c.(sprintf('source_depth%d_m',t))=cfg.source_depths_m(depths(t));
            component_paths{slots(t)}=sprintf('dataset/%s/%s/src_%05d_r%02d_d%02d.wav', ...
                cfg.combo_folders{slots(t)+1},cfg.active_split,indices(t),ranges(t),depths(t));
           
            [wave,cache]=propagate_sources(cfg,cache,indices(t),ranges(t),depths(t));
            targets(:,t)=wave;
        end
        if any(~isfinite(targets(:))) || any(~any(targets~=0,1))
            numeric_rejected=numeric_rejected+1; continue;
        end
       
        filename=sprintf('combined_%05d.wav',accepted+1);
        expected_band='';
        if nt==2
            expected_band=card_band;
            if card_mismatch, expected_band='any'; end
        end
        p=write_scene(folder,filename,targets,slots,component_paths,cfg,noise_clip,expected_band);
        entry=struct('candidate_id',id,'file_name',p.file_name,'audio_path',p.audio_path,'combination',name);
        % audio_path相对dataset根目录；input_path/s1_path相对运行根目录。
        entry.audio_path=regexprep(p.audio_path,'^dataset/','');
        entry.duration_s=cfg.segment_len_s;
        if nt==1
            src=cache.sources(indices);
            entry.class_id=src.class_id; entry.class_name=src.class_name;
            entry.raw_relative_path=src.raw_relative_path; entry.segment_index=src.segment_index;
            entry.original_fs=src.original_fs; entry.start_sample=src.core_start_sample;
            entry.stop_sample=src.core_stop_sample; entry.QC_status=src.QC_status;
            entry.QC_reasons=src.QC_reasons; entry.input_path=src.input_path;
            entry.history_padding_samples=src.history_padding_samples;
            entry.multi_target_eligible=src.multi_target_eligible;
            entry.history_QC_status=src.history_QC_status; entry.history_QC_reasons=src.history_QC_reasons;
            entry.fir_supported=src.fir_supported;
        end
        entry.s1_path=component_paths{1}; entry.s2_path=component_paths{2}; entry.s3_path=component_paths{3};
        entry.combIdx=accepted+1; entry.candidateIdx=attempted; entry.n_targets=nt;
        entry.split=cfg.active_split; entry.fs=cfg.target_fs;
        entry.band_lo_hz=0; entry.band_hi_hz=cfg.target_fs/2;
        entry.source1_index=c.source1_index; entry.source2_index=c.source2_index;
        entry.recording1_index=c.recording1_index; entry.recording2_index=c.recording2_index;
        entry.recording1=c.recording1; entry.recording2=c.recording2;
        entry.range1_km=c.range1_km; entry.range2_km=c.range2_km;
        entry.source1_x_km=c.source1_x_km; entry.source2_x_km=c.source2_x_km;
        entry.source_depth1_m=c.source_depth1_m; entry.source_depth2_m=c.source_depth2_m;
        entry.keeper_slot=c.keeper_slot; entry.signed_offset_m=c.signed_offset_m;
        entry.separation_m=c.separation_m; entry.direction=c.direction;
        entry.sir_db=p.sir_db; entry.signed_sir_db=p.signed_sir_db; entry.sir_band=p.sir_band;
        entry.overlap_energy1=p.overlap_energy1; entry.overlap_energy2=p.overlap_energy2;
        entry.stronger_source=p.stronger_source;
        entry.card_target_db=c.card_target_db; entry.card_band=c.card_band;
        entry.predicted_signed_db=c.predicted_signed_db;
        entry.drift_db=p.signed_sir_db-c.predicted_signed_db;
        entry.stronger_scheduled=c.stronger_scheduled;
        entry.candidate_switches=c.candidate_switches;
        entry.keeper_redraws=c.keeper_redraws; entry.ladder_level=c.ladder_level;
        entry.shortlist_size=c.shortlist_size;
        entry.pair_search_reason=c.pair_search_reason;
        entry.receiver_depth_m=NaN;
        entry.receiver_x_km=NaN;
        if nt>0
            entry.receiver_depth_m=cache.geometry.receiver_depth_m;
            entry.receiver_x_km=cache.geometry.receiver_x_km;
        end
        entry.class1='none'; entry.class2='none'; entry.class_id1=NaN; entry.class_id2=NaN;
        filenames={'none','none','none'};
        for t=1:nt
            entry.(sprintf('class%d',t))=cfg.class_names{slots(t)};
            entry.(sprintf('class_id%d',t))=cfg.class_ids(slots(t));
            filenames{slots(t)}=cache.sources(indices(t)).row{5};
        end
        entry.fileA=filenames{1}; entry.fileB=filenames{2}; entry.fileC=filenames{3};
        entry.label_Cargo=ismember(1,slots); entry.label_Tanker=ismember(2,slots); entry.label_Tug=ismember(3,slots);
        entry.overlap_start_s=0; entry.overlap_end_s=cfg.segment_len_s;
        entry.receiver_model=cfg.receiver_model;
        entry.source_normalization=cfg.normalization.method;
        entry.normalization_scope=cfg.normalization.scope;
        entry.normalization_target_rms=cfg.normalization.target_rms;
        entry.input_gain1=NaN; entry.input_rms_before1=NaN; entry.input_rms_after1=NaN;
        if nt>0
            entry.input_gain1=cache.sources(indices(1)).input_gain;
            entry.input_rms_before1=cache.sources(indices(1)).input_rms_before;
            entry.input_rms_after1=cache.sources(indices(1)).input_rms_after;
        end
        entry.input_gain2=NaN; entry.input_rms_before2=NaN; entry.input_rms_after2=NaN;
        if nt==2
            entry.input_gain2=cache.sources(indices(2)).input_gain;
            entry.input_rms_before2=cache.sources(indices(2)).input_rms_before;
            entry.input_rms_after2=cache.sources(indices(2)).input_rms_after;
        end
        entry.common_scale=p.common_scale;
        entry.mix_rms=p.mix_rms;
        entry.mixing_stage='receiver_after_independent_propagation';
        entry.noise_model=c.noise_model;
        entry.noise_wind_speed_m_s=c.noise_wind_speed_m_s;
        entry.noise_shipping_factor=c.noise_shipping_factor;
        entry.noise_input_gain=c.noise_input_gain;
        entry.noise_input_rms_before=c.noise_input_rms_before;
        entry.noise_input_rms_after=c.noise_input_rms_after;
        entry.noise_receiver_target_rms=c.noise_receiver_target_rms;
        entry.noise_receiver_gain=c.noise_receiver_gain;
        if nt==0
            entry.mixing_stage='standalone_receiver_wenz_noise';
            entry.receiver_model='empirical_receiver_wenz_noise';
            entry.overlap_start_s=NaN; entry.overlap_end_s=NaN;
        end
        % Commit only after every component passes write/readback.
        if isempty(selected), selected=entry;
        else, selected(end+1,1)=entry; end %#ok<AGROW>
        uses(recording_indices)=uses(recording_indices)+1;
        clip_uses(indices)=clip_uses(indices)+1;
        if nt>0
            used_scenes(scene_key)=true;
            cache.total_clip_uses(indices)=cache.total_clip_uses(indices)+1;
        end
        if nt==2
            key=sprintf('%d_%d',indices);
            if ~isKey(pair_state.pair_uses,key), pair_state.pair_uses(key)=0; end
            pair_state.pair_uses(key)=pair_state.pair_uses(key)+1;
            pair_state.range_uses(ranges(1))=pair_state.range_uses(ranges(1))+1;
            pair_state.range_uses(ranges(2))=pair_state.range_uses(ranges(2))+1;
            primary_count=primary_count+strcmp(p.sir_band,'primary');
            supplement_count=supplement_count+strcmp(p.sir_band,'supplement');
            outside_count=outside_count+strcmp(p.sir_band,'outside');
            card_mismatch_count=card_mismatch_count+card_mismatch;
        end
        accepted=accepted+1;
        if mod(accepted,cfg.progress_every_scenes)==0 || accepted==requested
            fprintf('%s/%s: accepted %d/%d from %d proposals.\n', ...
                cfg.active_split,name,accepted,requested,attempted);
        end
    end
    if accepted==requested
        stop_reason='quota_filled';
    elseif attempted>=max_attempts
        stop_reason='attempt_budget_exhausted';
    else
        stop_reason='source_capacity_exhausted';
    end
catch err
    % Keep the one scene manifest for completed files if this combination stops.
    write_dataset_table(fullfile(folder,'all_info.txt'),selected);
    rethrow(err);
end
write_dataset_table(fullfile(folder,'all_info.txt'),selected);
unique_pairs=0; if nt==2, unique_pairs=pair_state.pair_uses.Count; end
mix_rms=[]; if ~isempty(selected), mix_rms=[selected.mix_rms]; end
result=struct('requested',requested,'accepted',accepted,'attempted',attempted, ...
    'shortage',requested-accepted,'stop_reason',stop_reason, ...
    'primary_sir_scenes',primary_count,'supplement_sir_scenes',supplement_count, ...
    'outside_sir_scenes',outside_count,'card_band_mismatch_scenes',card_mismatch_count, ...
    'primary_fraction_actual',primary_count/max(accepted,1), ...
    'supplement_fraction_actual',supplement_count/max(accepted,1), ...
    'used_recordings',nnz(uses),'used_clips',nnz(clip_uses),'unique_pairs',unique_pairs, ...
    'numeric_rejected',numeric_rejected,'pairing_rejected',pairing_rejected,'mix_rms',mix_rms);
end
function index=draw_clip(group,uses)
% 在整个合法片段池内优先使用累计次数最少者；并列时等概率。
group=group(uses(group)==min(uses(group)));
index=group(randi(numel(group)));
end

function [p,cache]=schedule_pair(cfg,cache,slots,state,depths,keeper,target_signed,card_band,used_scenes)
% 先按累计次数选片段，再用频域估计排序；仅对少量候选做完整传播。
% 估计不决定达标与否，最终 SIR 始终使用实际 5 秒、float32 分量能量。
p=struct('found',false,'reason','no_eligible_keeper');
matcher=3-keeper;
group=cache.groups{slots(keeper)};
eligible=group(state.recording_uses(group)<state.recording_caps(group));
clips=[cache.recordings(eligible).source_indices];
clips=clips(state.clip_uses(clips)<state.clip_caps(clips));
if isempty(clips), return; end
keeper_source=draw_clip(clips,state.sampling_uses);

matcher_group=cache.groups{slots(matcher)};
matcher_clips=[];
for rec=matcher_group(state.recording_uses(matcher_group)<state.recording_caps(matcher_group))
    rec_clips=cache.recordings(rec).source_indices;
    rec_clips=rec_clips(state.clip_uses(rec_clips)<state.clip_caps(rec_clips));
    matcher_clips=[matcher_clips rec_clips(arrayfun(@pair_open,rec_clips))]; %#ok<AGROW>
end
if isempty(matcher_clips)
    p.reason='no_eligible_matcher'; return;
end

best=[]; best_error=inf; switches=0; ladder=0;
visited_ranges=[]; previous_batch=[];
for redraw=0:cfg.pairing.keeper_redraws_max
    available_ranges=setdiff(1:numel(cfg.ranges_km),visited_ranges);
    if isempty(available_ranges), break; end
    keeper_range=available_ranges(randi(numel(available_ranges)));
    visited_ranges(end+1)=keeper_range; %#ok<AGROW>
    if redraw>0, ladder=max(ladder,3); end
    [e_keeper,cache]=propagate_sources(cfg,cache,keeper_source,keeper_range,depths(keeper),'energy');

    % 重抽位置时优先换一批搭档；次数优先、并列随机，不优先选缓存命中者。
    pool=setdiff(matcher_clips,previous_batch,'stable');
    if isempty(pool), pool=matcher_clips; end
    pool=pool(randperm(numel(pool)));
    [~,order]=sort(state.sampling_uses(pool));
    batch=pool(order(1:min(cfg.pairing.matcher_batch_size,numel(pool))));
    previous_batch=union(previous_batch,batch);
    range_grid=setdiff(1:numel(cfg.ranges_km),keeper_range);
    candidates=zeros(0,6);
    for clip=batch
        [energies,cache]=propagate_sources(cfg,cache,clip,range_grid,depths(matcher),'estimate');
        if keeper==1, predicted=10*log10(e_keeper./energies);
        else, predicted=10*log10(energies./e_keeper); end
        for j=1:numel(range_grid)
            indices=zeros(1,2); ranges=zeros(1,2);
            indices(keeper)=keeper_source; ranges(keeper)=keeper_range;
            indices(matcher)=clip; ranges(matcher)=range_grid(j);
            key=sprintf('%d_%d_%d;', [indices;ranges;depths]);
            if ~isfinite(predicted(j)) || isKey(used_scenes,key), continue; end
            candidates(end+1,:)=[clip range_grid(j) predicted(j) ... %#ok<AGROW>
                abs(predicted(j)-target_signed) state.sampling_uses(clip) state.range_uses(range_grid(j))];
        end
    end
    shortlist_size=size(candidates,1);
    candidates=candidates(randperm(shortlist_size),:);
    candidates=sortrows(candidates,[4 5 6]);
    checked_clips=[]; checked=0;
    while ~isempty(candidates) && (checked==0 || switches<cfg.pairing.candidate_switches_max)
        % 无解时先换搭档，所有本批搭档都试过后才重试同一搭档的其他位置。
        pick=find(~ismember(candidates(:,1),checked_clips),1);
        if isempty(pick), pick=1; end
        row=candidates(pick,:); candidates(pick,:)=[];
        if checked>0, switches=switches+1; ladder=max(ladder,2); end
        checked=checked+1; checked_clips(end+1)=row(1); %#ok<AGROW>
        [e_matcher,cache]=propagate_sources(cfg,cache,row(1),row(2),depths(matcher),'energy');
        indices=zeros(1,2); ranges=zeros(1,2); energy=zeros(1,2);
        indices(keeper)=keeper_source; ranges(keeper)=keeper_range; energy(keeper)=e_keeper;
        indices(matcher)=row(1); ranges(matcher)=row(2); energy(matcher)=e_matcher;
        measured=sir_from_energy(energy,cfg);
        match=strcmp(measured.sir_band,card_band);
        candidate=struct('found',true,'reason','scheduled','indices',indices,'ranges',ranges, ...
            'card_mismatch',~match,'predicted_signed_db',row(3), ...
            'candidate_switches',switches,'keeper_redraws',redraw, ...
            'ladder_level',ladder,'shortlist_size',shortlist_size);
        actual_error=abs(measured.signed_sir_db-target_signed);
        if actual_error<best_error, best=candidate; best_error=actual_error; end
        if match, p=candidate; return; end
    end
end
if ~isempty(best)
    % 沿用 L4：只能保存已经精算、未重复的合法候选，记录实际越界结果。
    p=best; p.reason='L4_best_verified_mismatch'; p.ladder_level=4;
    p.candidate_switches=switches; p.keeper_redraws=redraw;
else
    p.reason='no_unique_candidate';
end

    function tf=pair_open(clip_index)
        pair=zeros(1,2); pair(keeper)=keeper_source; pair(matcher)=clip_index;
        key=sprintf('%d_%d',pair);
        tf=~isKey(state.pair_uses,key) || state.pair_uses(key)<cfg.pairing.max_pair_uses;
    end
end

function deck=target_deck(cfg,requested,combo_name)
% Exact stratified target deck: primary_fraction of the cards draw inside the
% inset primary band, the rest inside the inset supplement band. Dealing by
% accepted-scene index makes the target-card split exact; L4 can change the
% realized waveform-band counts, which are reported separately.
n_primary=round(requested*cfg.pairing.primary_fraction);
n_supplement=requested-n_primary;
rng(v6_seed([cfg.active_split '|' combo_name '|deck'],cfg.physical_seed),'twister');
p=cfg.pairing.deck_primary_draw_db;
s=cfg.pairing.deck_supplement_draw_db;
target=[p(1)+(p(2)-p(1))*rand(1,n_primary),s(1)+(s(2)-s(1))*rand(1,n_supplement)];
band=[repmat({'primary'},1,n_primary),repmat({'supplement'},1,n_supplement)];
order=randperm(requested);
deck=struct('target_db',target(order),'band',{band(order)});
end

function s=sir_from_energy(energy,cfg)
% Mainline energies refer to both references over the same full 5-second window.
signed=10*log10(energy(1)/energy(2));
sir=abs(signed); band='outside';
if sir>cfg.pairing.primary_bounds_db(1) && sir<cfg.pairing.primary_bounds_db(2)
    band='primary';
elseif sir>cfg.pairing.supplement_bounds_db(1) && sir<cfg.pairing.supplement_bounds_db(2)
    band='supplement';
end
stronger=1; if signed<0, stronger=2; end
s=struct('sir_db',sir,'signed_sir_db',signed,'sir_band',band, ...
    'energy1',energy(1),'energy2',energy(2),'stronger_source',stronger);
end

function x=generate_wenz_noise(cfg,wind_speed,shipping_factor)
% 三分量经验谱只用于谱形成形；最终 RMS 由归一化及 Fix-D 决定。
% 沿用旧模型：低于 10 Hz 不计湍流项，航运/风噪项仍保留。
% 该低频边界是本实现约定，不是对当前海域的实测标定。
N=cfg.target_fs*cfg.segment_len_s;
f=(1:floor(N/2)).'*(cfg.target_fs/N); fk=f/1000;
turbulence=10.^((17-30*log10(max(fk,cfg.noise.turbulence_min_hz/1000)))/10);
turbulence(f<cfg.noise.turbulence_min_hz)=0;
shipping=10.^((40+20*(shipping_factor-0.5)+26*log10(fk)-60*log10(fk+0.03))/10);
wind=10.^((50+7.5*sqrt(wind_speed)+20*log10(fk)-40*log10(fk+0.4))/10);
amplitude=sqrt(turbulence+shipping+wind);
positive=amplitude.*(randn(size(f))+1i*randn(size(f)))/sqrt(2);
if mod(N,2)==0
    positive(end)=amplitude(end)*randn;
    spectrum=[0;positive;conj(flipud(positive(1:end-1)))];
else
    spectrum=[0;positive;conj(flipud(positive))];
end
x=real(ifft(spectrum)); x=x-mean(x);
end

function [x,report]=normalize_noise_input(x,cfg)
% 合成 noise 的完整 5 秒 RMS 归一化；随后由 Fix-D 设置接收电平。
input_rms=sqrt(mean(x.^2));
assert(isfinite(input_rms) && input_rms>0,'Cannot normalize an empty/zero-power input.');
gain=cfg.normalization.target_rms/input_rms;
x=x*gain;
report=struct('input_gain',gain,'input_rms_before',input_rms, ...
    'input_rms_after',sqrt(mean(x.^2)));
end

function record=write_scene(root,name,targets,slots,component_paths,cfg,noise_clip,expected_band)
% 每个实际分量只保存一份；单目标直接引用分量，双目标/noise 单独保存场景。
N=cfg.target_fs*cfg.segment_len_s;
decoded=zeros(N,numel(slots));
for t=1:numel(slots)
    path=fullfile(cfg.run_root,component_paths{slots(t)});
    component=single(targets(:,t)*cfg.output_gain);
    decoded(:,t)=write_wave(path,component,cfg);
end
if numel(slots)==1
    audio_path=component_paths{slots};
    mix=decoded(:,1);
else
    if isempty(slots), output=noise_clip; else, output=sum(targets,2); end
    if isempty(slots)
        root=fullfile(root,'mix');
        if ~isfolder(root), mkdir(root); end
    end
    mix=write_wave(fullfile(root,name),single(output*cfg.output_gain),cfg);
    relative_root=root(numel(cfg.run_root)+2:end);
    audio_path=strrep(fullfile(relative_root,name),'\','/');
end
[~,stem,extension]=fileparts(audio_path);
sir=struct('sir_db',NaN,'signed_sir_db',NaN,'sir_band','not_applicable', ...
    'energy1',NaN,'energy2',NaN,'stronger_source',NaN);
if numel(slots)==2
    sir=sir_from_energy(sum(decoded.^2,1),cfg);
    assert(strcmp(expected_band,'any') || strcmp(sir.sir_band,expected_band), ...
        'Saved scene differs from its selected SIR band.');
end
record=struct('file_name',[stem extension],'audio_path',audio_path, ...
    'common_scale',cfg.output_gain,'mix_rms',sqrt(mean(mix.^2)), ...
    'sir_db',sir.sir_db,'signed_sir_db',sir.signed_sir_db,'sir_band',sir.sir_band, ...
    'overlap_energy1',sir.energy1,'overlap_energy2',sir.energy2,'stronger_source',sir.stronger_source);
end

function wave=write_wave(path,output,cfg)
% 已有集中分量直接回读；新文件只写一次，不生成全零占位文件。
N=cfg.target_fs*cfg.segment_len_s;
assert(isequal(size(output),[N 1]) && all(isfinite(output)) && any(output~=0), ...
    'Invalid or silent output waveform.');
if ~isfile(path)
    v7_write_float_wav(path,output,cfg.target_fs);
end
[wave,fs]=audioread(path);
assert(fs==cfg.target_fs && isequal(size(wave),[N 1]) && all(isfinite(wave)) && any(wave~=0), ...
    'Invalid saved WAV: %s.',path);
end
