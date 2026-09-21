function [value,cache]=propagate_sources(cfg,cache,index,r,d,mode)
% wave: 接收波形；energy: 精算/复用 5 秒能量；estimate: 小批候选频域排序。
if nargin<6, mode='wave'; end
cache.clock=cache.clock+1;
if strcmp(mode,'estimate')
    value=cache.energy_table(index,r,d);
    missing=isnan(value);
    value(missing)=cache.energy_estimates(index,r(missing),d);
    missing=isnan(value);
    if any(missing)
        [Fx,cache]=source_fft(cfg,cache,index);
        weighted=2*abs(Fx).^2; weighted(1)=weighted(1)/2;
        nfft=cache.geometry.nfft;
        if mod(nfft,2)==0, weighted(end)=weighted(end)/2; end
        % Parseval 全响应能量乘 5/8，作为平稳条件下的接收窗估计。
        % 不作逐片段拟合或在线校准，精确验收仍取真实 5 秒窗口。
        scale=cfg.segment_len_s/(cfg.segment_len_s+cfg.source_history_s) ...
            *cfg.output_gain^2/nfft;
        for j=find(missing)
            value(j)=scale*(weighted.'*cache.channel_power{r(j),d});
            cache.energy_estimates(index,r(j),d)=value(j);
        end
    end
    return;
end
if strcmp(mode,'energy') && ~isnan(cache.energy_table(index,r,d))
    value=cache.energy_table(index,r,d); return;
end
y=cache.waves{index,r,d};
if isempty(y)
    [Fx,cache]=source_fft(cfg,cache,index);
    y=apply_channel(Fx,cache.channels{r,d}, ...
        round(cfg.source_history_s*cfg.target_fs),round(cfg.segment_len_s*cfg.target_fs));
    assert(isequal(size(y),[round(cfg.segment_len_s*cfg.target_fs) 1]) && ...
        isreal(y) && all(isfinite(y)) && any(y~=0), ...
        'V6:PropagationCache','Invalid propagated reference.');
    if cache.wave_count>=cfg.cache.waveforms_in_memory
        eligible=find(cache.wave_ticks>0);
        [~,j]=min(cache.wave_ticks(eligible)); victim=eligible(j);
        cache.waves{victim}=[]; cache.wave_ticks(victim)=0;
        cache.wave_count=cache.wave_count-1;
    end
    cache.waves{index,r,d}=y; cache.wave_count=cache.wave_count+1;
end
cache.wave_ticks(index,r,d)=cache.clock;
if isnan(cache.energy_table(index,r,d))
    cache.energy_table(index,r,d)=sum(double(single(y*cfg.output_gain)).^2);
end
if strcmp(mode,'energy'), value=cache.energy_table(index,r,d);
else, value=y; end
end

function [Fx,cache]=source_fft(cfg,cache,index)
% 只保留近期片段的输入 FFT；同一片段尝试不同位置时直接复用。
Fx=cache.source_ffts{index};
if isempty(Fx)
    context=load(cache.sources(index).context_path,'x'); x=context.x(:);
    N=round(cfg.segment_len_s*cfg.target_fs); P=round(cfg.source_history_s*cfg.target_fs);
    nfft=cache.geometry.nfft;
    assert(numel(x)==P+N && all(isfinite(x)),'Invalid source context.');
    full=fft(x,nfft); Fx=full(1:floor(nfft/2)+1);
    if cache.source_fft_count>=cfg.cache.source_ffts_in_memory
        eligible=find(cache.source_fft_ticks>0);
        [~,j]=min(cache.source_fft_ticks(eligible)); victim=eligible(j);
        cache.source_ffts{victim}=[]; cache.source_fft_ticks(victim)=0;
        cache.source_fft_count=cache.source_fft_count-1;
    end
    cache.source_ffts{index}=Fx; cache.source_fft_count=cache.source_fft_count+1;
end
cache.source_fft_ticks(index)=cache.clock;
end

function y=apply_channel(Fx,channel,history_samples,core_samples)
% 与原主线相同的线性卷积及截窗，只复用已经计算的输入频谱。
nfft=channel.nfft; P=channel.memory_samples;
assert(history_samples>=P && nfft>=history_samples+core_samples+P, ...
    'Insufficient history or FFT length for linear convolution.');
positive=Fx.*channel.H;
positive(1)=real(positive(1));
if mod(nfft,2)==0
    positive(end)=real(positive(end)); full=[positive;conj(positive(end-1:-1:2))];
else
    full=[positive;conj(positive(end:-1:2))];
end
response=real(ifft(full));
y=response(history_samples+(1:core_samples));
end
