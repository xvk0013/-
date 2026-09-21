function q=v7_waveform_quality(x,fs,cfg,expected_samples)
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

