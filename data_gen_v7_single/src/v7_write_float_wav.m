function v7_write_float_wav(path,x,fs)
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

