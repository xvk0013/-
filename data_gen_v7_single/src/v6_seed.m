function seed=v6_seed(key,base)
% 仅生成可复现的抽样种子；不读取文件，也不用于声学缓存或来源匹配。
md=java.security.MessageDigest.getInstance('SHA-256');
md.update(uint8(unicode2native(sprintf('%d|%s',base,char(key)),'UTF-8')));
bytes=typecast(md.digest(),'uint8');
seed=sum(double(bytes(1:4)).*reshape(256.^(3:-1:0),size(bytes(1:4))));
end
