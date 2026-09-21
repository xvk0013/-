function write_dataset_table(path,header,rows)
% Write one TSV table; a struct array may be supplied instead of header/rows.
if nargin==2
    records=header;
    if isempty(records), return; end
    header=fieldnames(records).';
    rows=cell(numel(records),numel(header));
    for r=1:numel(records)
        for c=1:numel(header), rows{r,c}=records(r).(header{c}); end
    end
end
fid=fopen(path,'wt','n','UTF-8');
assert(fid>=0,'Cannot write %s.',path);
close_file=onCleanup(@() fclose(fid));
data=[header;rows];
for r=1:size(data,1)
    fields=cell(1,size(data,2));
    for c=1:size(data,2)
        v=data{r,c};
        if isnumeric(v) || islogical(v)
            if isscalar(v), v=sprintf('%.17g',v); else, v=mat2str(v,17); end
        end
        fields{c}=regexprep(char(v),'[\t\r\n]',' ');
    end
    fprintf(fid,'%s\n',strjoin(fields,sprintf('\t')));
end
end
