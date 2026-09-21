function assignment=balance_recordings(counts,fractions,seed,slack_fraction,max_steps)
% 按切片数平衡，但移动/交换的最小单位永远是整条录音。
counts=double(counts(:)); n=numel(counts); fractions=fractions(:).';
assert(n>=3 && all(counts>0) && numel(fractions)==3 && ...
    abs(sum(fractions)-1)<1e-12,'每类至少需要3条含合格片段的录音。');
old=rng; cleanup=onCleanup(@() rng(old)); rng(seed,'twister');
exact=n*fractions; quota=floor(exact);
[~,order]=sort(exact-quota,'descend');
quota(order(1:n-sum(quota)))=quota(order(1:n-sum(quota)))+1;
while any(quota==0)
    missing=find(quota==0,1); [~,donor]=max(quota);
    quota(donor)=quota(donor)-1; quota(missing)=1;
end
% 长录音优先，等长时用固定种子排序；不读取预测或识别成绩。
[~,order]=sortrows([-counts rand(n,1)],[1 2]);
target=sum(counts)*fractions; loads=zeros(1,3); numbers=loads;
assignment=zeros(n,1);
for j=order.'
    available=find(numbers<quota);
    [~,k]=min(loads(available)./target(available)); s=available(k);
    assignment(j)=s; loads(s)=loads(s)+counts(j); numbers(s)=numbers(s)+1;
end
% 不锁死录音配额。有限的整录音移动/交换优先减小切片数量误差。
slack=max(2,ceil(n*slack_fraction));
lower=max(1,quota-slack); upper=min(n-2,quota+slack);
for step=1:max_steps
    best_gain=0; action=[];
    for a=1:2
        for b=a+1:3
            ia=find(assignment==a); ib=find(assignment==b);
            delta=counts(ib).'-counts(ia);
            old_error=(loads(a)-target(a))^2+(loads(b)-target(b))^2;
            gain=old_error-(loads(a)+delta-target(a)).^2-(loads(b)-delta-target(b)).^2;
            [value,index]=max(gain(:));
            if value>best_gain+1e-8
                [u,v]=ind2sub(size(gain),index);
                best_gain=value; action=[2 ia(u) ib(v) a b];
            end
        end
    end
    for a=1:3
        if numbers(a)<=lower(a), continue; end
        ia=find(assignment==a);
        for b=setdiff(1:3,a)
            if numbers(b)>=upper(b), continue; end
            w=counts(ia);
            gain=(loads(a)-target(a))^2+(loads(b)-target(b))^2 ...
                -(loads(a)-w-target(a)).^2-(loads(b)+w-target(b)).^2;
            [value,u]=max(gain);
            if value>best_gain+1e-8
                best_gain=value; action=[1 ia(u) 0 a b];
            end
        end
    end
    if isempty(action), break; end
    i=action(2); a=action(4); b=action(5);
    if action(1)==2
        j=action(3); delta=counts(j)-counts(i);
        assignment([i j])=[b;a]; loads([a b])=loads([a b])+[delta -delta];
    else
        assignment(i)=b; loads([a b])=loads([a b])+[-counts(i) counts(i)];
        numbers([a b])=numbers([a b])+[-1 1];
    end
end
end
