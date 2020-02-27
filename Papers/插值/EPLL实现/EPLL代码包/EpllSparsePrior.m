function [x,D,Psnr] = EpllSparsePrior(y,y0,D,sigma,OuterIter,InnerIter)
%    
% [x,D,Psnr] = EpllSparsePrior(y,y0,D,sigma,OuterIter,InnerIter)
%   
% 
%   参数：
%         输入：    - y： 加噪图像
%                   - y0： 原始图像 （用于PSNR评价）
%                   - sigma： 噪声标准差
%                   - D： 词典
%                   - OuterIter： 外部循环次数（去噪循环）
%                   - InnerIter： 内部循环次数（词典训练循环）
% 
%         输出：    -x： 去噪图像
%                   -D： 训练过的词典
%                   -Psnr： PSNR评价
% 

%  注意： 该函数需要使用K-SVD和OMP工具包，请将这两个工具包设置在Matlab路径中，
%  工具包可从以下链接中下载：
%  http://www.cs.technion.ac.il/~ronrubin/software.html
% 

% 
%   检查是否安装K-SVD和OMP工具包
    
    if ( exist('omp2','file')==0 ||  exist('ksvd','file')==0 )
        %%
        % 
        %   for x = 1:10
        %       disp(x)
        %   end
        % 
        error('ksvd and omp packages needed.')
    end


Psnr = zeros(OuterIter+1,1);
Psnr(1) = PSNR(y,y0);
A = y;
Std = sigma;
[n,m] = size(D);
Yn = getPatches(y,sqrt(n));


for iter=1:OuterIter
    disp(['Outer-Iter ',num2str(iter),'...']);
    lamda = 20/Std;
    [A,D,X] = CleanKSVD_ms(A,n,m,sqrt(n)*Std,InnerIter,D,lamda);
   
    % 估计下一个阈值
    M = size(X,2);
    Nnz = zeros(n,M);
    for i=1:M
        Nnz(:,i)=(nnz(X(:,i))+1)* Std^2/n;
    end
    Nz = Std*ones(size(y));
    
    Nz = sqrt((myRecoverImage2(Nz,Nnz,sqrt(n),lamda)));
    [h,x] = hist(Nz(:),50);
    [~,ma] = max(h);
    mode = x(ma);

    % 修正因子
    if iter==1,
        Yd = D*X;
        StdEst = sqrt( abs(sigma^2 - var( Yd - Yn )) );
        ErSt = zeros(size(Yd));
        for i=1:M
            ErSt(:,i)=StdEst(i)^2;
        end
        ErIm = myRecoverImage2(Std*ones(size(y)),ErSt,sqrt(n),lamda);
        ErStf = sqrt(sum(getPatches(ErIm,sqrt(n)),1))/sqrt(n);
        [H,x2]=hist(ErStf,50);
        [~,Ma]=max(H);
        ModeErStdf = x2(Ma);
        %定义修正因子
        factor = ModeErStdf/mode;    
    end
    Std = mode*factor;
    Psnr(iter+1) = PSNR(A,y0);
end

x = A;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%       其他相关函数      %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [ PsnrDb,MSE ] = PSNR(I,Itest,varargin)
if nargin==2,
    R=255;
end
if nargin==3,
    v=cell2mat(varargin);
    R=v(1);
end

if size(I,1)~=size(Itest,1)||size(I,2)~=size(Itest,2)
    error('Error.The data to compare must have the same size. Sorry!')
    return;
end

[n m]=size(I);

MSE=sum(sum((I-Itest).^2))/(m*n);

PsnrDb=10*log10(R^2/MSE);

end


%% 重建图像
function Aout = myRecoverImage2(A,Yd,p,lamda)
[~,M]=size(A);
i=1;
j=1;
Weight=zeros(size(A));
Aout=zeros(size(A));
for k=1:size(Yd,2)
    patch=reshape(Yd(:,k),[p,p]);
    Aout(i:i+p-1,j:j+p-1)=Aout(i:i+p-1,j:j+p-1)+patch;
    Weight(i:i+p-1,j:j+p-1)=Weight(i:i+p-1,j:j+p-1)+1;
    if j<M-p+1 
        j=j+1; 
    else
        j=1; i=i+1; 
    end;
end;    
Aout=(lamda*A+Aout)./(lamda+Weight);			%公式（9）

end


%% 训练词典并估计重建图像
function [Aout,Dict,X,Yd] = CleanKSVD_ms(A,n,m,E,IterNum,InitDict,lamda)

[N,M]=size(A);
p=sqrt(n);
%Y=zeros(n,(N-p+1)*(M-p+1));
%pt=1;
%for i=1:1:N-p+1
%    for j=1:1:M-p+1
%        patch=A(i:i+p-1,j:j+p-1);
%        Y(:,pt)=patch(:);
%        pt=pt+1;
%    end
%end
Y = getPatches(A,p);

if isempty(InitDict)
    params.initdict=odct2dict([sqrt(n),sqrt(n)],[sqrt(m),sqrt(m)]);
else
    params.initdict=InitDict;
end
params.Edata=E; 
means = mean(Y,1);
params.data=Y-ones(n,1)*means;
params.iternum=IterNum;
[Dict,X]=ksvd(params,'');

Yd=Dict*X+ones(n,1)*means;

i=1;
j=1;
Weight=zeros(size(A));
Aout=zeros(size(A));
for k=1:size(Yd,2)
    patch=reshape(Yd(:,k),[p,p]);
    Aout(i:i+p-1,j:j+p-1)=Aout(i:i+p-1,j:j+p-1)+patch;
    Weight(i:i+p-1,j:j+p-1)=Weight(i:i+p-1,j:j+p-1)+1;
    if j<M-p+1 
        j=j+1; 
    else
        j=1; i=i+1; 
    end
end    
Aout=(lamda*A+Aout)./(lamda+Weight);			%公式（9）

end



%% 获取图像中重叠小块
function Y = getPatches(A,p)
[N,M]=size(A);
n=p^2;
Y=zeros(n,(N-p+1)*(M-p+1));
pt=1;
for i=1:1:N-p+1
    for j=1:1:M-p+1
        patch=A(i:i+p-1,j:j+p-1);
        Y(:,pt)=patch(:);
        pt=pt+1;
    end
end

end
