%% 去噪声程序

clear
clc

%% 参数选择
sigma = 30;                 % 加入的噪声强度
OuterIter = 3;              % 去噪迭代循环次数
InnerIter = 15;             % 训练迭代循环次数

load Im            % 载入图像
load Dict                   % 载入训练词典

y0 = Im; 
imshow(y0,[0 255]),
% 原始图像
y = Im + sigma*randn(size(Im));         % 加噪图像
imshow(y,[0 255]),
tic
[x,D,Psnr] = EpllSparsePrior(y,y0,Dict,sigma,OuterIter,InnerIter);     % 函数入口
toc

figure,
imshow(y0,[0 255]),
title('Original Image')
figure,
imshow(y,[0 255]),
title(['Noisy Image. \sigma = ',num2str(sigma)])
figure
imshow(x,[0 255])
title(['Denoised Image. PSNR = ',num2str(Psnr(end))])