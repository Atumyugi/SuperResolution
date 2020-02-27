%% ȥ��������

clear
clc

%% ����ѡ��
sigma = 30;                 % ���������ǿ��
OuterIter = 3;              % ȥ�����ѭ������
InnerIter = 15;             % ѵ������ѭ������

load Im            % ����ͼ��
load Dict                   % ����ѵ���ʵ�

y0 = Im; 
imshow(y0,[0 255]),
% ԭʼͼ��
y = Im + sigma*randn(size(Im));         % ����ͼ��
imshow(y,[0 255]),
tic
[x,D,Psnr] = EpllSparsePrior(y,y0,Dict,sigma,OuterIter,InnerIter);     % �������
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