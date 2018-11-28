#使用图片的自相似进行图像放大,高频补偿使用SAD的方法,在放大缩小得到图像本身损失LOSS数组的方法是线性插值
import cv2
import numpy as np
#匹配最佳相似度 SAD
import Interpolation-Bilinear
def SAD(loss):
    sad = 0
    for i in range(len(loss)):
        for j in range(len(loss[0])):
            sad = sad + loss[i][j]
    return sad

#Local-similar 具体算法
def freqSAD(axa,bxb):
    simialr = np.zeros(axa.shape,dtype=np.uint8)
    sadmin = 100000
    ij = [0,0]
    lenh = len(bxb)-len(axa)+1
    lenw = len(bxb[0])-len(axa[0])+1
    for i in range(lenh):
        for j in range(lenw):
            bxbcut = bxb[i:i+len(axa),j:j+len(axa[0])]
            loss = bxbcut-axa
            sad = SAD(loss)
            if sad<sadmin:
                sadmin = sad
                simialr = loss
                ij = [i,j]
    return ij

#3:2 放大图像
def resize3bi2(a):
    height, weight,_ = a.shape
    hs, ws = int(height / 1.5), int(weight / 1.5)
    hb, wb = int(height * 1.5), int(weight * 1.5)

    a_1s = Interpolation-Bilinear.function_bili(a, ws, hs)
    #a_1s = cv2.resize(a, (ws, hs), interpolation=cv2.INTER_LINEAR)
    a_1 = cv2.resize(a_1s, (weight, height), interpolation=cv2.INTER_LINEAR)
    e = a - a_1
    a_2 = cv2.resize(a, (wb, hb), interpolation=cv2.INTER_LINEAR)
    for k in range(3):
        i = 0
        j = 0
        while (i < len(a_2) - 5 and j < len(a_2[0]) - 5):
            axa = a_2[i:i + 5, j:j + 5][k]
            bxb = a_2[i:i + 10, j:j + 10][k]
            ij = freqSAD(axa, bxb)
            ii, jj = ij
            a_2[i:i + 5, j:j + 5][k] = a_2[i:i + 5, j:j + 5][k] + e[ii:ii + 5, jj:jj + 5][k]
            i = i + 5
            j = j + 5
    return a_2

xin = cv2.imread('xin.jpg')  #替换成自己的图片路径
height, weight,_ = xin.shape
xin_2 = resize3bi2(xin)
xin_3 = resize3bi2(xin_2)
xin_4 = resize3bi2(xin_3)
xin_5 = cv2.resize(xin_2,(3*weight,3*height),interpolation=cv2.INTER_CUBIC)
xin_liner = cv2.resize(xin,(3*weight,3*height),interpolation=cv2.INTER_LINEAR)
xin_cubic = cv2.resize(xin,(3*weight,3*height),interpolation=cv2.INTER_CUBIC)
cv2.imshow('MY',xin_5)       #Local-similar 算法的图像
cv2.imshow('CUBIC',xin_cubic)  #cubic 算法的图像
cv2.waitKey(0)
cv2.destroyAllWindows()
