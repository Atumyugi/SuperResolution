import cv2
import numpy as np
#临近插值算法nearest neighbor
def function_near(img,shape_width,shape_height):

    height,width,channels = img.shape
    emptyImage = np.zeros((shape_height,shape_width,channels), np.uint8)
    w_h = height / shape_height   #1/放大倍数
    w_w = width / shape_width      #1/放大倍数
    for i in range(len(emptyImage)):
        for j in range(len(emptyImage[0])):
            x = int(i*w_h)
            y = int(j*w_w)
            emptyImage[i, j] = img[x, y] #把三个通道的值都传过去了
    return emptyImage
