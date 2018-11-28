import cv2
import numpy as np
#双线性插值算法 bilinear interpolation
def function_bili(img,dst_width,dst_height):

    src_height,src_width,channels = img.shape
    emptyImage = np.zeros((dst_height,dst_width,channels),np.uint8)
    value = [0,0,0]
    w_h = src_height/dst_height
    w_w = src_width/dst_width
    for i in range(dst_height):
        for j in range(dst_width):
            srcx = i*w_h
            srcy = j*w_w
            u = srcx-int(srcx)
            v = srcy-int(srcy)
            x = int(srcx)
            y = int(srcy)
            if x+1<src_height and y+1<src_width:
                f_ij = img[x,y]
                f_ija1 = img[x,y+1]
                f_ia1j = img[x+1,y]
                f_iaja = img[x+1,y+1]
            for k in range(3):
                value[k] = (1-u)*(1-v)*f_ij[k] + (1-u)*v*f_ija1[k] + u*(1-v)*f_ia1j[k] + u*v*f_iaja[k]

            emptyImage[i,j] = (value[0],value[1],value[2])
    return emptyImage
