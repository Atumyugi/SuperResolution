import torch
import cv2
import h5py
import numpy as np
from torch import nn,optim
import torch.nn.functional as F
from torchvision import transforms
import torch.utils.data as dataf
import matplotlib.pyplot as plt
import torchvision
#from torchsummary import summary
import math
from model  import wdsr_b
import skimage.measure as measure
import argparse,os
from EDSR import common_utils
from EDSR import config
from tqdm import tqdm
parser = config.get_args()


def WDSR_super_resolution(testimg_path,labelimg_path,test_directory_path,label_directory_path,txt_path,save_path,wdsr_model):

    lr_path = test_directory_path + testimg_path
    hr_path = label_directory_path + labelimg_path
    #print(lr_path,hr_path)
    img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
    print(testimg_path,' shape: ',img.shape)
    source_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
    source_img_y = cv2.cvtColor(source_img, cv2.COLOR_RGB2YCrCb)
    source_img_y = source_img_y[:, :, 0]
    height, weight, _ = img.shape
    sr_height, sr_weight = height * 4, weight * 4

    img_cubic = cv2.resize(img, (sr_weight, sr_height), cv2.INTER_CUBIC)

    cv2.imwrite(save_path+'Cubic_'+testimg_path, img_cubic)
    # height,weight = height*4,weight*4
    # SR_img = cv2.resize(img,(int(weight/4),int(height/4)),cv2.INTER_CUBIC)
    SR_img = np.transpose(img, [2, 0, 1])
    input_x = np.zeros((1, 3, height, weight), dtype=float)
    input_x[0, :, :, :] = SR_img  # .astype(float)

    crop_height = height // 2
    crop_weight = weight // 4
    input_x_left_top = input_x[:, :, 0:crop_height, 0:crop_weight]
    input_x_left_mid_top = input_x[:, :, 0:crop_height, crop_weight:2 * crop_weight]
    input_x_right_mid_top = input_x[:, :, 0:crop_height, 2 * crop_weight:3 * crop_weight]
    input_x_right_top = input_x[:, :, 0:crop_height, 3 * crop_weight:weight]
    input_x_left_bot = input_x[:, :, crop_height:height, 0:crop_weight]
    input_x_left_mid_bot = input_x[:, :, crop_height:height, crop_weight:2 * crop_weight]
    input_x_right_mid_bot = input_x[:, :, crop_height:height, 2 * crop_weight:3 * crop_weight]
    input_x_right_bot = input_x[:, :, crop_height:height, 3 * crop_weight:weight]

    out_left_top = common_utils.predict(input_x_left_top,wdsr_model)
    out_left_mid_top = common_utils.predict(input_x_left_mid_top,wdsr_model)
    out_right_mid_top = common_utils.predict(input_x_right_mid_top,wdsr_model)
    out_right_top = common_utils.predict(input_x_right_top,wdsr_model)
    out_left_bot = common_utils.predict(input_x_left_bot,wdsr_model)
    out_left_mid_bot = common_utils.predict(input_x_left_mid_bot,wdsr_model)
    out_right_mid_bot = common_utils.predict(input_x_right_mid_bot,wdsr_model)
    out_right_bot = common_utils.predict(input_x_right_bot,wdsr_model)
    out = np.zeros((height * 4, weight * 4, 3))
    out[0:height * 2, 0:weight, :] = out_left_top
    out[0:height * 2, weight:2 * weight, :] = out_left_mid_top
    out[0:height * 2, 2 * weight:3 * weight, :] = out_right_mid_top
    out[0:height * 2, 3 * weight:4 * weight, :] = out_right_top
    out[height * 2:height * 4, 0:weight, :] = out_left_bot
    out[height * 2:height * 4, weight:2 * weight, :] = out_left_mid_bot
    out[height * 2:height * 4, 2 * weight:3 * weight, :] = out_right_mid_bot
    out[height * 2:height * 4, 3 * weight:4 * weight, :] = out_right_bot

    # out  = out*255
    out[out[:] > 255] = 255
    out[out[:] < 0] = 0
    out = out.astype(np.uint8)

    cv2.imwrite(save_path+'WDSR_'+testimg_path, out)
    img_cubic_y = cv2.cvtColor(img_cubic, cv2.COLOR_RGB2YUV)
    img_cubic_y = img_cubic_y[:, :, 0]
    out_y = cv2.cvtColor(out, cv2.COLOR_RGB2YUV)
    out_y = out_y[:, :, 0]
    f = open(txt_path,'a')
    f.write(testimg_path+'\n')
    f.write('PSNR:CUBIC_Y\n')
    f.write(str(common_utils.PSNR(source_img_y, img_cubic_y))+'\n')
    f.write('PSNR:WDSR_Y'+'\n')
    f.write(str(common_utils.PSNR(source_img_y, out_y))+'\n')
    f.write('PSNR:CUBIC'+'\n')
    f.write(str(common_utils.PSNR(source_img, img_cubic))+'\n')
    f.write('PSNR:WDSR'+'\n')
    f.write(str(common_utils.PSNR(source_img, out))+'\n')
    f.write('SSIM:CUBIC'+'\n')
    f.write(str(common_utils.SSIM(source_img, img_cubic))+'\n')
    f.write('SSIM:WDSR'+'\n')
    f.write(str(common_utils.SSIM(source_img, out))+'\n')
    f.close()
    print(testimg_path,'finished......')
    return common_utils.PSNR(source_img, img_cubic),common_utils.PSNR(source_img, out)

def WDSR_PSNRavg(testimg_path,labelimg_path,test_directory_path,label_directory_path,wdsr_model):

    lr_path = test_directory_path + testimg_path
    hr_path = label_directory_path + labelimg_path
    #print(lr_path,hr_path)
    img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
    source_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
    source_img_y = cv2.cvtColor(source_img, cv2.COLOR_RGB2YCrCb)
    source_img_y = source_img_y[:, :, 0]
    height, weight, _ = img.shape
    sr_height, sr_weight = height * 4, weight * 4

    img_cubic = cv2.resize(img, (sr_weight, sr_height), cv2.INTER_CUBIC)

    SR_img = np.transpose(img, [2, 0, 1])
    input_x = np.zeros((1, 3, height, weight), dtype=float)
    input_x[0, :, :, :] = SR_img  # .astype(float)

    crop_height = height // 2
    crop_weight = weight // 4
    input_x_left_top = input_x[:, :, 0:crop_height, 0:crop_weight]
    input_x_left_mid_top = input_x[:, :, 0:crop_height, crop_weight:2 * crop_weight]
    input_x_right_mid_top = input_x[:, :, 0:crop_height, 2 * crop_weight:3 * crop_weight]
    input_x_right_top = input_x[:, :, 0:crop_height, 3 * crop_weight:weight]
    input_x_left_bot = input_x[:, :, crop_height:height, 0:crop_weight]
    input_x_left_mid_bot = input_x[:, :, crop_height:height, crop_weight:2 * crop_weight]
    input_x_right_mid_bot = input_x[:, :, crop_height:height, 2 * crop_weight:3 * crop_weight]
    input_x_right_bot = input_x[:, :, crop_height:height, 3 * crop_weight:weight]

    out_left_top = common_utils.predict(input_x_left_top,wdsr_model)
    out_left_mid_top = common_utils.predict(input_x_left_mid_top,wdsr_model)
    out_right_mid_top = common_utils.predict(input_x_right_mid_top,wdsr_model)
    out_right_top = common_utils.predict(input_x_right_top,wdsr_model)
    out_left_bot = common_utils.predict(input_x_left_bot,wdsr_model)
    out_left_mid_bot = common_utils.predict(input_x_left_mid_bot,wdsr_model)
    out_right_mid_bot = common_utils.predict(input_x_right_mid_bot,wdsr_model)
    out_right_bot = common_utils.predict(input_x_right_bot,wdsr_model)
    out = np.zeros((height * 4, weight * 4, 3))
    out[0:height * 2, 0:weight, :] = out_left_top
    out[0:height * 2, weight:2 * weight, :] = out_left_mid_top
    out[0:height * 2, 2 * weight:3 * weight, :] = out_right_mid_top
    out[0:height * 2, 3 * weight:4 * weight, :] = out_right_top
    out[height * 2:height * 4, 0:weight, :] = out_left_bot
    out[height * 2:height * 4, weight:2 * weight, :] = out_left_mid_bot
    out[height * 2:height * 4, 2 * weight:3 * weight, :] = out_right_mid_bot
    out[height * 2:height * 4, 3 * weight:4 * weight, :] = out_right_bot

    # out  = out*255
    out[out[:] > 255] = 255
    out[out[:] < 0] = 0
    out = out.astype(np.uint8)

    return common_utils.PSNR(source_img, img_cubic),common_utils.PSNR(source_img, out)
def main():
    global opt
    opt = parser.parse_args()
    print(opt)
    #edsr_model = wdsr_b.MODEL()
    #print(model)
    #summary(model,(3,40,40))
    wdsr_model = torch.load(opt.model_path)
    wdsr_model.cuda()


    lr_names = os.listdir(opt.test_directory_path)
    lr_names = sorted(lr_names)  # 排序
    hr_names = os.listdir(opt.label_directory_path)
    hr_names = sorted(hr_names)
    nums = lr_names.__len__()  # 训练图片的数量


    cubic_avg = []
    wdsr_avg = []
    if opt.test_type == 'A':
        print('Images need to be handle:', nums)
        for i in range(nums):
            print('The',i,'/',nums,'image:')
            lr_name = lr_names[i]
            hr_name = hr_names[i]
            cubic_psnr,out_psnr = WDSR_super_resolution(lr_name,hr_name,opt.test_directory_path,opt.label_directory_path,opt.txt_path,opt.save_path,wdsr_model)
            cubic_avg.append(cubic_psnr)
            wdsr_avg.append(out_psnr)
    elif opt.test_type == 'B':
        for i in tqdm(range(nums)):
            lr_name = lr_names[i]
            hr_name = hr_names[i]
            cubic_psnr,out_psnr = WDSR_PSNRavg(lr_name,hr_name,opt.test_directory_path,opt.label_directory_path,wdsr_model)
            cubic_avg.append(cubic_psnr)
            wdsr_avg.append(out_psnr)
    print('END.....')
    cubic_avg = np.array(cubic_avg)
    wdsr_avg = np.array(wdsr_avg)
    print('cubic_avg:',cubic_avg.mean())
    print('wdsr_avg:',wdsr_avg.mean())

main()