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
#from EDSR  import edsr
from model import wdsr_b
import skimage.measure as measure
import argparse, os
#from tensorboardX import SummaryWriter
from tqdm import tqdm

def predict(input_x,wdsr_model):
    input_x = torch.from_numpy(input_x)
    input_x = torch.tensor(input_x,requires_grad=True,dtype = torch.float)
    input_x = input_x.cuda()
    wdsr_out =  wdsr_model(input_x)

    wdsr_out = wdsr_out.cpu().data[0].numpy()
    wdsr_out = np.transpose(wdsr_out,[1,2,0])
    #print(wdsr_out.shape)
    #edsr_out = edsr_out/255
    return wdsr_out

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def SSIM(im1,im2,imrange = 255):
    ssim = measure.compare_ssim(im1,im2,data_range=imrange,multichannel=True)
    return ssim



def WDSR_eval(testimg_path,labelimg_path,test_directory_path,label_directory_path,wdsr_model):


    lr_path = test_directory_path + testimg_path
    hr_path = label_directory_path + labelimg_path
    #print(lr_path,hr_path)
    img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
    #print(testimg_path,' shape: ',img.shape)
    source_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
    #source_img_y = cv2.cvtColor(source_img, cv2.COLOR_RGB2YCrCb)
    #source_img_y = source_img_y[:, :, 0]
    height, weight, _ = img.shape
    sr_height, sr_weight = height * 4, weight * 4

    #img_cubic = cv2.resize(img, (sr_weight, sr_height), cv2.INTER_CUBIC)
    #cv2.imwrite(save_path+'Cubic_'+testimg_path, img_cubic)

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

    out_left_top = predict(input_x_left_top,wdsr_model)
    out_left_mid_top = predict(input_x_left_mid_top,wdsr_model)
    out_right_mid_top = predict(input_x_right_mid_top,wdsr_model)
    out_right_top = predict(input_x_right_top,wdsr_model)
    out_left_bot = predict(input_x_left_bot,wdsr_model)
    out_left_mid_bot = predict(input_x_left_mid_bot,wdsr_model)
    out_right_mid_bot = predict(input_x_right_mid_bot,wdsr_model)
    out_right_bot = predict(input_x_right_bot,wdsr_model)
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

    #cv2.imwrite(save_path+'WDSR_'+testimg_path, out)

    #out_y = cv2.cvtColor(out, cv2.COLOR_RGB2YUV)
    #out_y = out_y[:, :, 0]
    #print(testimg_path,'finished......')
    return PSNR(source_img, out)

def adjust_learning_rate(epoch,lr):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = lr * (0.5 ** (epoch // 50))  # init 0.1 epoch 100
    return lr

def load_dataset(file,batchSize):
    with h5py.File(file,'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        print(data.shape)
        print(label.shape)
    train_data = torch.from_numpy(data)
    train_data = torch.tensor(train_data)
    train_label = torch.from_numpy(label)
    train_label = torch.tensor(train_label)
    print(train_data.shape,train_label.shape)
    dataset = dataf.TensorDataset(train_data,train_label)
    loader = dataf.DataLoader(dataset,batch_size=batchSize,shuffle=True)

    return loader