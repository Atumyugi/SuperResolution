import torch
import numpy as np
import argparse,os
from WDSR import common_utils
from WDSR import config_my
from tqdm import tqdm
parser = config_my.get_args()


def main():
    global opt
    opt = parser.parse_args()
    print(opt)
    wdsr_model = torch.load(opt.model_path)
    wdsr_model.cuda()

    lr_names = os.listdir(opt.test_directory_path)
    lr_names = sorted(lr_names)  # 排序
    if opt.label_directory_path != None:
        hr_names = os.listdir(opt.label_directory_path)
        hr_names = sorted(hr_names)
    nums = lr_names.__len__()  # 训练图片的数量

    cubic_avg = []
    wdsr_avg = []
    print('Test Type: ',opt.test_type)
    if opt.test_type == 'A':
        print('Images need to be handle:', nums)
        for i in range(nums):
            print('The',i,'/',nums,'image:')
            lr_name = lr_names[i]
            hr_name = hr_names[i]
            cubic_psnr,out_psnr = common_utils.WDSR_super_resolution(lr_name,hr_name,opt.test_directory_path,opt.label_directory_path,opt.txt_path,opt.save_path,wdsr_model)
            cubic_avg.append(cubic_psnr)
            wdsr_avg.append(out_psnr)
        print('END.....')
        cubic_avg = np.array(cubic_avg)
        wdsr_avg = np.array(wdsr_avg)
        print('cubic_avg:', cubic_avg.mean())
        print('wdsr_avg:', wdsr_avg.mean())
    elif opt.test_type == 'B':
        for i in tqdm(range(nums)):
            lr_name = lr_names[i]
            hr_name = hr_names[i]
            cubic_psnr,out_psnr = common_utils.WDSR_PSNRavg(lr_name,hr_name,opt.test_directory_path,opt.label_directory_path,wdsr_model)
            cubic_avg.append(cubic_psnr)
            wdsr_avg.append(out_psnr)
        print('END.....')
        cubic_avg = np.array(cubic_avg)
        wdsr_avg = np.array(wdsr_avg)
        print('cubic_avg:', cubic_avg.mean())
        print('wdsr_avg:', wdsr_avg.mean())
    elif opt.test_type == 'C':
        print('Your Type have not hr path, make sure opt.label_directory_path is None ')
        for i in tqdm(range(nums)):
            lr_name = lr_names[i]
            common_utils.WDSR_create_srimg(lr_name,opt.test_directory_path,opt.save_path,wdsr_model)
        print('END.....')

main()