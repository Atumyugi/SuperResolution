import torch
import h5py
import numpy as np
from torch import nn,optim
import torch.utils.data as dataf
from model import edsr
from model import wdsr_b
from model import my
import skimage.measure as measure
import argparse, os
from tensorboardX import SummaryWriter
from tqdm import tqdm
#os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
from WDSR import common_utils
from WDSR import config
parser = config.get_args()


writer = SummaryWriter('log/flip_image')

def training(loader,model,lr,lossfunction,optimizer,epoches,savepath):

    loss_mat = []
    lr_names = os.listdir(opt.test_directory_path)
    lr_names = sorted(lr_names)  # 排序
    hr_names = os.listdir(opt.label_directory_path)
    hr_names = sorted(hr_names)
    nums = lr_names.__len__()
    model.cuda()
    model = nn.DataParallel(model, device_ids=[0,1])
    lossfunction.cuda()
    total_step = len(loader)
    print(total_step)
    for epoch in range(epoches):
        lr = common_utils.adjust_learning_rate(epoch=epoch, lr=lr)
        # lr = lr
        print('Epoch : {}'.format(epoch + 1))
        print('*' * 10)
        running_loss = 0.0

        for i, data in enumerate(loader, 1):
            lr_img, hr_img = data
            # lr:[16,3,12,12] lr:[16,3,48,48]



            lr_img = torch.tensor(lr_img, requires_grad=True,dtype=torch.float32)

            lr_img = lr_img.cuda()
            hr_img = hr_img.cuda().type(torch.float32)
            # forward
            optimizer.zero_grad()


            out = model(lr_img)
            mse_loss = lossfunction(out, hr_img)
            running_loss += mse_loss.item()


            mse_loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print('enum:{} Loss:{:.6f}'.format(
                    i, mse_loss
                ))

            writer.add_scalar('Train/mes_loss', mse_loss.item(), epoch * total_step + i)

        if epoch % 10 == 0:
            model.eval()
            print('eval-----验证图像数:', nums, '个')
            wdsr_avg = []
            for i in tqdm(range(nums)):
                #print('第', i, '/', nums, '个图像:', '\r', end='')
                lr_name = lr_names[i]
                hr_name = hr_names[i]
                out_psnr = common_utils.WDSR_eval(lr_name, hr_name, opt.test_directory_path,
                                     opt.label_directory_path,model)
                wdsr_avg.append(out_psnr)
            wdsr_avg = np.array(wdsr_avg)
            print('avg__wdsr:', wdsr_avg.mean())
            if wdsr_avg.mean() > opt.psnr_max :
                torch.save(model, 'wdsr_max_psnr.pkl')
        if epoch % 20 == 0:
            print('Save model for ',epoch,'epoch')
            torch.save(model,opt.check_point_path+str(epoch)+'.pkl')

        print('Finish {} Epoch, Loss: {:.6f}'.format(
            epoch + 1, running_loss
        ))
        writer.add_scalar('Train/running_loss', running_loss, epoch)
    torch.save(model, savepath)
    print('Save Success......')

def main():

    global opt
    opt = parser.parse_args()
    print(opt)
    loader = common_utils.load_dataset(opt.file,opt.batchSize)

    if opt.model == 'MY':
        model = my.MY()
    elif opt.model == 'WDSR':
        model = wdsr_b.MODEL()

    lossfunction = nn.L1Loss()
    optimizer = optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),
                            lr=opt.learning_rate,weight_decay=0,
                           betas = (0.9, 0.999),eps=1e-08)  #（0.9，0.99） （0.01，0.1）

    training(loader=loader,model=model,lr=opt.learning_rate,lossfunction=lossfunction,optimizer=optimizer,epoches=opt.epoches,savepath=opt.savepath)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

main()