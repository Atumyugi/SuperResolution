import torch
import numpy as np
import torch.nn as nn
import torchvision

class BOX(nn.Module):
    def __init__(self):
        super(BOX,self).__init__()

        self.box = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1,16,31,1,15),

            ) for _ in range(24)
        ])
        self.lstm = nn.LSTM(16, 64, 1, batch_first=True)
        self.classifier = nn.Linear(64, 9)

        self.trans = nn.Linear(3000,1)
    def forward(self, x):

        out_i = torch.zeros((50,16,24,1))
        #out_i = out_i.cuda()
        for j in range(24):
            res = self.box[j](x[:, :, j, :])
            res = self.trans(res)
            out_i[:, :, j, :] = res
        out_i.squeeze_(-1)
        out_i = out_i.permute(0, 2, 1)
        out_i, _ = self.lstm(out_i)
        out_i = out_i[:, -1, :]
        out_i = self.classifier(out_i)
        return out_i

class BigBox(nn.Module):
    def __init__(self):
        super(BigBox,self).__init__()
        self.bigbox = nn.ModuleList([
            nn.Sequential( BOX() ) for _ in range(12)
        ])
    def forward(self, x):
        out = torch.zeros((50,12,9))
        #out = out.cuda()
        for i in range(12):
            out_i = self.bigbox[i](x[:,i:i+1,:,:])
            out[:,i,:] = out_i
        return out



net = BigBox()
#net.cuda()
x = torch.randn((50,12,24,3000))
x= torch.tensor(x, requires_grad=True)

#x = x.cuda()
out = net(x)
print(out.shape)

