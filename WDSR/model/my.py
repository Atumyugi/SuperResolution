## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
from model import common
import torch
import torch.nn as nn

wn = lambda x: torch.nn.utils.weight_norm(x)
def make_model(args, parent=False):
    return MY(args)

## WDSR block
class Block(nn.Module):
    def __init__(
        self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(Block, self).__init__()
        self.res_scale = res_scale
        body = []
        expand = 6
        linear = 0.8
        body.append(
            wn(nn.Conv2d(n_feats, n_feats*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(n_feats*expand, int(n_feats*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv2d(int(n_feats*linear), n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1,wn=wn):

        super(RCAB, self).__init__()
        body = []
        expand = 6
        linear = 0.8
        body.append(
            wn(nn.Conv2d(n_feat, n_feat * expand, 1, padding=1 // 2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(n_feat * expand, int(n_feat * linear), 1, padding=1 // 2)))
        body.append(
            wn(nn.Conv2d(int(n_feat * linear), n_feat, kernel_size, padding=kernel_size // 2)))

        body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1,wn=wn) \
            for _ in range(n_resblocks)]
        modules_body.append(wn(conv(n_feat, n_feat, kernel_size)))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Channel Attention Network (RCAN)
class MY(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(MY, self).__init__()

        n_resgroups = 10
        n_resblocks = 20
        n_feats = 64
        kernel_size = 3
        reduction = 16
        scale = 4
        act = nn.ReLU(True)
        n_colors = 3
        res_scale = 0.1
        wn = lambda x: torch.nn.utils.weight_norm(x)
        # RGB mean for DIV2K
        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
            [0.4488, 0.4371, 0.4040])).view([1, 3, 1, 1])
        # self.sub_mean = common.MeanShift(args.rgb_range)
        head = []
        head.append(
            wn(nn.Conv2d(3, n_feats, 3, padding=3 // 2)))

        # define body module
        body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        #body.append(wn(conv(n_feats, n_feats, kernel_size)))


        # define tail module
        tail = []
        out_feats = scale * scale * 3
        tail.append(
            wn(nn.Conv2d(n_feats, out_feats, 3, padding=3 // 2)))
        tail.append(nn.PixelShuffle(scale))

        skip = []
        skip.append(
            wn(nn.Conv2d(3, out_feats, 5, padding=5 // 2))
        )
        skip.append(nn.PixelShuffle(scale))

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)


    def forward(self, x):
        x = (x - self.rgb_mean.cuda() * 255) / 127.5
        # x = self.sub_mean(x)
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        x = x * 127.5 + self.rgb_mean.cuda() * 255

        return x

