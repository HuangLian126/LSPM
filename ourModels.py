import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from vgg import VGG

class LSPMLayer(Module):
    def __init__(self, in_planes, S):
        super(LSPMLayer, self).__init__()

        self.in_planes = in_planes
        self.S = S

        self.GAP      = nn.AdaptiveAvgPool2d(self.S)
        self.GAP_1X1  = nn.Conv2d(self.in_planes, self.in_planes, 1, 1, bias=False)
        self.GAP_Rule = nn.ReLU(inplace=True)
        self.conv     = nn.Conv2d(in_channels=in_planes, out_channels=S*S, kernel_size=1, bias=False)

    def forward(self, x):

        B, C, H, W = x.size()

        x_reshape_T = x.view(B, -1, H*W).permute(0, 2, 1)           # B,HW,C
        x_reshape   = x.view(B, -1, H*W)                            # B,C,HW
        MM1         = torch.bmm(x_reshape_T, x_reshape)             # B, HW, HW
        MM1         = F.softmax(MM1, dim=-1)                        # B,HW, HW

        x_conv = self.conv(x).view(B, -1, H*W)                      # B, SS, HW
        MM2    = torch.bmm(x_conv, MM1)                             # B, SS, HW

        GAP = self.GAP(x)
        GAP = self.GAP_1X1(GAP)
        GAP = self.GAP_Rule(GAP)
        GAP = GAP.view(B, -1, self.S*self.S)                          # B, C, SS

        MM3     = torch.bmm(GAP, MM2).view(B, C, H, W)               # B, C, H, W
        results = torch.add(MM3, x)

        return results

class LSPM(Module):
    def __init__(self, in_planes):
        super(LSPM, self).__init__()
        self.in_planes = in_planes
        self.LSPM_1 = LSPMLayer(self.in_planes, 1)
        self.LSPM_2 = LSPMLayer(self.in_planes, 2)
        self.LSPM_3 = LSPMLayer(self.in_planes, 3)
        self.LSPM_6 = LSPMLayer(self.in_planes, 6)

        self.conv  = nn.Conv2d(5*self.in_planes, self.in_planes, 1, 1, bias=False)

    def forward(self, x):
        '''
        inputs:
        x:  input feature maps(B, C, H, W)
        returns:
        out:  B, C, H, W
        '''

        LSPM_1 = self.LSPM_1(x)
        LSPM_2 = self.LSPM_2(x)
        LSPM_3 = self.LSPM_3(x)
        LSPM_6 = self.LSPM_6(x)
        out = torch.cat([x, LSPM_1, LSPM_2, LSPM_3, LSPM_6], dim=1)
        out = self.conv(out)
        return out

class FAMCA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FAMCA, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Conv2d(3*out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu_cat = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels//4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels//4, out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, left, down, right):
        # left1,    down 2,   right 3

        down = F.interpolate(down, size=left.size()[2], mode='bilinear', align_corners=True)
        down = self.relu(self.conv(down))

        merge = torch.cat((left, down, right), dim=1)
        merge = self.relu_cat(self.conv_cat(merge))

        b, c, _, _ = merge.size()
        y = self.avg_pool(merge).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        out = torch.mul(y, merge)

        return out

class FAMCA_Single(nn.Module):
    def __init__(self, channels):
        super(FAMCA_Single, self).__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(3*channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//4, channels, bias=False),
            nn.Sigmoid()
        )
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.relu(self.conv(self.upsample_2(x)))

        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = torch.mul(y, x)

        return out

class model_VGG(nn.Module):
    def __init__(self, channel=32):
        super(model_VGG, self).__init__()
        self.vgg = VGG()

        self.score = nn.Conv2d(128, 1, 1, 1)

        self.lspm = LSPM(512)

        self.aggregation_4 = FAMCA(512, 512)
        self.aggregation_3 = FAMCA(512, 256)
        self.aggregation_2 = FAMCA(256, 128)
        self.aggregation_1 = FAMCA_Single(128)

        self.out_planes = [512, 256, 128]
        infos = []
        for ii in self.out_planes:
            infos.append(nn.Sequential(nn.Conv2d(512, ii, 3, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.infos = nn.ModuleList(infos)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):

        x1 = self.vgg.conv1(x)
        x2 = self.vgg.conv2(x1)
        x3 = self.vgg.conv3(x2)
        x4 = self.vgg.conv4(x3)
        x5 = self.vgg.conv5(x4)

        lspm = self.lspm(x5)
        GG = []
        GG.append(self.infos[0](self.upsample_2(lspm)))
        GG.append(self.infos[1](self.upsample_4(lspm)))
        GG.append(self.infos[2](self.upsample_8(lspm)))

        merge  = self.aggregation_4(x4, x5,    GG[0])
        merge  = self.aggregation_3(x3, merge, GG[1])
        merge  = self.aggregation_2(x2, merge, GG[2])
        merge  = self.aggregation_1(merge)
        merge  = self.score(merge)
        result = F.interpolate(merge, x1.size()[2], mode='bilinear', align_corners=True)

        return result
