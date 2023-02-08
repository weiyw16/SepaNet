# Machine learning based P- and S- wave separation
#              Author: Yanwen Wei
#              Email: wei_yanwen@163.com
#              Date: 2021-12-28
#
import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import Sampler

class FCN_simple(nn.Module):
    def __init__(self, nph, nop):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(nph, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1)

        self.classifier1p = nn.Conv2d(512, nop, kernel_size=1)
        # self.classifier2p = nn.Conv2d(32, 32, kernel_size=1)
        # self.classifier3p = nn.Conv2d(32, 32, kernel_size=1)
        # self.classifier4p = nn.Conv2d(32, nop, kernel_size=1)

        self.classifier1s = nn.Conv2d(512, nop, kernel_size=1)
        # self.classifier2s = nn.Conv2d(32, 32, kernel_size=1)
        # self.classifier3s = nn.Conv2d(32, 32, kernel_size=1)
        # self.classifier4s = nn.Conv2d(32, nop, kernel_size=1)

        # self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # self.pad = nn.ReflectionPad2d(1)
        # self.deconv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        # self.deconv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        # self.deconv3 = nn.Conv2d(256, 128, kernel_size=(5, 3), stride=1, padding=(1, 0))
        # self.deconv4 = nn.Conv2d(128, 64, kernel_size=(5, 3), stride=1, padding=(1, 0))
        # self.deconv5 = nn.Conv2d(64, 32, kernel_size=(5, 3), stride=1, padding=(1, 0))

        # self.bn1 = nn.BatchNorm2d(512)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.bn31 = nn.BatchNorm2d(128)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.bn41 = nn.BatchNorm2d(64)
        # self.bn5 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        # x1 = x
        x = self.relu(self.conv2(x))
        # x2 = x
        x = self.relu(self.conv3(x))
        # x3 = x
        x = self.relu(self.conv4(x))
        # x4 = x
        x = self.relu(self.conv5(x))
        # x5 = x

        # score = self.relu(self.deconv1(self.pad(self.upsample1(x5))))  # size=(N, 512, x.H/16, x.W/16)
        # score = self.bn1(score + x4)  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        # score = self.relu(self.deconv2(self.pad(self.upsample2(score))))  # size=(N, 256, x.H/8, x.W/8)
        # score = self.bn2(score + x3)  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        # score = self.bn3(self.relu(self.deconv3(self.pad(self.upsample3(score)))))  # size=(N, 128, x.H/4, x.W/4)
        # score = self.bn31(score + x2)
        # score = self.bn4(self.relu(self.deconv4(self.pad(self.upsample4(score)))))  # size=(N, 64, x.H/2, x.W/2)
        # score = self.bn41(score + x1)
        # score = self.bn5(self.relu(self.deconv5(self.pad(self.upsample5(score)))))  # size=(N, 32, x.H, x.W)
        # score1 = self.classifier4p(
        #     self.classifier3p(self.classifier2p(self.classifier1p(score))))  # size=(N, n_class, x.H/1, x.W/1)
        # score2 = self.classifier4s(self.classifier3s(self.classifier2s(self.classifier1s(score))))
        score1 = self.classifier1p(x)
        score2 = self.classifier1s(x)
        return score1, score2  # size=(N, n_class, x.H/1, x.W/1)

def lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch + 1 - 1000 ) / float(1000 + 1)
    #lr_l = 1.0 - max(0, epoch + 1 - 100 ) / float(100 + 1)
    return lr_l

# <editor-fold ...>
class FCN8s_conv_upsample_out2_v2_k5_c5(nn.Module):
    def __init__(self, nph, nop):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(nph, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(5, 3), stride=2, padding=(2, 1), dilation=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(5, 3), stride=2, padding=(2, 1), dilation=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(5, 3), stride=2, padding=(2, 1), dilation=1)

        self.classifier1p = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier2p = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier3p = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier4p = nn.Conv2d(32, nop, kernel_size=1)

        self.classifier1s = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier2s = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier3s = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier4s = nn.Conv2d(32, nop, kernel_size=1)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.pad = nn.ReflectionPad2d(1)
        self.deconv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.deconv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.deconv3 = nn.Conv2d(256, 128, kernel_size=(5, 3), stride=1, padding=(1, 0))
        self.deconv4 = nn.Conv2d(128, 64, kernel_size=(5, 3), stride=1, padding=(1, 0))
        self.deconv5 = nn.Conv2d(64, 32, kernel_size=(5, 3), stride=1, padding=(1, 0))

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn31 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn41 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x1 = x
        x = self.relu(self.conv2(x))
        x2 = x
        x = self.relu(self.conv3(x))
        x3 = x  # size=(N, 256, x.H/8,  x.W/8)

        x = self.relu(self.conv4(x))
        x4 = x  # size=(N, 512, x.H/16, x.W/16)
        x = self.relu(self.conv5(x))
        x5 = x  # size=(N, 512, x.H/32, x.W/32)

        score = self.relu(self.deconv1(self.pad(self.upsample1(x5))))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(self.pad(self.upsample2(score))))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(self.pad(self.upsample3(score)))))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn31(score + x2)
        score = self.bn4(self.relu(self.deconv4(self.pad(self.upsample4(score)))))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn41(score + x1)
        score = self.bn5(self.relu(self.deconv5(self.pad(self.upsample5(score)))))  # size=(N, 32, x.H, x.W)
        score1 = self.classifier4p(
            self.classifier3p(self.classifier2p(self.classifier1p(score))))  # size=(N, n_class, x.H/1, x.W/1)
        score2 = self.classifier4s(self.classifier3s(self.classifier2s(self.classifier1s(score))))

        return score1, score2  # size=(N, n_class, x.H/1, x.W/1)
# </editor-fold ...>

class FCN8s_conv_upsample_out2_v2(nn.Module):
    # class FCN8s_conv_batch(nn.Module):
    # class FCN8s_pool_nobatch(nn.Module):

    def __init__(self, nph, nop):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(nph, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        #         self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        #         self.bn1     = nn.BatchNorm2d(512)
        #         self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        #         self.bn2     = nn.BatchNorm2d(256)
        #         self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        #         self.bn3     = nn.BatchNorm2d(128)
        #         self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        #         self.bn4     = nn.BatchNorm2d(64)
        #         self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        #         self.bn5     = nn.BatchNorm2d(32)
        self.classifier1p = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier2p = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier3p = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier4p = nn.Conv2d(32, nop, kernel_size=1)

        self.classifier1s = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier2s = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier3s = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier4s = nn.Conv2d(32, nop, kernel_size=1)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.pad = nn.ReflectionPad2d(1)
        self.deconv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.deconv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.deconv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.deconv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0)
        self.deconv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x3 = x  # size=(N, 256, x.H/8,  x.W/8)

        x = self.relu(self.conv4(x))
        x4 = x  # size=(N, 512, x.H/16, x.W/16)
        x = self.relu(self.conv5(x))
        x5 = x  # size=(N, 512, x.H/32, x.W/32)

        score = self.relu(self.deconv1(self.pad(self.upsample1(x5))))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(self.pad(self.upsample2(score))))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(self.pad(self.upsample3(score)))))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(self.pad(self.upsample4(score)))))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(self.pad(self.upsample5(score)))))  # size=(N, 32, x.H, x.W)
        score1 = self.classifier4p(
            self.classifier3p(self.classifier2p(self.classifier1p(score))))  # size=(N, n_class, x.H/1, x.W/1)
        score2 = self.classifier4s(self.classifier3s(self.classifier2s(self.classifier1s(score))))

        return score1, score2  # size=(N, n_class, x.H/1, x.W/1)


class FCN8s_conv_upsample_out2_v2_k5(nn.Module):
    # class FCN8s_conv_batch(nn.Module):
    # class FCN8s_pool_nobatch(nn.Module):

    def __init__(self, nph, nop):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(nph, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(5, 3), stride=2, padding=(2, 1), dilation=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(5, 3), stride=2, padding=(2, 1), dilation=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(5, 3), stride=2, padding=(2, 1), dilation=1)
        #         self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        #         self.bn1     = nn.BatchNorm2d(512)
        #         self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        #         self.bn2     = nn.BatchNorm2d(256)
        #         self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        #         self.bn3     = nn.BatchNorm2d(128)
        #         self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        #         self.bn4     = nn.BatchNorm2d(64)
        #         self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        #         self.bn5     = nn.BatchNorm2d(32)
        self.classifier1p = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier2p = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier3p = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier4p = nn.Conv2d(32, nop, kernel_size=1)

        self.classifier1s = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier2s = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier3s = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier4s = nn.Conv2d(32, nop, kernel_size=1)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.pad = nn.ReflectionPad2d(1)
        self.deconv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.deconv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.deconv3 = nn.Conv2d(256, 128, kernel_size=(5, 3), stride=1, padding=(1, 0))
        self.deconv4 = nn.Conv2d(128, 64, kernel_size=(5, 3), stride=1, padding=(1, 0))
        self.deconv5 = nn.Conv2d(64, 32, kernel_size=(5, 3), stride=1, padding=(1, 0))

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x3 = x  # size=(N, 256, x.H/8,  x.W/8)

        x = self.relu(self.conv4(x))
        x4 = x  # size=(N, 512, x.H/16, x.W/16)
        x = self.relu(self.conv5(x))
        x5 = x  # size=(N, 512, x.H/32, x.W/32)

        score = self.relu(self.deconv1(self.pad(self.upsample1(x5))))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(self.pad(self.upsample2(score))))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(self.pad(self.upsample3(score)))))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(self.pad(self.upsample4(score)))))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(self.pad(self.upsample5(score)))))  # size=(N, 32, x.H, x.W)
        score1 = self.classifier4p(
            self.classifier3p(self.classifier2p(self.classifier1p(score))))  # size=(N, n_class, x.H/1, x.W/1)
        score2 = self.classifier4s(self.classifier3s(self.classifier2s(self.classifier1s(score))))

        return score1, score2  # size=(N, n_class, x.H/1, x.W/1)


class FCN8s_conv_upsample_out2_v2_k5_c5_hook(nn.Module):
    def __init__(self, nph, nop):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(nph, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(5, 3), stride=2, padding=(2, 1), dilation=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(5, 3), stride=2, padding=(2, 1), dilation=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(5, 3), stride=2, padding=(2, 1), dilation=1)

        self.classifier1p = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier2p = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier3p = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier4p = nn.Conv2d(32, nop, kernel_size=1)

        self.classifier1s = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier2s = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier3s = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier4s = nn.Conv2d(32, nop, kernel_size=1)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.pad = nn.ReflectionPad2d(1)
        self.deconv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.deconv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.deconv3 = nn.Conv2d(256, 128, kernel_size=(5, 3), stride=1, padding=(1, 0))
        self.deconv4 = nn.Conv2d(128, 64, kernel_size=(5, 3), stride=1, padding=(1, 0))
        self.deconv5 = nn.Conv2d(64, 32, kernel_size=(5, 3), stride=1, padding=(1, 0))

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn31 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn41 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)


    def forward(self, x, whichlayer=None, midlayer=None):

        if whichlayer == 1:
            x = midlayer
        x = self.relu(self.conv1(x))
        x1 = x
        if whichlayer == 2:
            x = midlayer
        x = self.relu(self.conv2(x))
        x2 = x
        if whichlayer == 3:
            x = midlayer
        x = self.relu(self.conv3(x))
        x3 = x  # size=(N, 256, x.H/8,  x.W/8)

        if whichlayer == 4:
            x = midlayer
        x = self.relu(self.conv4(x))
        x4 = x  # size=(N, 512, x.H/16, x.W/16)
        if whichlayer == 5:
            x = midlayer
        x = self.relu(self.conv5(x))
        x5 = x  # size=(N, 512, x.H/32, x.W/32)

        if whichlayer == 6:
            x5 = midlayer
        score = self.relu(self.deconv1(self.pad(self.upsample1(x5))))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        if whichlayer == 7:
            score = midlayer
        score = self.relu(self.deconv2(self.pad(self.upsample2(score))))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        if whichlayer == 8:
            score = midlayer
        score = self.bn3(self.relu(self.deconv3(self.pad(self.upsample3(score)))))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn31(score + x2)
        if whichlayer == 9:
            score = midlayer
        score = self.bn4(self.relu(self.deconv4(self.pad(self.upsample4(score)))))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn41(score + x1)
        if whichlayer == 10:
            score = midlayer
        score = self.bn5(self.relu(self.deconv5(self.pad(self.upsample5(score)))))  # size=(N, 32, x.H, x.W)

        #P
        if whichlayer == 11:
            score = midlayer
        score1_1 = self.classifier1p(score)
        score2_1 = self.classifier1s(score)
        if whichlayer == 12:
            score1_1 = midlayer
        score1_2 = self.classifier2p(score1_1)
        if whichlayer == 13:
            score1_2 = midlayer
        score1_3 = self.classifier3p(score1_2)
        if whichlayer == 14:
            score1_3 = midlayer
        score1 = self.classifier4p(score1_3)

        #S
        if whichlayer == 15:
            score2_1 = midlayer
        score2_2 = self.classifier2p(score2_1)
        if whichlayer == 16:
            score2_2 = midlayer
        score2_3 = self.classifier3p(score2_2)
        if whichlayer == 17:
            score2_3 = midlayer
        score2 = self.classifier4p(score2_3)

        # score1 = self.classifier4p(
        #     self.classifier3p(self.classifier2p(self.classifier1p(score))))  # size=(N, n_class, x.H/1, x.W/1)
        # score2 = self.classifier4s(self.classifier3s(self.classifier2s(self.classifier1s(score))))

        return score1, score2  # size=(N, n_class, x.H/1, x.W/1)

class FCN8s_conv_upsample_out2_v2_k5_c5_addhook(nn.Module):
    def __init__(self, nph, nop):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(nph, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(5, 3), stride=2, padding=(2, 1), dilation=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(5, 3), stride=2, padding=(2, 1), dilation=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(5, 3), stride=2, padding=(2, 1), dilation=1)

        self.classifier1p = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier2p = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier3p = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier4p = nn.Conv2d(32, nop, kernel_size=1)

        self.classifier1s = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier2s = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier3s = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier4s = nn.Conv2d(32, nop, kernel_size=1)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.pad = nn.ReflectionPad2d(1)
        self.deconv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.deconv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.deconv3 = nn.Conv2d(256, 128, kernel_size=(5, 3), stride=1, padding=(1, 0))
        self.deconv4 = nn.Conv2d(128, 64, kernel_size=(5, 3), stride=1, padding=(1, 0))
        self.deconv5 = nn.Conv2d(64, 32, kernel_size=(5, 3), stride=1, padding=(1, 0))

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn31 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn41 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)

    def forward(self, x, x1):
        x = self.relu(self.conv1(x))
        # x1 = x
        x = self.relu(self.conv2(x))
        x2 = x
        x = self.relu(self.conv3(x))
        x3 = x  # size=(N, 256, x.H/8,  x.W/8)

        x = self.relu(self.conv4(x))
        x4 = x  # size=(N, 512, x.H/16, x.W/16)
        x = self.relu(self.conv5(x))
        x5 = x  # size=(N, 512, x.H/32, x.W/32)

        score = self.relu(self.deconv1(self.pad(self.upsample1(x5))))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(self.pad(self.upsample2(score))))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(self.pad(self.upsample3(score)))))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn31(score + x2)
        score = self.bn4(self.relu(self.deconv4(self.pad(self.upsample4(score)))))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn41(score + x1)
        score = self.bn5(self.relu(self.deconv5(self.pad(self.upsample5(score)))))  # size=(N, 32, x.H, x.W)
        score1 = self.classifier4p(
            self.classifier3p(self.classifier2p(self.classifier1p(score))))  # size=(N, n_class, x.H/1, x.W/1)
        score2 = self.classifier4s(self.classifier3s(self.classifier2s(self.classifier1s(score))))

        return score1, score2  # size=(N, n_class, x.H/1, x.W/1)


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        # net = torch.nn.parallel.DistributedDataParallel(net, gpu_ids)
        # net.to(gpu_ids[0])
    init_weights(net, init_type, gain=init_gain)
    return net



class myRandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """
    def __init__(self, data_source, replacement=False, seed=None, num_samples=None):

        self.data_source = data_source
        # 这个参数控制的应该为是否重复采样
        self.replacement = replacement
        self._num_samples = num_samples
        self.seed = seed
    # 省略类型检查
    @property
    def num_samples(self):
        # dataset size might change at runtime
        # 初始化时不传入num_samples的时候使用数据源的长度
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples
    # 返回数据集长度
    def __len__(self):
        return self.num_samples
    def __iter__(self):

        # torch.manual_seed(self.seed)
        n = len(self.data_source)
        if self.replacement:
         # 生成的随机数是可能重复的
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
    # 生成的随机数是不重复的
        return iter(torch.randperm(n).tolist())