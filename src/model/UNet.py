# https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, bilinear=True):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.bilinear = bilinear
        self.complexity = 45

        self.inc = DoubleConv(input_channels, )
        self.down1 = Down(, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, output_channels)

    def set_model()


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits