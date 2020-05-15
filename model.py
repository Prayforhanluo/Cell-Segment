# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:14:03 2019

@author: LuoHan
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    
    def __init__(self, inplanes, planes):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=1, padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes, kernel_size=3,
                          stride=1, padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConvBlock(nn.Module):
    
    def __init__(self, inplanes, planes):
        super(UpConvBlock, self).__init__()
        
        self.upconv = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=1, padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)
                )
    
    def forward(self, x):
        out = self.upconv(x)
        return out

class UNet(nn.Module):
    
    def __init__(self, inplanes=1, planes=1, base=16):
        super(UNet, self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        c = [base *(2**i) for i in range(0, 6)]
        
        self.conv1 = ConvBlock(inplanes, c[0])
        self.conv2 = ConvBlock(c[0], c[1])
        self.conv3 = ConvBlock(c[1], c[2])
        self.conv4 = ConvBlock(c[2], c[3])
        self.conv5 = ConvBlock(c[3], c[4])
        self.conv6 = ConvBlock(c[4], c[5])
        
        self.up6 = UpConvBlock(c[5], c[4])
        self.upconv6 = ConvBlock(c[5], c[4])
        
        self.up5 = UpConvBlock(c[4], c[3])
        self.upconv5 = ConvBlock(c[4], c[3])
        
        self.up4 = UpConvBlock(c[3], c[2])
        self.upconv4 = ConvBlock(c[3], c[2])
        
        self.up3 = UpConvBlock(c[2], c[1])
        self.upconv3 = ConvBlock(c[2], c[1])
        
        self.up2 = UpConvBlock(c[1], c[0])
        self.upconv2 = ConvBlock(c[1], c[0])
        
        self.fconv = nn.Conv2d(c[0], planes, kernel_size=1,
                               stride=1, padding=0)
        
        
    def forward(self, x):
        
        ##Encoder part
        x1 = self.conv1(x)
        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)
        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)
        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)
        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)
        x6 = self.maxpool(x5)
        x6 = self.conv6(x6)
        
        ##Decoder part
        d6 = self.up6(x6)
        d6 = torch.cat((x5, d6), dim=1)
        d6 = self.upconv6(d6)

        d5 = self.up5(d6)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.upconv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.upconv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.upconv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.upconv2(d2)

        d1 = self.fconv(d2)
        return d1

        