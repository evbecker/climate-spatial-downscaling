import math

import torch
from torch import nn
import torch.nn.functional as F

from model.blocks import ConstantInput, LFF, StyledConv, ToRGB, PixelNorm, EqualLinear, StyledResBlock, ResBlock, ToRGBNoMod, EqualConv2d, StyledConvNoNoise, ConvLayer, Self_Attn, ConstantInputPatch

class Conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,upsample=False,downsample=False):
        super(Conv,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)
        self.relu=nn.LeakyReLU(0.2)
        self.sample=False
        if upsample:
            self.sample=True
            self.samp=nn.UpsamplingBilinear2d(scale_factor=2)
        if downsample:
            self.sample=True
            self.samp=nn.MaxPool2d(stride=2,kernel_size=2)
    def forward(self,x):
        if self.sample:
            x=self.samp(x)
        x=self.conv(x)
        return self.relu(x)

class AutoEncoder(nn.Module):
    def __init__(self, size1=256,size2=256,style_dim=512,
                 activation=None, linear_size = 512, channel_multiplier=2,output_channels=1, **kwargs):
        super(AutoEncoder, self).__init__()

        self.size1 = size1
        self.size2=size2
        demodulate = True
        self.demodulate = demodulate


        self.channels = {
            0: linear_size,
            1: linear_size,
            2: linear_size,
            3: linear_size,
            4: int(linear_size/2) * channel_multiplier,
            5: int(linear_size/4) * channel_multiplier,
            6: int(linear_size/8) * channel_multiplier,
            7: int(linear_size/8) * channel_multiplier,
            8: int(linear_size/16) * channel_multiplier,
        }

        in_channels = int(self.channels[0])
        #self.project1 = ConvLayer(output_channels, in_channels, 3, downsample = True)
        self.project1=Conv(output_channels,in_channels,3,stride=1,padding=1,downsample=True)
        #self.project2 = ConvLayer(in_channels, in_channels, 3, downsample = True)
        self.project2=Conv(in_channels,in_channels,3,stride=1,padding=1,downsample=True)
        self.att = Self_Attn(in_channels, "relu")
        #self.up_project1 = ConvLayer(in_channels, in_channels, 3, upsample = True)
        #self.up_project2 = ConvLayer(in_channels, in_channels, 3, upsample = True)
        self.up_project1=Conv(in_channels,in_channels,3,stride=1,padding=1,upsample=True)
        self.up_project2=Conv(in_channels,in_channels,3,stride=1,padding=1,upsample=True)


        self.linears = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.log_size = min(int(math.log(size1, 2)),int(math.log(size2, 2)))

        self.n_intermediate = self.log_size - 1
        self.to_rgb_stride = 2
        for i in range(0, self.log_size - 1):
            out_channels = self.channels[i]
            self.linears.append(Conv(in_channels,out_channels,kernel_size=(1,1),stride=1,padding=0))
            self.linears.append(Conv(out_channels,out_channels,kernel_size=(1,1),stride=1,padding=0))
            '''
            self.linears.append(StyledConv(in_channels, out_channels, 1, style_dim,
                                           demodulate=demodulate, activation=activation))
            self.linears.append(StyledConv(out_channels, out_channels, 1, style_dim,
                                           demodulate=demodulate, activation=activation))
            '''
            #self.to_rgbs.append(ToRGB(out_channels, style_dim,output_channels, upsample=False))
            self.to_rgbs.append(Conv(out_channels,output_channels,kernel_size=(1,1),stride=1,padding=0))

            in_channels = out_channels



    def forward(self,
                latent,
                ):

        #print(latent.size())
        latent = self.project1(latent)
        #print(latent.size())
#         print(latent.shape)
        latent = self.project2(latent)
        #print(latent.size())
        latent, _ = self.att(latent)
        #print(latent.size())
#         print(latent.shape)
        latent = self.up_project1(latent)
        #print(latent.size())
#         print(latent.shape)
        latent = self.up_project2(latent)
#         print(latent.shape)
        #print(latent.size())
        x = latent

        rgb = 0

        for i in range(self.n_intermediate):
            for j in range(self.to_rgb_stride):
                x = self.linears[i*self.to_rgb_stride + j](x)

            temp = self.to_rgbs[i](x)
            rgb+=temp
        #print(rgb.size())
        return rgb