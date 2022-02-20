import math

import torch
from torch import nn
import torch.nn.functional as F

from model.blocks import ConstantInput, LFF, PixelNorm, EqualLinear, StyledResBlock, ToRGBNoMod, EqualConv2d, StyledConvNoNoise, ConvLayer, Self_Attn, ConstantInputPatch

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


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3,stride=1,padding=1, downsample=True):
        super().__init__()

        self.conv1 = Conv(in_channel, in_channel, kernel_size,stride,padding)
        self.conv2 = Conv(in_channel, out_channel, kernel_size,stride,padding, downsample=downsample)

        self.skip = Conv(in_channel, out_channel, 1,stride,0, downsample=downsample)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            #self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)
            self.blur=nn.Conv2d(out_channel,out_channel,kernel_size=(3,3),stride=(1,1),padding=(1,1))

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            #self.blur = Blur(blur_kernel, pad=(pad0, pad1))
            self.blur=nn.Conv2d(in_channel,in_channel,kernel_size=(3,3),stride=(1,1),padding=(1,1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out



class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        activation=None,
        downsample=False,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
            downsample=downsample,
        )

        self.activation = activation
        self.noise = NoiseInjection()
        self.activate=nn.LeakyReLU(0.2)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out





class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise



class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim,out_channel, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.upsample = upsample
        if upsample:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv = ModulatedConv2d(in_channel, out_channel, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            if self.upsample:
                skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(self, size1=256,size2=256, hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
                 activation=None, linear_size = 512, channel_multiplier=2, coord_size = 3,output_channels=1, **kwargs):
        super(Generator, self).__init__()

        self.size1 = size1
        self.size2=size2
        demodulate = True
        self.demodulate = demodulate
        self.coord_size = coord_size
        self.lff = LFF(hidden_size, coord_size = coord_size)
        self.emb = ConstantInput(hidden_size, size1=size1,size2=size2)


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

        multiplier = 2
        in_channels = int(self.channels[0])
        self.project1=Conv(2,in_channels,3,stride=1,padding=1,downsample=True)
        #self.project2 = ConvLayer(in_channels, in_channels, 3, downsample = True)
        self.project2=Conv(in_channels,in_channels,3,stride=1,padding=1,downsample=True)
        self.att=Self_Attn(in_channels,"relu")
        #self.up_project1 = ConvLayer(in_channels, in_channels, 3, upsample = True)
        #self.up_project2 = ConvLayer(in_channels, in_channels, 3, upsample = True)
        self.up_project1=Conv(in_channels,in_channels,3,stride=1,padding=1,upsample=True)
        self.up_project2=Conv(in_channels,in_channels,3,stride=1,padding=1,upsample=True)
        #self.conv1=Conv(int(multiplier*hidden_size),in_channels,1,0,0,upsample=False,downsample=False,use_noise=True)
        self.conv1 = StyledConv(int(multiplier*hidden_size), in_channels, 1, style_dim, demodulate=demodulate,activation=activation)

        self.linears = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.log_size = min(int(math.log(size1, 2)),int(math.log(size2, 2)))

        self.n_intermediate = self.log_size - 1
        self.to_rgb_stride = 2
        for i in range(0, self.log_size - 1):
            out_channels = self.channels[i]
            self.linears.append(StyledConv(in_channels,out_channels,1,style_dim,demodulate=demodulate,activation=activation))
            self.linears.append(StyledConv(out_channels,out_channels,1,style_dim,demodulate=demodulate,activation=activation))
            self.to_rgbs.append(ToRGB(out_channels,style_dim,output_channels,upsample=False))

            in_channels = out_channels

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(nn.Linear(style_dim,style_dim))
            layers.append(nn.LeakyReLU())

        self.style = nn.Sequential(*layers)

    def forward(self,
                coords,
                latent,
                input2,
                noise,
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                ):

        noise = noise[0]

        if truncation < 1:
            noise = truncation_latent + truncation * (noise - truncation_latent)

        if not input_is_latent:
            noise = self.style(noise)

        x = self.lff(coords)

        batch_size, _, w, h = coords.shape
        emb = self.emb(x)
        x = torch.cat([x, emb], 1)
        x = self.conv1(x, noise)

        if len(input2.shape) == 1:
            input2 = input2.unsqueeze(0)
            latent = latent.unsqueeze(0)

        latent = torch.cat([latent, input2], 1)
        latent = self.project1(latent)
#         print(latent.shape)
        latent = self.project2(latent)
        latent, _ = self.att(latent)
#         print(latent.shape)
        latent = self.up_project1(latent)
#         print(latent.shape)
        latent = self.up_project2(latent)
#         print(latent.shape)

        x = x + latent

        rgb = 0

        for i in range(self.n_intermediate):
            for j in range(self.to_rgb_stride):
                x = self.linears[i*self.to_rgb_stride + j](x, noise)

            rgb = self.to_rgbs[i](x, noise, rgb)

        if return_latents:
            return rgb, noise
        else:
            return rgb


class Discriminator(nn.Module):
    def __init__(self, size1,size2, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], input_size=3, n_first_layers=0, **kwargs):
        super().__init__()

        self.input_size = input_size
        log_size=min(int(math.log(size1,2)),int(math.log(size2,2)))
        size=2**log_size
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [Conv(input_size, channels[size], 1,1,0)]
        convs.extend([ConvLayer(channels[size], channels[size], 3,1,1) for _ in range(n_first_layers)])


        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = Conv(in_channel + 1, channels[4], 3,1,1)
        self.final_linear = nn.Sequential(
            nn.Linear(channels[4] * 5 * 5, channels[4]),
            nn.LeakyReLU(),
            nn.Linear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out