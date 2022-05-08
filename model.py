import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
import torchvision.models as models


class Conv_Block(nn.Sequential):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super(Conv_Block, self).__init__(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size,
                      stride=stride, padding=0, groups=groups, bias=bias),
            nn.GroupNorm(
                num_groups=1, num_channels=output_channel, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )


class IRB_Block(nn.Module):
    def __init__(self, input_channel, output_channel, expansion_ratio=2):
        super(IRB_Block, self).__init__()
        self.shortcut = (input_channel == output_channel)
        bottleneck_dim = int(round(input_channel*expansion_ratio))
        layers = []
        if expansion_ratio != 1:
            layers.append(Conv_Block(
                input_channel, bottleneck_dim, kernel_size=1, padding=0))
        # dw
        layers.append(Conv_Block(bottleneck_dim, bottleneck_dim,
                      groups=bottleneck_dim, bias=True))
        # pw
        layers.append(nn.Conv2d(bottleneck_dim, output_channel,
                      kernel_size=1, padding=0, bias=False))
        layers.append(nn.GroupNorm(
            num_groups=1, num_channels=output_channel, affine=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        if self.shortcut:
            out = input + out
        return out


class Generator(nn.Module):
    def __init__(self,):
        super(Generator, self).__init__()

        self.Block_A = nn.Sequential(
            Conv_Block(3, 32, kernel_size=7, padding=3),
            Conv_Block(32, 64, stride=2, padding=(0, 1, 0, 1)),
            Conv_Block(64, 64)
        )

        self.Block_B = nn.Sequential(
            Conv_Block(64, 128, stride=2, padding=(0, 1, 0, 1)),
            Conv_Block(128, 128)
        )

        self.Block_C = nn.Sequential(
            Conv_Block(128, 128),
            IRB_Block(128, 256, 2),
            IRB_Block(256, 256, 2),
            IRB_Block(256, 256, 2),
            IRB_Block(256, 256, 2),
            Conv_Block(256, 128)
        )

        self.Block_D = nn.Sequential(
            Conv_Block(128, 128),
            Conv_Block(128, 128)
        )

        self.Block_E = nn.Sequential(
            Conv_Block(128, 64),
            Conv_Block(64, 64),
            Conv_Block(64, 32, kernel_size=7, padding=3)
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, input, align_corners=True):
        output = self.Block_A(input)
        half_size = output.size()[-2:]
        output = self.Block_B(output)
        output = self.Block_C(output)

        if align_corners:
            output = F.interpolate(
                output, size=half_size, mode="bilinear", align_corners=align_corners)
        else:
            output = F.interpolate(
                output, scale_factor=2, mode="bilinear", align_corners=align_corners)
        output = self.Block_D(output)

        if align_corners:
            output = F.interpolate(
                output, size=input.size()[-2:], mode="bilinear", align_corners=align_corners)
        else:
            output = F.interpolate(
                output, scale_factor=2, mode="bilinear", align_corners=align_corners)
        output = self.Block_E(output)

        output = self.out_layer(output)
        return output


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        channel = 32
        out_channel = channel

        layers = [
            nn.Conv2d(3, channel, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        for i in range(args.d_layers):
            layers += [
                nn.Conv2d(out_channel, channel*2, kernel_size=3,
                          stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channel*2, channel*4, kernel_size=3,
                          stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups=1, num_channels=channel*4),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            out_channel = channel * 4
            channel *= 2

        layers += [
            nn.Conv2d(channel*2, channel*2, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=channel*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel*2, 1, kernel_size=3,
                      stride=1, padding=1, bias=False),
        ]

        for i in range(len(layers)):
            if isinstance(layers[i], nn.Conv2d):
                layers[i] = spectral_norm(layers[i])

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        output = self.layers(input)
        return output

class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19,self).__init__()
        vgg19 = models.vgg19(pretrained=True).features
        self.vgg = nn.Sequential()
        i = 0
        for layer in list(vgg19):
            if i > 25:
                break
            self.vgg.add_module(str(i),layer)
            i += 1

    def forward(self,input):
        return self.vgg(input)