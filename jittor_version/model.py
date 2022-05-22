import jittor as jt
from jittor import init
from jittor import nn
from jittor import models
from utils import init_D_weights,init_G_weights

jt.flags.use_cuda = 1

class Conv_Block(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super(Conv_Block, self).__init__()
        layers = []
        layers.append(nn.ReflectionPad2d(padding))
        layers.append(nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias))
        layers.append(nn.GroupNorm(num_groups=1, num_channels=output_channel, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        self.layers = nn.Sequential(*layers)

    def execute(self, input):
        out = self.layers(input)
        # print(out)
        return out

class IRB_Block(nn.Module):
    def __init__(self, input_channel, output_channel, expansion_ratio=2):
        super(IRB_Block, self).__init__()
        self.shortcut = (input_channel == output_channel)
        bottleneck_dim = int(round((input_channel * expansion_ratio)))
        layers = []
        if (expansion_ratio != 1):
            layers.append(Conv_Block(input_channel, bottleneck_dim, kernel_size=1, padding=0))
        layers.append(Conv_Block(bottleneck_dim, bottleneck_dim, groups=bottleneck_dim, bias=True))
        layers.append(nn.Conv2d(bottleneck_dim, output_channel, kernel_size=1, padding=0, bias=False))
        layers.append(nn.GroupNorm(num_groups=1, num_channels=output_channel, affine=True))
        self.layers = nn.Sequential(*layers)

    def execute(self, input):
        out = self.layers(input)
        if self.shortcut:
            out = (input + out)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.Block_A = nn.Sequential(Conv_Block(3, 32, kernel_size=7, padding=3), Conv_Block(32, 64, stride=2, padding=(0, 1, 0, 1)), Conv_Block(64, 64))
        self.Block_B = nn.Sequential(Conv_Block(64, 128, stride=2, padding=(0, 1, 0, 1)), Conv_Block(128, 128))
        self.Block_C = nn.Sequential(Conv_Block(128, 128), IRB_Block(128, 256, 2), IRB_Block(256, 256, 2), IRB_Block(256, 256, 2), IRB_Block(256, 256, 2), Conv_Block(256, 128))
        self.Block_D = nn.Sequential(Conv_Block(128, 128), Conv_Block(128, 128))
        self.Block_E = nn.Sequential(Conv_Block(128, 64), Conv_Block(64, 64), Conv_Block(64, 32, kernel_size=7, padding=3))
        self.out_layer = nn.Sequential(nn.Conv(32, 3, 1, stride=1, padding=0, bias=False), nn.Tanh())
        init_G_weights(self)

    def execute(self, input, align_corners=True):
        output = self.Block_A(input)
        half_size = output.shape[- 2:]
        output = self.Block_B(output)
        output = self.Block_C(output)
        if align_corners:
            output = nn.interpolate(output, size=half_size, mode='bilinear', align_corners=align_corners)
        else:
            output = nn.interpolate(output, scale_factor=2, mode='bilinear', align_corners=align_corners)
        output = self.Block_D(output)
        if align_corners:
            output = nn.interpolate(output, size=input.shape[(- 2):], mode='bilinear', align_corners=align_corners)
        else:
            output = nn.interpolate(output, scale_factor=2, mode='bilinear', align_corners=align_corners)
        output = self.Block_E(output)
        output = self.out_layer(output)
        return output

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        channel = 32
        out_channel = channel
        layers = [nn.Conv(3, channel, 3, stride=1, padding=1, bias=False), nn.LeakyReLU(scale=0.2)]
        for i in range(1, args.d_layers):
            layers += [nn.Conv(out_channel, (channel * 2), 3, stride=2, padding=1, bias=False), nn.LeakyReLU(scale=0.2), nn.Conv((channel * 2), (channel * 4), 3, stride=1, padding=1, bias=False), nn.GroupNorm(1, (channel * 4), affine=None), nn.LeakyReLU(scale=0.2)]
            out_channel = (channel * 4)
            channel *= 2
        layers += [nn.Conv((channel * 2), (channel * 2), 3, stride=1, padding=1, bias=False), nn.GroupNorm(1, (channel * 2), affine=None), nn.LeakyReLU(scale=0.2), nn.Conv((channel * 2), 1, 3, stride=1, padding=1, bias=False)]
        self.layers = nn.Sequential(*layers)
        init_D_weights(self)

    def execute(self, input):
        output = self.layers(input)
        return output

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        vgg19 = jt.models.vgg19(pretrained=True).features
        self.vgg = nn.Sequential()
        i = 0
        for layer in list(vgg19):
            if (i > 25):
                break
            self.vgg.add_module(str(i), layer)
            i += 1

    def execute(self, input):
        return self.vgg(self.normalize_input(input))

    def normalize_input(self, input):
        input = (((input + 1) / 2) * 255)
        b = input[..., 0, :, :]
        g = input[..., 1, :, :]
        r = input[..., 2, :, :]
        b = (b - VGG_MEAN[0])
        g = (g - VGG_MEAN[1])
        r = (r - VGG_MEAN[2])
        b1 = jt.array(b)
        g1 = jt.array(g)
        r1 = jt.array(r)
        img = jt.misc.stack((b1, g1, r1), 1)
        return img
