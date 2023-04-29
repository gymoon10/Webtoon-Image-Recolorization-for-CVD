import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


# --------------------- Blocks -----------------------------
class DownsamplerBlock(nn.Module):
    '''Downsampling by concatenating parallel output of
    3x3 conv(stride=2) & max-pooling'''

    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput,
                              (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):
    '''Factorized residual layer
    dilation can gather more context (a.k.a. atrous conv)'''

    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        # factorize 3x3 conv to (3x1) x (1x3) x (3x1) x (1x3)
        # non-linearity can be added to each decomposed 1D filters
        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
            1 * dilated, 0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
            0, 1 * dilated), bias=True, dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)  # non-linearity

        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)

        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output + input)  # +input = identity (residual connection)


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        # do not use max-unpooling operation
        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


# ---------------------- Encoder -----------------------------
class Encoder(nn.Module):

    def __init__(self):
        super().__init__()  # input: (N, 3, 256, 256)
        self.initial_block = DownsamplerBlock(3, 16)  # (N, 16, 128, 128)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))  # (N, 64, 64, 64)

        for x in range(0, 5):  # 5 times with no dilation
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))  # (N, 64, 64, 64)

        self.layers.append(DownsamplerBlock(64, 128))  # (N, 128, 32, 32)

        for x in range(0, 2):  # 2 times (with dilation)
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))  # (N, 128, 32, 32)

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        return output


# ----------------------- Decoder --------------------------------------
class Decoder(nn.Module):
    def __init__(self):  # input: (N, 128, 32, 32)
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))  # (N, 64, 64, 64)
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))  # (N, 64, 64, 64)

        self.layers.append(UpsamplerBlock(64, 16))  # (N, 16, 128, 128)
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))  # (N, 16, 128, 128)

        self.output_conv = nn.ConvTranspose2d(
            16, 3, 2, stride=2, padding=0, output_padding=0, bias=True)  # (N, 3, 256, 256)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


# ------------------------- Generator ------------------------------
class GeneratorNet(nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input):
        output = self.encoder(input)
        return self.decoder.forward(output)


# ------------------------- Discriminator -----------------------------
class Dis_block(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.stage_1 = Dis_block(in_channels * 2, 64, normalize=False)
        self.stage_2 = Dis_block(64, 128)
        self.stage_3 = Dis_block(128, 256)
        self.stage_4 = Dis_block(256, 512)

        self.patch1 = nn.Conv2d(512, 256, 3, padding=1)
        self.patch2 = nn.Conv2d(256, 64, 3, padding=1)
        self.patch3 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, a, b):  # a, b : (N, 3, 256, 256)
        x = torch.cat((a, b), 1)  # (N, 6, 256, 256)
        x = self.stage_1(x)  # (N, 64, 128, 128)
        x = self.stage_2(x)  # (N, 128, 64, 64)
        x = self.stage_3(x)  # (N, 256, 32, 32)
        x = self.stage_4(x)  # (N, 512, 16, 16)

        x = self.patch1(x)  # (N, 256, 16, 16)
        x = self.patch2(x)  # (N, 64, 16, 16)
        x = self.patch3(x)  # (N, 1, 16, 16)

        x = torch.sigmoid(x)

        return x

