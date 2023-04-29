import torch
import torch.nn as nn


# UNet
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels)),

        layers.append(nn.LeakyReLU(0.2))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        x = self.down(x)
        return x


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((x, skip), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):  # (N, 3, 256, 256)
        d1 = self.down1(x)  # (N, 64, 128, 128)
        d2 = self.down2(d1)  # (N, 128, 64, 64)
        d3 = self.down3(d2)  # (N, 256, 32, 32)
        d4 = self.down4(d3)  # (N, 512, 16, 16)
        d5 = self.down5(d4)  # (N, 512, 8, 8)
        d6 = self.down6(d5)  # (N, 512, 4, 4)
        d7 = self.down7(d6)  # (N, 512, 2, 2)
        d8 = self.down8(d7)  # (N, 512, 1, 1)

        u1 = self.up1(d8, d7)  # (N, 1024, 2, 2)
        u2 = self.up2(u1, d6)  # (N, 1024, 4, 4)
        u3 = self.up3(u2, d5)  # (N, 1024, 8, 8)
        u4 = self.up4(u3, d4)  # (N, 1024, 16, 16)
        u5 = self.up5(u4, d3)  # (N, 512, 32, 32)
        u6 = self.up6(u5, d2)  # (N, 256, 64, 64)
        u7 = self.up7(u6, d1)  # (N, 128, 128, 128)
        u8 = self.up8(u7)  # (N, 3, 256, 256)

        return u8

class Dis_block(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
    
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.stage_1 = Dis_block(in_channels*2, 64, normalize=False)
        self.stage_2 = Dis_block(64, 128)
        self.stage_3 = Dis_block(128, 256)
        self.stage_4 = Dis_block(256, 512)

        self.patch = nn.Conv2d(512, 1, 3, padding=1)  # 16x16 패치 생성

    def forward(self, a, b):
        x = torch.cat((a, b), 1)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.patch(x)
        x = torch.sigmoid(x)

        return x


# 가중치 초기화
def initialize_weights(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
