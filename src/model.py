"""Две модели: U-Net, написанный с нуля, и U-Net с предобученным энкодером ResNet-34."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from classes import NUM_CLASSES


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Классический U-Net: 4 понижения разрешения, пропускные соединения, BatchNorm."""

    def __init__(self, num_classes=NUM_CLASSES, chs=(32, 64, 128, 256, 512)):
        super().__init__()
        self.enc = nn.ModuleList([block(3 if i == 0 else chs[i - 1], c) for i, c in enumerate(chs)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2) for i in range(len(chs) - 1, 0, -1)])
        self.dec = nn.ModuleList([block(chs[i - 1] * 2, chs[i - 1]) for i in range(len(chs) - 1, 0, -1)])
        self.head = nn.Conv2d(chs[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips[:-1])):
            x = dec(torch.cat([up(x), skip], 1))
        return self.head(x)


class ResNetUNet(nn.Module):
    """U-Net с энкодером ResNet-34 (ImageNet): сильно лучше с нуля на малом датасете (800 картинок)."""

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        r = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)                  # 1/2,  64
        self.pool, self.l1, self.l2, self.l3, self.l4 = r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4
        self.d4 = block(512 + 256, 256)    # 1/16
        self.d3 = block(256 + 128, 128)    # 1/8
        self.d2 = block(128 + 64, 64)      # 1/4
        self.d1 = block(64 + 64, 64)       # 1/2
        self.head = nn.Conv2d(64, num_classes, 1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        s0 = self.stem(x); s1 = self.l1(self.pool(s0)); s2 = self.l2(s1); s3 = self.l3(s2); s4 = self.l4(s3)
        d = self.d4(torch.cat([self._up(s4, s3), s3], 1))
        d = self.d3(torch.cat([self._up(d, s2), s2], 1))
        d = self.d2(torch.cat([self._up(d, s1), s1], 1))
        d = self.d1(torch.cat([self._up(d, s0), s0], 1))
        return F.interpolate(self.head(d), size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(name, pretrained=True):
    if name == "unet":
        return UNet()
    if name == "resnet34_unet":
        return ResNetUNet(pretrained=pretrained)
    raise ValueError(f"Неизвестная модель: {name}")
