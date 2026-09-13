from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.block(features)


class AttentionGate(nn.Module):
    def __init__(self, skip_channels: int, gate_channels: int, hidden: int) -> None:
        super().__init__()
        self.skip = nn.Conv2d(skip_channels, hidden, 1, bias=False)
        self.gate = nn.Conv2d(gate_channels, hidden, 1, bias=False)
        self.score = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(hidden, 1, 1), nn.Sigmoid())

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate = nn.functional.interpolate(gate, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        weights = self.score(self.skip(skip) + self.gate(gate))
        return skip * weights


class PlantSegmentationNet(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 16)
        self.enc2 = ConvBlock(16, 32)
        self.bridge = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.att2 = AttentionGate(32, 32, 16)
        self.dec2 = ConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.att1 = AttentionGate(16, 16, 8)
        self.dec1 = ConvBlock(32, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        skip1 = self.enc1(image)
        skip2 = self.enc2(self.pool(skip1))
        bridge = self.bridge(self.pool(skip2))
        x = self.up2(bridge)
        x = self.dec2(torch.cat((x, self.att2(skip2, x)), dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat((x, self.att1(skip1, x)), dim=1))
        return self.head(x)
