from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet3DLight(nn.Module):
    """
    Лёгкая 3D U-Net для сегментации по вокселям.

    Вход:  (B, C, D, H, W)
    Выход: (B, K, D, H, W) logits
    """

    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 16) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels должен быть > 0")
        if num_classes <= 1:
            raise ValueError("num_classes должен быть >= 2 (для CrossEntropy)")

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.enc1 = _conv_block(in_channels, c1)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = _conv_block(c1, c2)
        self.pool2 = nn.MaxPool3d(2)
        self.bottleneck = _conv_block(c2, c3)

        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _conv_block(c2 + c2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _conv_block(c1 + c1, c1)

        self.head = nn.Conv3d(c1, num_classes, kernel_size=1)

    @staticmethod
    def _center_crop_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Center-crop x по D/H/W под размер ref (используется для skip-коннектов).
        """
        _, _, D, H, W = x.shape
        _, _, d, h, w = ref.shape
        sd = max((D - d) // 2, 0)
        sh = max((H - h) // 2, 0)
        sw = max((W - w) // 2, 0)
        return x[:, :, sd:sd + d, sh:sh + h, sw:sw + w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        # Из-за нечётных размеров возможны расхождения — подгоняем кропом
        e2c = self._center_crop_to(e2, u2)
        d2 = self.dec2(torch.cat([u2, e2c], dim=1))

        u1 = self.up1(d2)
        e1c = self._center_crop_to(e1, u1)
        d1 = self.dec1(torch.cat([u1, e1c], dim=1))

        return self.head(d1)
