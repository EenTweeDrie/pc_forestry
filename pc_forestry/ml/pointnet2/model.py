from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pc_forestry.predict.models.pointnet2_utils import (
    PointNetFeaturePropagation,
    PointNetSetAbstraction,
)


class PointNet2Segmenter(nn.Module):
    """
    Лёгкий PointNet++ (SSG) для point-wise сегментации.

    Вход:  (B, C, N), где первые 3 канала — xyz
    Выход: (B, K, N) logits
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        sa1_npoint: int = 256,
        sa2_npoint: int = 64,
        sa3_npoint: int = 16,
    ) -> None:
        super().__init__()
        if in_channels < 3:
            raise ValueError("in_channels должен быть >= 3, так как первые каналы — xyz.")
        if num_classes <= 1:
            raise ValueError("num_classes должен быть >= 2.")

        extra_channels = int(in_channels) - 3
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.extra_channels = extra_channels
        self.model_kwargs = {
            "sa1_npoint": int(sa1_npoint),
            "sa2_npoint": int(sa2_npoint),
            "sa3_npoint": int(sa3_npoint),
        }

        self.sa1 = PointNetSetAbstraction(
            npoint=int(sa1_npoint),
            radius=0.15,
            nsample=32,
            in_channel=extra_channels + 3,
            mlp=[32, 32, 64],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=int(sa2_npoint),
            radius=0.3,
            nsample=32,
            in_channel=64 + 3,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=int(sa3_npoint),
            radius=0.6,
            nsample=32,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 256, 512],
            group_all=True,
        )

        self.fp4 = PointNetFeaturePropagation(512 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(256 + 128, [256, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 64, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128 + extra_channels, [128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xyz = x[:, :3, :]
        points = x[:, 3:, :] if self.extra_channels > 0 else None

        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, points, l1_points)

        feat = F.relu(self.bn1(self.conv1(l0_points)))
        feat = self.drop1(feat)
        return self.conv2(feat)
