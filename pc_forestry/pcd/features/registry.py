from __future__ import annotations

from typing import Dict, Iterable, List
from .base import VoxelFeature


class FeatureRegistry:
    def __init__(self) -> None:
        self._features: Dict[str, VoxelFeature] = {}

    def register(self, feature: VoxelFeature, overwrite: bool = False) -> None:
        name = feature.name
        if not name:
            raise ValueError("Feature must have non-empty name")
        if (not overwrite) and (name in self._features):
            raise ValueError(f"Feature '{name}' is already registered")
        self._features[name] = feature

    def get(self, name: str) -> VoxelFeature:
        if name not in self._features:
            raise KeyError(f"Feature '{name}' is not registered")
        return self._features[name]

    def names(self) -> List[str]:
        return sorted(self._features.keys())

    def items(self) -> Iterable[tuple[str, VoxelFeature]]:
        return self._features.items()


registry = FeatureRegistry()


def register_feature(feature: VoxelFeature, overwrite: bool = False) -> None:
    registry.register(feature, overwrite=overwrite)
