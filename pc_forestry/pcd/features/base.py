from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class VoxelFeature(ABC):
    """
    Базовый интерфейс фичи вокселей.
    Реализация должна задать:
    - name: уникальное имя фичи
    - dim: размерность выхода (1 для скаляра, k для вектора)
    - doc: краткое описание
    - compute(grid, **kwargs) -> np.ndarray формы (N,) или (N, dim)
    """

    name: str = ""
    dim: int = 1
    doc: str = ""

    @abstractmethod
    def compute(self, grid: Any, **kwargs) -> np.ndarray:
        raise NotImplementedError
