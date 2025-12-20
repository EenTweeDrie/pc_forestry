"""
3D CNN (воксельная сегментация) поверх признаков, сохранённых в features.csv.

Ожидаемый формат входного CSV (sep=';'):
- x, y, z: целочисленные индексы вокселей
- name: идентификатор объекта/дерева (группировка в 3D объёмы)
- target: целевая метка для вокселя (класс)
- остальные числовые колонки: входные признаки (каналы)
"""

from .dataset import Features3DVolumeDataset, pad_collate_3d  # noqa: F401
from .model import UNet3DLight  # noqa: F401
