# pc_forestry/ml/ml_inferencer.py
"""
Этот модуль содержит класс для выполнения инференса (предсказаний)
с использованием обученных моделей машинного обучения.
"""

import os
from typing import Any

import numpy as np

try:
    from ..pcd.TREE import TREE
    from ..pcd.VOXEL import VOXELGRID
except ImportError:
    TREE = None
    VOXELGRID = None


class MLInferencer:
    """
    Класс для выполнения инференса (предсказания) на данных о деревьях.
    """

    def __init__(self, model: Any):
        """
        Инициализирует инференсер с обученной моделью.

        Args:
            model (Any): Обученная модель (например, CatBoost, RandomForest и т.д.).
        """
        if model is None:
            raise ValueError("Модель не может быть None.")
        self._model = model

    def predict_for_tree(self, file_path: str, voxel_size: float = 0.5) -> 'VOXELGRID':
        """
        Выполняет инференс для одного дерева из файла.

        Для каждого слоя вокселей вычисляются признаки и делается предсказание.
        Полученные предсказания сохраняются в атрибут `label` каждого вокселя.

        Args:
            file_path (str): Путь к файлу с данными о дереве (.pcd, .las, .txt).
            voxel_size (float): Размер вокселя.

        Returns:
            VOXELGRID: Объект воксельной сетки с результатами предсказания.
        """
        if TREE is None or VOXELGRID is None:
            raise RuntimeError(
                "Модули TREE/VOXELGRID недоступны. Проверьте PYTHONPATH."
            )

        assert os.path.exists(file_path), f"Файл не найден: {file_path}"

        pc = TREE.read(file_path)
        pc.shift_to_coordinate()
        if pc.normals is None or np.all(pc.normals == 0):
            pc.compute_feature('normals')
        if pc.illuminance is None or np.all(pc.illuminance == 0):
            pc.compute_feature('illuminance')

        vg = VOXELGRID.create(pc, voxel_size, verbose=False)

        index = np.array([voxel.index for voxel in vg.voxels])
        max_layer = int(np.max(index[:, 2]))

        for voxel in vg.voxels:
            voxel.label = 2  # unclassified

        voxels_total: list = []

        for layer in range(max_layer + 1):
            vg.calculate_distances_to_previous_layer_by_layer(
                pc.coordinate, layer=layer)
            voxels_layer = vg.get_voxels_by_layer(layer=layer)
            if not voxels_layer:
                continue

            voxels_total.extend(voxels_layer)

            # Создаем временную воксельную сетку, включающую все воксели до текущего слоя
            vg_total = VOXELGRID(PC=None, voxel_size=vg.voxel_size,
                                 voxels=list(voxels_total))
            vg_total.calculate_distances_to_coordinate(pc.coordinate)

            # Подготовка признаков для всех обработанных вокселей
            df_features = vg_total.normalized_df.drop(
                columns=["label"], errors="ignore")

            # Предсказание вероятностей
            proba = self._model.predict_proba(df_features)[:, 1]

            # Присваиваем метку и вероятность только вокселям текущего слоя
            # Используем предсказания для последних добавленных вокселей
            for voxel, p in zip(voxels_layer, proba[-len(voxels_layer):]):
                voxel.label = int(p >= 0.5)  # Порог 0.5
                voxel.proba = p

        return vg
