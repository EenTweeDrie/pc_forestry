# pc_forestry/ml/ml_inferencer.py
"""
Этот модуль содержит класс для выполнения инференса (предсказаний)
с использованием обученных моделей машинного обучения.
"""
from __future__ import annotations
import os
from typing import Any
from tqdm import tqdm
import numpy as np

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:
    TabNetClassifier = None

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

    def predict_for_tree(self, pc: TREE, voxel_size: float) -> VOXELGRID:
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

        pc.shift_to_coordinate()
        pc.compute_feature('normals')
        if pc.illuminance is None or np.all(pc.illuminance == 0):
            pc.compute_feature('illuminance')

        vg = VOXELGRID.create(pc, voxel_size, verbose=False)

        index = np.array([voxel.index for voxel in vg.voxels])
        max_layer = int(np.max(index[:, 2]))
        min_layer = int(np.min(index[:, 2]))

        for voxel in vg.voxels:
            voxel.label = 2  # unclassified

        voxels_total: list = []

        for layer in tqdm(range(min_layer, max_layer + 1), desc="Инференс по слоям"):
            vg.calculate_distances_to_previous_layer_by_layer(pc.coordinate, layer=layer)
            vg.calculate_distances_to_previous_layer_by_layer_XY(pc.coordinate, layer=layer)
            voxels_layer = vg.get_voxels_by_layer(layer=layer)
            if not voxels_layer:
                continue

            voxels_total.extend(voxels_layer)

            # Создаем временную воксельную сетку, включающую все воксели до текущего слоя
            vg_total = VOXELGRID(PC=None, voxel_size=vg.voxel_size,
                                 voxels=list(voxels_total))
            vg_total.calculate_distances_to_coordinate(pc.coordinate)
            vg_total.calculate_distances_to_coordinate_XY(pc.coordinate)

            # Подготовка признаков для всех обработанных вокселей
            df_features = vg_total.normalized_df.drop(
                columns=["label"], errors="ignore")

            # Предсказание вероятностей
            # TabNet требует numpy-массив, в то время как другие модели могут работать с DataFrame
            if TabNetClassifier and isinstance(self._model, TabNetClassifier):
                features_np = df_features.fillna(0).values.astype(np.float32)
                proba = self._model.predict_proba(features_np)
                proba = proba[:, 0]
            else:
                proba = self._model.predict_proba(df_features)[:, 1]

            # Присваиваем метку и вероятность только вокселям текущего слоя
            # Используем предсказания для последних добавленных вокселей
            for voxel, p in zip(voxels_layer, proba[-len(voxels_layer):]):
                voxel.label = int(p >= 0.5)  # Порог 0.5
                voxel.proba = p

        return vg
