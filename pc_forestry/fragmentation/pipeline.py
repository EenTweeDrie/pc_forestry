from typing import Any, Dict, Optional
from ..path_manager import PathManager
from ..coordinates.multi_pipeline import MultiCoordinatesPipeline
import os
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from ..pcd.PCD import PCD
from ..pcd.lcc import LCC
from ..pcd.is_inside import ray_tracing_numpy_numba
from ..utils.compare import diff, diff_area, segment_vertical_planes_mask, select_n_largest_clusters
from scipy.spatial.distance import cdist
from loguru import logger
from ..pcd.TREE import TREE
from ..coordinates.split_trees import split_by_trunks_and_coords, load_known_coords_from_csv


class FragmentationPipeline:
    def __init__(self, base_path: str, file_path: str, mesh_path: str) -> None:
        self.params: Dict[str, Any] = {}
        self.base_path = base_path
        self.path_manager = PathManager().set_base_dir(base_path)
        self.file_path = file_path
        self.mesh_path = mesh_path
        self.file_name = os.path.basename(file_path)
        self._output_dir: Optional[str] = None

    def set_params(self, params: Dict[str, Any]) -> "FragmentationPipeline":
        self.params.update(params)
        return self

    def update_params(self, params: Dict[str, Any]) -> "FragmentationPipeline":
        self.params.update(params)
        return self

    def coordinates(self, version: str, base_params: dict | None = None) -> "FragmentationPipeline":
        logger.info(f"Running coordinates_first for {self.file_name}")
        base_params = {
            'mesh_height_from': 0.3,
            'mesh_height_to': 3,
            'stump_algorithm': 'intensity',
            'algo': 'birch',
            'n_clusters': 12,
            'height_limit_1': 0.65,
            'height_limit_2': 1.35,
            'eps_XY': 0.4,
            'eps_Z': 3.5,
            'intensity_cut_vor_tes_percent': 99,
            'voxel_size': 0.5,
            'connectivity': 26
        } or base_params

        param_sets = [
            {**base_params, 'intensity_cut': 10000, 'lab_threshold_auto': False, 'lab_a_threshold': -5, 'lab_b_threshold': 10, 'priority': 0},
            {**base_params, 'intensity_cut': 10000, 'lab_threshold_auto': True, 'priority': 1},
            {**base_params, 'intensity_cut': 10000, 'lab_threshold_auto': False, 'lab_a_threshold': -50, 'lab_b_threshold': 50, 'priority': 2},
        ]

        if version == 'v1':
            file_path = self.file_path
        else:
            file_path = os.path.join(self.base_path, f"{os.path.splitext(os.path.basename(self.file_name))[0]}_{version}.pcd")

        multi_cp = (
            MultiCoordinatesPipeline(
                base_path=os.path.join(self.base_path, f'coordinates_{version}'),
                file_path=file_path
            )
            .set_mesh(self.mesh_path)
            .set_param_sets(param_sets)
        )

        multi_cp.run(force_cut=True, force_cells=True, force_stumps=True)
        return self

    def choose_segments(self, version: str) -> "FragmentationPipeline":
        logger.info(f"Running choose_segments for {self.file_name}")
        if version == 'v1':
            postfix = ''
        else:
            postfix = f'_{version}'
        coords = pd.read_csv(os.path.join(self.base_path, f'coordinates_{version}', os.path.splitext(
            os.path.basename(self.file_name))[0] + postfix + f'_Clear_Excess.csv'), sep=';')
        pc_source = PCD.read(self.file_path)

        # Директория для сохранения найденных кластеров
        save_dir = os.path.join(self.base_path, f'found_clusters_{version}')
        coord_save_dir = os.path.join(save_dir, 'coord')
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(coord_save_dir, exist_ok=True)

        # Для ускорения извлечем координаты в numpy массив
        coords_xy = coords[['X', 'Y']].values

        for i in np.unique(pc_source.tree_id):
            mask = (pc_source.tree_id == i)
            pc_cluster = pc_source.clone().index_cut(mask)

            if pc_cluster.points.shape[0] == 0:
                continue

            clustering = LCC(voxel_size=0.5, connectivity=26).fit(pc_cluster.points)
            for j in np.unique(clustering.labels_):
                if j == -1:  # Пропускаем шум
                    continue

                mask_mini = (clustering.labels_ == j)
                pc_cluster_mini = pc_cluster.clone().index_cut(mask_mini)

                if pc_cluster_mini.points.shape[0] == 0:
                    continue

                # Создаем выпуклую оболочку (многоугольник) из 2D-проекции кластера
                points_2d = pc_cluster_mini.points[:, :2]

                is_inside_mask = np.zeros(coords_xy.shape[0], dtype=bool)
                # Для создания оболочки нужно как минимум 3 точки
                if points_2d.shape[0] >= 3:
                    try:
                        hull = ConvexHull(points_2d)
                        polygon = points_2d[hull.vertices]
                        # Проверяем, попадает ли какая-либо координата из файла coords в границы многоугольника
                        is_inside_mask = ray_tracing_numpy_numba(coords_xy, polygon)
                    except Exception:  # scipy.spatial.qhull.QhullError может возникнуть для коллинеарных точек
                        # is_inside_mask останется со значениями False, если не удалось построить оболочку
                        pass

                # Проверка теперь основана на многоугольнике, а не на отдельных диапазонах X/Y.
                # Чтобы следующий код (`if np.any(x_in_range & y_in_range)`) работал,
                # присваиваем результирующую маску обеим переменным.
                x_in_range = is_inside_mask
                y_in_range = is_inside_mask

                # Если нашлась хотя бы одна координата, сохраняем кластер
                if np.any(x_in_range & y_in_range):
                    base_filename = f"cluster_tree_id_{i}_label_{j}"

                    # Сохраняем кластер
                    save_path = os.path.join(save_dir, f"{base_filename}.pcd")
                    pc_cluster_mini.save(save_path)

                    # Сохраняем соответствующие координаты
                    matching_coords = coords[is_inside_mask]
                    coord_save_path = os.path.join(coord_save_dir, f"{base_filename}.csv")
                    matching_coords.to_csv(coord_save_path, index=False, sep=';')

        return self

    def vertical_planes(self, version: str, find_walls: bool = True) -> "FragmentationPipeline":
        logger.info(f"Running vertical_planes for {self.file_name}")
        # Директория для сохранения финальных кластеров
        final_save_dir = os.path.join(self.base_path, 'final_clusters')
        os.makedirs(final_save_dir, exist_ok=True)

        found_clusters_dir = os.path.join(self.base_path, f'found_clusters_{version}')
        coord_save_dir = os.path.join(found_clusters_dir, 'coord')

        for file in os.listdir(found_clusters_dir):
            if not file.endswith('.pcd'):
                continue

            pc = PCD.read(os.path.join(found_clusters_dir, file))

            # По умолчанию сохраняем все точки
            keep_mask = np.ones(pc.points.shape[0], dtype=bool)

            # Находим точки, принадлежащие вертикальным плоскостям (стенам)
            if find_walls:
                wall_mask = segment_vertical_planes_mask(
                    pc.points,
                    distance_threshold=0.1,
                    min_points_for_plane=50000,
                    verticality_threshold=0.05,
                    num_iterations=1000
                )
                pc_wall = pc.clone().index_cut(wall_mask)

                if pc_wall.points.shape[0] > 100:
                    # Кластеризуем точки стены
                    clustering_wall = LCC(voxel_size=0.1, connectivity=18).fit(pc_wall.points)
                    labels_wall = clustering_wall.labels_

                    # Находим метки N самых больших кластеров
                    largest_cluster_labels = select_n_largest_clusters(labels_wall, 3)

                    # Создаем маску для удаления точек, принадлежащих самым большим кластерам
                    remove_mask = np.zeros(pc.points.shape[0], dtype=bool)
                    remove_mask[wall_mask] = np.isin(labels_wall, largest_cluster_labels)

                    # Инвертируем маску, чтобы получить точки для сохранения
                    keep_mask = ~remove_mask

                if not np.any(keep_mask):
                    continue

                pc_filtered = pc.clone().index_cut(keep_mask)
            else:
                pc_filtered = pc

            if pc_filtered.points.shape[0] < 3:
                continue

            clustering = LCC(voxel_size=0.25, connectivity=26).fit(pc_filtered.points)
            labels = clustering.labels_

            unique_labels, counts = np.unique(labels, return_counts=True)

            # Оставляем только кластеры размером > 10000 точек и убираем шум (-1)
            valid_clusters_mask = (unique_labels != -1) & (counts > 10000)
            unique_labels = unique_labels[valid_clusters_mask]

            if len(unique_labels) == 0:
                continue

            # Открываем файл с координатами, принадлежащему этому файлу
            base_filename = os.path.splitext(file)[0]
            coord_path = os.path.join(coord_save_dir, f"{base_filename}.csv")

            if not os.path.exists(coord_path):
                continue

            current_coords = pd.read_csv(coord_path, sep=';')
            current_coords_xy = current_coords[['X', 'Y']].values

            # Инициализируем массивы для хранения ближайшего кластера и минимального расстояния для каждой координаты
            min_distances = np.full(current_coords_xy.shape[0], np.inf)
            closest_cluster_labels = np.full(current_coords_xy.shape[0], -1, dtype=int)

            # Перебираем каждый кластер, чтобы найти ближайший для каждой координаты
            for label in unique_labels:
                cluster_mask = (labels == label)
                cluster_points_xy = pc_filtered.points[cluster_mask, :2]

                # Для создания оболочки нужно как минимум 3 точки
                if cluster_points_xy.shape[0] < 3:
                    continue

                try:
                    # Создаем выпуклую оболочку (многоугольник) из 2D-проекции кластера
                    hull = ConvexHull(cluster_points_xy)
                    polygon = cluster_points_xy[hull.vertices]

                    # Находим координаты, которые попадают внутрь оболочки кластера
                    is_inside_mask = ray_tracing_numpy_numba(current_coords_xy, polygon)

                    if not np.any(is_inside_mask):
                        continue

                    # Координаты, попавшие внутрь
                    inside_coords_xy = current_coords_xy[is_inside_mask]

                    # Рассчитываем расстояние от каждой внутренней координаты до ближайшей точки в кластере
                    distances_to_points = cdist(inside_coords_xy, cluster_points_xy).min(axis=1)

                    # Обновляем информацию о ближайшем кластере, если найдено меньшее расстояние
                    current_min_distances = min_distances[is_inside_mask]
                    update_mask = distances_to_points < current_min_distances

                    # Получаем глобальные индексы координат, которые нужно обновить
                    global_indices_to_update = np.where(is_inside_mask)[0][update_mask]

                    # Обновляем минимальные расстояния и метки ближайших кластеров
                    min_distances[global_indices_to_update] = distances_to_points[update_mask]
                    closest_cluster_labels[global_indices_to_update] = label

                except Exception:  # scipy.spatial.qhull.QhullError может возникнуть для коллинеарных точек
                    pass

            # Получаем уникальные метки кластеров, которые являются ближайшими к какой-либо из координат
            final_labels_to_save = np.unique(closest_cluster_labels)
            final_labels_to_save = final_labels_to_save[final_labels_to_save != -1]

            # Сохраняем только эти кластеры
            for i, label_to_save in enumerate(final_labels_to_save):
                final_mask = (labels == label_to_save)
                final_cluster_pc = pc_filtered.clone().index_cut(final_mask)

                if final_cluster_pc.points.shape[0] > 0:
                    final_filename = f"{base_filename}_final_cluster_{i}.pcd"
                    final_save_path = os.path.join(final_save_dir, final_filename)
                    final_cluster_pc.save(final_save_path)

        return self

    def merge_clusters(self, version: str) -> "FragmentationPipeline":
        logger.info(f"Running merge_clusters for {self.file_name}")
        pc_merge = None
        for file in os.listdir(os.path.join(self.base_path, 'final_clusters')):
            if not file.endswith('.pcd'):
                continue

            pc = PCD.read(os.path.join(self.base_path, 'final_clusters', file))

            if pc_merge is None:
                pc_merge = pc.clone()
            else:
                pc_merge.append(pc)

        pc_merge.save(os.path.join(self.base_path, f"{os.path.splitext(os.path.basename(self.file_name))[0]}_{version}.pcd"))

        return self

    def split_trees(self, version: str) -> "FragmentationPipeline":

        os.makedirs(os.path.join(self.base_path, 'out'), exist_ok=True)

        for file in os.listdir(os.path.join(self.base_path, f'found_clusters_{version}')):
            full_path = os.path.join(self.base_path, f'found_clusters_{version}', file)
            if not file.endswith('.pcd'):
                continue
            pc = TREE.read(full_path)
            pc.find_trunk_ml(
                model_path=r'D:\lidar\data\classification\v3\models\catboost_model.pkl',
                config={'voxel_size': 0.3, 'type_df': 'original', 'fast_mode': True, 'proba_threshold': 0.4},
            )

            # Загрузите XY-координаты деревьев для этого кластера (если есть)
            csv_path = os.path.join(self.base_path, f'found_clusters_{version}', 'coord', file.replace('.pcd', '.csv'))
            # coords_xy = load_known_coords_from_csv(csv_path)

            # Выполнить сегментацию
            trees = split_by_trunks_and_coords(
                pc,
                known_tree_coords_xy=None,  # или None, если хотите опираться только на стволы
                params={
                    'n_neighbors': 16,
                    'beta': 1.0,
                    'max_match_dist': 0.35,     # радиус «прилипания» координат к найденным стволам
                    'z_slice_height': 1.5,     # высота нижнего среза для поиска стволов
                    'min_cluster_size': 1000,    # минимальный размер кластера ствола
                }
            )
            file_name = file.split('.')[0]

            # trees — это список PCD, каждый элемент — отдельное дерево
            for i, t in enumerate(trees):
                if t.points.shape[0] < 5000:
                    continue
                t.save(os.path.join(self.base_path, 'out', f'{file_name}_{i}.pcd'))

        return self

    def run(self,) -> "FragmentationPipeline":
        logger.info(f"Running run for {self.file_name}")
        self.coordinates(version='v1')
        self.choose_segments(version='v1')
        self.vertical_planes(version='v1')
        self.merge_clusters(version='v2')
        self.coordinates(version='v2')
        self.choose_segments(version='v2')
        self.split_trees(version='v2')
        return self
