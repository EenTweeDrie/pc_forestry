import os
import numpy as np
import pandas as pd
from .utils import calculate_stump_parameters
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
import hdbscan
from skimage.color import rgb2lab
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist
from ...pcd.lcc import LCC
from loguru import logger
import open3d as o3d
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...coordinates.mesh_adapter import MeshAdapter


def filter_channels(pc_cells, params):
    count_before = pc_cells.points.shape[0]
    if count_before < 2:
        return pc_cells.clone()

    rgb_normalized = pc_cells.rgb / 255.0
    lab_colors = rgb2lab(rgb_normalized)
    a_channel = lab_colors[:, 1]
    b_channel = lab_colors[:, 2]

    # We want to find thresholds lab_a_threshold and lab_b_threshold
    # such that the number of points satisfying (a >= lab_a_threshold) AND (b <= lab_b_threshold)
    # is approximately half of the total number of points.
    # This is under-constrained, so we assume a relationship between the quantiles.
    # If a and b channels were independent, we would choose quantiles q_a and q_b
    # such that (1-q_a)*q_b = 0.5. A symmetric choice is 1-q_a = q_b = sqrt(0.5).
    # This means q_a = 1 - sqrt(0.5) and q_b = sqrt(0.5).
    q_a = 1.0 - np.sqrt(0.5)
    q_b = np.sqrt(0.5)

    lab_a_threshold = np.quantile(a_channel, q_a)
    lab_b_threshold = np.quantile(b_channel, q_b)

    if not params.get('lab_threshold_auto', False):
        lab_a_threshold = params.get('lab_a_threshold')
        lab_b_threshold = params.get('lab_b_threshold')

    # Маска для зелёных (a* < порога) и жёлтых (b* > порога) точек
    green_mask = a_channel < lab_a_threshold
    yellow_mask = b_channel > lab_b_threshold
    vegetation_mask = green_mask | yellow_mask

    non_vegetation_indices = np.where(~vegetation_mask)[0]
    pc_filtered = pc_cells.clone()
    pc_filtered.index_cut(non_vegetation_indices)
    count_after = pc_filtered.points.shape[0]
    logger.debug(f"Filter channels: {count_before} -> {count_after} (target: {count_before // 2})")
    return pc_filtered


def process_cell_file(pc_cells, counter, params, path_manager, mesh_adapter: Optional['MeshAdapter']):
    """Обработка одного файла ячейки для поиска пней"""
    results = {
        'counter': counter,
        'names': [],
        'x_coords': [],
        'y_coords': [],
        'diameters': []
    }
    # pc_cells.show(color_field='rgb')

    # pc_cells.show(color_field='rgb')
    print('before filter_channels', pc_cells.points.shape[0])
    pc_cells = filter_channels(pc_cells, params)
    print('after filter_channels', pc_cells.points.shape[0])
    # pc_cells.show(color_field='rgb')
    # pc_cells.show(color_field='intensity')
    count_before = pc_cells.points.shape[0]
    idx_labels = np.where(pc_cells.intensity >= params['intensity_cut'])
    pc_cells.index_cut(idx_labels)
    # pc_cells.show(color_field='intensity')
    count_after = pc_cells.points.shape[0]
    logger.debug(f"Cut intensity: {count_before} -> {count_after}")

    # Первичная кластеризация по всем координатам
    P = pc_cells.df[['x', 'y', 'z']]
    # P = pd.DataFrame(pc_cells.points, columns=['X', 'Y', 'Z'])
    # print(P)
    X = np.asarray(P)
    # clustering = hdbscan.HDBSCAN(min_samples=params.get('hdbscan_min_samples_intensity', 10), gen_min_span_tree=True).fit(X)
    # labels_stumps = clustering.labels_
    # clustering = DBSCAN(eps=3, min_samples=50).fit(X)
    # labels_stumps = clustering.labels_

    clustering = LCC(voxel_size=params['voxel_size'], connectivity=params['connectivity']).fit(pc_cells.points)
    labels_stumps = clustering.labels_

    # pc_cells.show(labels=labels_stumps)

    for i in np.unique(labels_stumps):
        if i > -1:
            pc_stump = pc_cells.clone()
            idx_label = np.where(labels_stumps == i)
            pc_stump.index_cut(idx_label)

            diameter = 0
            if pc_stump.points.shape[0] >= 4:
                xy = pc_stump.points[:, :2]
                unique_xy = np.unique(xy, axis=0)
                if unique_xy.shape[0] >= 3:
                    try:
                        hull = ConvexHull(unique_xy, qhull_options='QJ')
                        hull_points = unique_xy[hull.vertices]
                        if hull_points.shape[0] > 1:
                            diameter = pdist(hull_points).max()
                    except QhullError:
                        # Вырожденный случай: считаем диаметр по попарным расстояниям
                        if unique_xy.shape[0] > 1:
                            diameter = pdist(unique_xy).max()
                elif unique_xy.shape[0] == 2:
                    diameter = np.linalg.norm(unique_xy[0] - unique_xy[1])
                else:
                    diameter = 0

            if diameter > 1:
                pc_stump.compute_field('normals')
                pc_stump.compute_field('illuminance')
                # Рассчитываем медиану освещенности, чтобы отфильтровать 50% точек
                median_illuminance = np.percentile(pc_stump.illuminance, 50)
                illuminance_indices = np.where(pc_stump.illuminance <= median_illuminance)
                pc_stump.index_cut(illuminance_indices)

            # Проверка высоты кластера
            height = pc_stump.points.max(axis=0)[2] - pc_stump.points.min(axis=0)[2]
            if height >= params['height_limit_1'] and pc_stump.points.shape[0] > 20:
                stumps_list = process_stump_cluster(pc_stump, results['counter'], params, path_manager, mesh_adapter)

                if stumps_list:
                    for stump_data in stumps_list:
                        results['counter'] = stump_data['counter']
                        results['names'].append(stump_data['name'])
                        results['x_coords'].append(stump_data['x'])
                        results['y_coords'].append(stump_data['y'])
                        results['diameters'].append(stump_data['diameter'])
    return results


def process_stump_cluster(pc_stump, counter, params, path_manager, mesh_adapter: Optional['MeshAdapter']):
    """Обработка кластера пня"""
    # Удаление выбросов
    lof = LocalOutlierFactor(n_neighbors=params.get('lof_n_neighbors', 20), contamination=params.get('lof_contamination', 0.1))
    inliers = lof.fit_predict(pc_stump.points) > 0
    pc_stump.index_cut(inliers)

    # if pc_stump.points.shape[0] < params.get('min_points_after_lof', 50):
    #     return None

    # Кластеризация по XY координатам
    xy_clusters = cluster_by_xy(pc_stump, params)

    # pc_stump.show(labels=xy_clusters)

    found_stumps = []
    cc = 0
    ca = 0
    h = 0
    ch = 0
    for j in np.unique(xy_clusters):
        if j == -1:
            continue

        pc_stump_clear = pc_stump.clone()
        idx_label = np.where(xy_clusters == j)
        pc_stump_clear.index_cut(idx_label)

        # pc_stump_clear.show(labels=xy_clusters)

        # Проверка высоты после XY кластеризации
        height = pc_stump_clear.points.max(axis=0)[2] - pc_stump_clear.points.min(axis=0)[2]
        ca += 1
        if height >= h:
            h = height
            ch = ca
        if height >= params['height_limit_2']:
            cc += 1
            stump_data = process_final_stump(pc_stump_clear, counter, params, path_manager, mesh_adapter)
            if stump_data:
                found_stumps.append(stump_data)
                # Обновляем счетчик для следующего потенциального пня в этой же ячейке
                counter = stump_data['counter']

    # idx_label = np.where(xy_clusters == ch)
    # pc_stump.index_cut(idx_label)
    # pc_stump.show()
    return found_stumps


def cluster_by_xy(pc_stump, params):
    """Кластеризация по XY координатам"""
    clustering = LCC(voxel_size=0.1, connectivity=26).fit(pc_stump.points)
    return clustering.labels_


def process_final_stump(pc_stump_clear, counter, params, path_manager, mesh_adapter: Optional['MeshAdapter']):
    """Финальная обработка пня"""
    # Кластеризация по Z координате
    z_clusters = cluster_by_z(pc_stump_clear, params)

    # Поиск самого большого кластера по Z
    largest_cluster_idx = find_largest_cluster(z_clusters)

    if largest_cluster_idx == -1:
        logger.error("No largest cluster found")
        return None

    # Получение финального кластера пня
    pc_stump_suitable = pc_stump_clear.clone()
    idx_label = np.where(z_clusters == largest_cluster_idx)
    pc_stump_suitable.index_cut(idx_label)

    # Вычисление параметров пня
    stump_params = calculate_stump_parameters(pc_stump_suitable, params)

    if not stump_params:
        logger.error("Stump parameters are not calculated")
        return None

    # Проверка высоты пня относительно меша
    if mesh_adapter:
        max_diff_height = params.get('max_diff_height', 0.5)
        mesh_height_from = params.get('mesh_height_from', 0.1)

        stump_xy = np.array([[stump_params['x'], stump_params['y']]])
        stump_z_min = pc_stump_suitable.points[:, 2].min()

        # Используем ray casting для определения высоты меша под пнем
        points = np.hstack([stump_xy, np.zeros((len(stump_xy), 1))])
        z_max = mesh_adapter.vertices[:, 2].max()
        rays_origin = np.copy(points)
        rays_origin[:, 2] = z_max + 1.0
        rays_dir = np.array([0.0, 0.0, -1.0])
        rays_dir_tiled = np.tile(rays_dir, (len(points), 1))
        rays = np.hstack([rays_origin, rays_dir_tiled]).astype(np.float32)

        ans = mesh_adapter.scene.cast_rays(o3d.core.Tensor(rays))
        t_hit = ans['t_hit'].numpy()
        mesh_height_at_stump = (rays_origin[:, 2] - t_hit)[0]

        if np.isinf(mesh_height_at_stump):
            logger.warning(
                f"Stump {counter+1}: Не удалось определить высоту меша в точке ({stump_params['x']:.2f}, {stump_params['y']:.2f}). Пропускаю проверку высоты.")
        else:
            target_z = mesh_height_at_stump + mesh_height_from
            z_diff = abs(stump_z_min - target_z)

            if z_diff > max_diff_height:
                logger.info(f"Пень {counter+1} отброшен: слишком большая разница высот с мешем ({z_diff:.2f}м > {max_diff_height:.2f}м).")
                return None

    # Сохранение файла пня
    counter += 1
    stumps_id = params.get('stumps_id')
    filename_stumps_out = f'{stumps_id}_{str(counter).rjust(4, "0")}.pcd'
    stumps_dir = path_manager.get_stumps_dir(stumps_id)
    os.makedirs(stumps_dir, exist_ok=True)
    fname_stumps_out = os.path.join(stumps_dir, filename_stumps_out)
    pc_stump_suitable.save(fname_stumps_out)

    return {
        'counter': counter,
        'name': filename_stumps_out,
        'x': stump_params['x'],
        'y': stump_params['y'],
        'diameter': stump_params['diameter']
    }


def cluster_by_z(pc_stump_clear, params):
    """Кластеризация по Z координате"""
    P = pd.DataFrame(pc_stump_clear.points[:, 2].reshape(-1, 1), columns=['Z'])
    X = np.asarray(P)

    if pc_stump_clear.points.shape[0] < 50000:
        clustering = DBSCAN(eps=params.get('eps_Z', 0.05), min_samples=params.get('dbscan_min_samples_z', 50)).fit(X)
        return clustering.labels_
    else:
        pc_clone = pc_stump_clear.clone()
        pc_clone.points[:, :2] = 0
        clustering = LCC(voxel_size=0.01, connectivity=26).fit(pc_clone.points)
        return clustering.labels_


def find_largest_z_cluster(pc_stump_clear, labels_Z):
    """Поиск самого большого кластера по Z"""
    return find_largest_cluster(labels_Z)


def find_largest_cluster(labels: np.ndarray) -> int:
    """
    Находит метку самого большого кластера (по количеству точек), игнорируя шум (-1).

    Args:
        labels: Массив меток кластеров.

    Returns:
        Метка самого большого кластера или -1, если кластеры не найдены.
    """
    if labels.size == 0:
        return -1

    # Игнорируем шум (-1)
    labels_no_noise = labels[labels != -1]
    if labels_no_noise.size == 0:
        return -1

    unique_labels, counts = np.unique(labels_no_noise, return_counts=True)
    if unique_labels.size == 0:
        return -1

    return unique_labels[np.argmax(counts)]
