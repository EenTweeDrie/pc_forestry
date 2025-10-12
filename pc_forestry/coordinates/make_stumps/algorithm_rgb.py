import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import hdbscan
from skimage.color import rgb2lab
from .utils import calculate_stump_parameters


def process_stump_candidate(pc_stump, counter, params, path_manager):
    """
    Финальная обработка и проверка кандидата в пни.
    Вычисляет параметры и сохраняет результат, если кандидат валиден.
    """
    # Проверка на минимальное количество точек в кандидате
    if pc_stump.points.shape[0] < params.get('min_points_in_stump', 100):
        return None

    # Вычисление параметров пня
    stump_params = calculate_stump_parameters(pc_stump, params)

    # Фильтр по диаметру пня
    if stump_params and params.get('stump_min_diameter', 0.05) < stump_params['diameter'] < params.get('stump_max_diameter', 1.5):
        counter += 1
        filename_stumps_out = f'rgb_{str(counter).rjust(4, "0")}.pcd'
        # Используем 'rgb' в качестве идентификатора для папки
        stumps_dir = path_manager.get_stumps_dir('rgb')
        os.makedirs(stumps_dir, exist_ok=True)
        fname_stumps_out = os.path.join(stumps_dir, filename_stumps_out)
        pc_stump.save(fname_stumps_out)

        return {
            'counter': counter,
            'name': filename_stumps_out,
            'x': stump_params['x'],
            'y': stump_params['y'],
            'diameter': stump_params['diameter']
        }

    return None


def filter_channels(pc_cells, params):
    rgb_normalized = pc_cells.rgb / 255.0
    lab_colors = rgb2lab(rgb_normalized)
    a_channel = lab_colors[:, 1]
    b_channel = lab_colors[:, 2]
    # Маска для зелёных (a* < порога) и жёлтых (b* > порога) точек
    green_mask = a_channel < params.get('lab_a_threshold', -10)
    yellow_mask = b_channel > params.get('lab_b_threshold', 50)
    vegetation_mask = green_mask | yellow_mask

    print(vegetation_mask.sum(), pc_cells.points.shape[0])

    non_vegetation_indices = np.where(~vegetation_mask)[0]
    pc_filtered = pc_cells.clone()
    pc_filtered.index_cut(non_vegetation_indices)
    return pc_filtered


def process_cell_file_rgb(pc_cells, counter, params, path_manager):
    """
    Обработка одного файла ячейки для поиска пней на основе цвета (RGB).
    """
    results = {
        'counter': counter,
        'names': [],
        'x_coords': [],
        'y_coords': [],
        'diameters': []
    }

    if not hasattr(pc_cells, 'rgb') or pc_cells.rgb is None:
        # Если в облаке нет информации о цвете, выходим
        return results

    # 1. Фильтрация по цвету
    # Нормализуем RGB в диапазон [0, 1] для конвертации
    rgb_normalized = pc_cells.rgb / 255.0
    # Конвертация в цветовое пространство LAB
    lab_colors = rgb2lab(rgb_normalized)

    a_channel = lab_colors[:, 1]
    b_channel = lab_colors[:, 2]

    # Маска для зелёных (a* < порога) и жёлтых (b* > порога) точек
    green_mask = a_channel < params.get('lab_a_threshold', -10)
    yellow_mask = b_channel > params.get('lab_b_threshold', 50)
    vegetation_mask = green_mask | yellow_mask

    print(vegetation_mask.sum(), pc_cells.points.shape[0])

    non_vegetation_indices = np.where(~vegetation_mask)[0]
    # pc_cells.show(color_field='rgb')
    pc_filtered = pc_cells.clone()
    pc_filtered.index_cut(non_vegetation_indices)
    # pc_filtered.show(color_field='rgb')

    if pc_filtered.points.shape[0] < params.get('min_points_for_clustering', 500):
        return results

    # 2. Пространственная кластеризация HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=params.get('hdbscan_min_cluster_size', 150),
        min_samples=params.get('hdbscan_min_samples', 20),
        gen_min_span_tree=True
    )
    clustering = clusterer.fit(pc_filtered.points)
    labels = clustering.labels_

    # 3. Обработка каждого кластера
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        for i in tqdm(unique_labels, desc="Processing RGB clusters", leave=False):
            if i == -1:  # -1 это шум в HDBSCAN
                continue

            pc_stump_candidate = pc_filtered.clone()
            idx_label = np.where(labels == i)
            pc_stump_candidate.index_cut(idx_label)

            # 4. Геометрическая фильтрация кластеров по высоте
            height = pc_stump_candidate.points.max(axis=0)[2] - pc_stump_candidate.points.min(axis=0)[2]

            if params.get('stump_min_height', 0.1) < height < params.get('stump_max_height', 2.0):
                stump_data = process_stump_candidate(
                    pc_stump_candidate, results['counter'], params, path_manager
                )
                if stump_data:
                    results['counter'] = stump_data['counter']
                    results['names'].append(stump_data['name'])
                    results['x_coords'].append(stump_data['x'])
                    results['y_coords'].append(stump_data['y'])
                    results['diameters'].append(stump_data['diameter'])

    return results
