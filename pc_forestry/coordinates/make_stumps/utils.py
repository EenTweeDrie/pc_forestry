import os
import numpy as np
import pandas as pd
import circle_fit as cf
import statistics
import math


def determine_stump_center(xy_list, x_median, y_median, check_x, check_y, params: dict):
    """Определение центра пня"""
    dist_threshold = params.get('stump_center_dist_threshold', 0.25)

    # 1. Проверяем расстояние от центра первого слоя до медианного центра облака
    dist1 = math.sqrt((xy_list[0][0] - check_x)**2 + (xy_list[0][1] - check_y)**2)
    if dist1 <= dist_threshold:
        return [xy_list[0][0], xy_list[0][1]]

    # 2. Если не прошло, проверяем расстояние от медианного центра слоев до медианного центра облака
    dist2 = math.sqrt((x_median - check_x)**2 + (y_median - check_y)**2)
    if dist2 <= dist_threshold:
        return [x_median, y_median]

    # 3. Если оба центра далеко, используем медианный центр всего облака
    return [check_x, check_y]


def calculate_stump_parameters(pc_stump_suitable, params: dict):
    """Вычисление параметров пня (центр и диаметр)"""
    if pc_stump_suitable.points.shape[0] < 10:
        return None

    x_min, y_min, z_min = pc_stump_suitable.points.min(axis=0)
    x_max, y_max, z_max = pc_stump_suitable.points.max(axis=0)

    # Проверка высоты пня
    if z_max - z_min <= params.get('stump_min_height_for_calc', 0.1):
        return None

    # Анализ по слоям
    r_list = []
    xy_list = []
    num_layers = params.get('stump_calc_layers', 4)
    layer = (z_max - z_min) / num_layers

    for l in range(num_layers):
        pc_layer = pc_stump_suitable.clone()
        idx_layer = np.where(
            (pc_layer.points[:, 2] >= l * layer + z_min) &
            (pc_layer.points[:, 2] < (l + 1) * layer + z_min)
        )
        if len(idx_layer[0]) < 3:
            continue
        pc_layer.index_cut(idx_layer)

        try:
            xc, yc, r, _ = cf.hyper_fit(pc_layer.points[:, :2])
        except Exception:
            xc, yc, r = 0, 0, 0

        if r > 0:
            r_list.append(r)
            xy_list.append([xc, yc])

    if not r_list or not xy_list:
        return None

    # Вычисление медианных значений
    xy_list = np.asarray(xy_list)
    r_median = statistics.median(r_list)
    x_median = statistics.median(xy_list[:, 0])
    y_median = statistics.median(xy_list[:, 1])

    # Проверочные значения
    check_x = np.median(pc_stump_suitable.points[:, 0])
    check_y = np.median(pc_stump_suitable.points[:, 1])
    check_r_median = ((x_max - x_min) + (y_max - y_min)) / 4

    # Коррекция радиуса
    r_correction_factor = params.get('stump_radius_correction_factor', 2.1)
    if (r_median > 0.65) or (r_median > r_correction_factor * check_r_median) or (r_median == 0.0):
        r_median = check_r_median

    # Определение центра
    save_center = determine_stump_center(
        xy_list, x_median, y_median, check_x, check_y, params
    )

    return {
        'x': save_center[0],
        'y': save_center[1],
        'diameter': r_median * 2
    }


def save_stumps_results(TN, TCX, TCY, TD, params, path_manager):
    """Сохранение результатов обработки пней"""
    if not TN:
        print("Не найдено ни одного пня для сохранения.")
        return

    TN = np.asarray(TN)
    TCX = np.asarray(TCX)
    TCY = np.asarray(TCY)
    TD = np.asarray(TD)

    # Определяем идентификатор алгоритма для имен файлов и колонок
    algo_id = params.get('stump_algorithm', 'intensity')
    if algo_id == 'intensity':
        id_val = params.get('intensity_cut', 'default')
    else:
        id_val = algo_id

    # Создание DataFrame и сохранение CSV
    bd = pd.DataFrame({
        f"Name_stump_{id_val}": TN,
        "X": TCX,
        "Y": TCY,
        f"Diameter_{id_val}": TD
    })

    stumps_dir = path_manager.get_stumps_dir(id_val)
    os.makedirs(stumps_dir, exist_ok=True)
    csv_path = os.path.join(stumps_dir, f'stumps_{id_val}.csv')
    bd.to_csv(csv_path, index=False, sep=';')
    print(f"Результаты сохранены в: {csv_path}")

    # Запись пути в файл координат
    coords_file_path = os.path.join(stumps_dir, "coordinates_paths.txt")
    with open(coords_file_path, "a") as file:
        file.write(f"\n{csv_path}")
