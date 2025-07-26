from numba import njit, prange
import numpy as np
from numba import njit, prange


def _create_voxel_grid_fast(points: np.ndarray, grid_cell_size: float):
    """
    Создает воксельную сетку для быстрого поиска соседей, используя
    векторизованные операции NumPy. Это значительно быстрее, чем подход
    с использованием Python-словарей и циклов.
    """
    num_points = points.shape[0]
    if num_points == 0:
        return None, None, None, None

    # 1. Определяем границы и размеры сетки (это и так было быстро)
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)

    min_bound -= grid_cell_size * 2  # Добавляем отступы для надежности
    max_bound += grid_cell_size * 2

    grid_dims = np.ceil((max_bound - min_bound) /
                        grid_cell_size).astype(np.int64)
    num_cells = np.prod(grid_dims)

    # 2. Векторизованно вычисляем хэш ячейки для КАЖДОЙ точки
    #    Это заменяет медленный Python-цикл.
    point_to_cell_idx = np.floor(
        (points - min_bound) / grid_cell_size).astype(np.int64)

    # Формула для получения уникального ID (хэша) для каждой 3D-ячейки
    cell_hashes = (point_to_cell_idx[:, 0] * grid_dims[1] * grid_dims[2] +
                   point_to_cell_idx[:, 1] * grid_dims[2] +
                   point_to_cell_idx[:, 2])

    # 3. Сортируем точки по хэшу их ячеек.
    #    `np.argsort` возвращает индексы, которые бы отсортировали массив.
    #    Это самая мощная часть оптимизации.
    sorted_indices = np.argsort(cell_hashes)

    # Применяем сортировку к хэшам и к исходным индексам точек
    sorted_hashes = cell_hashes[sorted_indices]
    # `point_indices_sorted` - это и есть наш главный массив, где индексы точек
    # сгруппированы по ячейкам. Он заменяет `np.concatenate(list(grid_dict.values()))`.
    point_indices_sorted = np.arange(
        num_points)[sorted_indices].astype(np.int32)

    # 4. Находим, где начинаются группы точек для каждой ячейки.
    #    `np.unique` с `return_index=True` очень эффективно находит
    #    первое вхождение каждого уникального хэша в отсортированном массиве.
    unique_hashes, first_indices = np.unique(sorted_hashes, return_index=True)

    # 5. Создаем таблицу для быстрого доступа `cell_starts_ends`.
    #    Она будет содержать начало и конец среза в `point_indices_sorted` для каждой ячейки.
    cell_starts_ends = np.zeros((num_cells, 2), dtype=np.int32)

    # Для всех ячеек, в которых есть точки...
    # ...записываем, где начинается их блок.
    cell_starts_ends[unique_hashes, 0] = first_indices.astype(np.int32)

    # Конец блока для одной ячейки - это начало блока для следующей.
    # Поэтому мы можем "сдвинуть" массив `first_indices` и добавить в конец
    # общее число точек.
    end_indices = np.append(first_indices[1:], num_points).astype(np.int32)
    cell_starts_ends[unique_hashes, 1] = end_indices

    return point_indices_sorted, cell_starts_ends, min_bound.astype(np.float32), grid_dims.astype(np.int32)


@njit(parallel=True, fastmath=True)
def _illuminance_kernel_numba(
    points,
    normals,
    num_rays,
    max_ray_distance,
    ao_neighbor_radius,
    num_steps,
    point_indices_sorted,
    cell_starts_ends,
    min_bound,
    grid_dims,
    grid_cell_size
):
    """
    Numba-ядро для расчета AO с использованием воксельной сетки.
    """
    num_points = points.shape[0]
    illuminance = np.zeros(num_points, dtype=np.float32)
    ao_neighbor_radius_sq = ao_neighbor_radius * \
        ao_neighbor_radius  # Сравниваем квадраты расстояний

    for i in prange(num_points):
        point_p = points[i]
        normal_p = normals[i]

        if np.linalg.norm(normal_p) < 1e-6:
            # Нейтральное значение для точек без нормали
            illuminance[i] = 0.5
            continue

        occluded_count = 0
        for r_idx in range(num_rays):
            # Генерация случайного луча в полусфере нормали
            ray_dir = np.random.randn(3).astype(np.float32)
            ray_dir /= np.linalg.norm(ray_dir)
            if np.dot(ray_dir, normal_p) < 0:
                ray_dir = -ray_dir

            is_occluded_this_ray = False
            for step in range(1, num_steps + 1):
                dist_along_ray = (step / num_steps) * max_ray_distance
                test_point = point_p + ray_dir * dist_along_ray

                # --- Логика поиска в воксельной сетке ---
                grid_coords = np.floor(
                    (test_point - min_bound) / grid_cell_size).astype(np.int32)

                # Итерируемся по текущей ячейке и 26 ее соседям (куб 3x3x3)
                for dz in range(-1, 2):
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            check_coords = grid_coords + \
                                np.array([dx, dy, dz])

                            # Проверяем, что координаты ячейки в пределах сетки
                            if (check_coords[0] >= 0 and check_coords[0] < grid_dims[0] and
                                check_coords[1] >= 0 and check_coords[1] < grid_dims[1] and
                                    check_coords[2] >= 0 and check_coords[2] < grid_dims[2]):

                                # Хэш ячейки для доступа к данным
                                cell_hash = check_coords[0] * grid_dims[1] * grid_dims[2] + \
                                    check_coords[1] * grid_dims[2] + \
                                    check_coords[2]

                                start, end = cell_starts_ends[cell_hash]
                                for pt_idx_in_sorted_array in range(start, end):
                                    # Индекс точки в исходном массиве `points`
                                    j = point_indices_sorted[pt_idx_in_sorted_array]

                                    # Пропускаем проверку с самой собой
                                    if i == j:
                                        continue

                                    # Проверяем расстояние
                                    vec_to_neighbor = points[j] - \
                                        test_point
                                    dist_sq = vec_to_neighbor[0]**2 + \
                                        vec_to_neighbor[1]**2 + \
                                        vec_to_neighbor[2]**2

                                    if dist_sq < ao_neighbor_radius_sq:
                                        is_occluded_this_ray = True
                                        break  # -> выход из цикла по точкам в ячейке

                                if is_occluded_this_ray:
                                    break  # -> выход из цикла по dx
                            if is_occluded_this_ray:
                                break  # -> выход из цикла по dy
                        if is_occluded_this_ray:
                            break  # -> выход из цикла по dz

                if is_occluded_this_ray:
                    occluded_count += 1
                    break  # -> выход из цикла по шагам луча (step)

        illuminance[i] = 1.0 - (occluded_count / num_rays)

    return illuminance
