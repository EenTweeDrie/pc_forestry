from numba import njit, prange
import numpy as np
from numba import njit, prange


def _create_voxel_grid_fast(points: np.ndarray, grid_cell_size: float):
    """
    Создает разреженное представление воксельной сетки для быстрого поиска соседей.

    Возвращает:
      - point_indices_sorted (int32): индексы точек, отсортированные по хэшам ячеек
      - unique_hashes (int64): уникальные хэши занятых ячеек (отсортированы)
      - starts (int32): начальные индексы срезов в point_indices_sorted для каждой ячейки
      - ends (int32): конечные индексы (excl.) срезов для каждой ячейки
      - min_bound (float32): минимальная граница сетки
      - grid_dims (int32): размеры сетки по осям (используется для пересчета хэша)
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

    # 2. Векторизованно вычисляем хэш ячейки для КАЖДОЙ точки
    #    Это заменяет медленный Python-цикл.
    point_to_cell_idx = np.floor(
        (points - min_bound) / grid_cell_size).astype(np.int64)

    # Формула для получения уникального ID (хэша) для каждой 3D-ячейки
    cell_hashes = (point_to_cell_idx[:, 0] * grid_dims[1] * grid_dims[2] +
                   point_to_cell_idx[:, 1] * grid_dims[2] +
                   point_to_cell_idx[:, 2]).astype(np.int64)

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
    starts = first_indices.astype(np.int32)
    ends = np.empty_like(starts)
    if starts.size > 0:
        ends[:-1] = starts[1:]
        ends[-1] = np.int32(num_points)

    return (
        point_indices_sorted,
        unique_hashes.astype(np.int64),
        starts,
        ends,
        min_bound.astype(np.float32),
        grid_dims.astype(np.int32),
    )


@njit(parallel=True, fastmath=True)
def _illuminance_kernel_numba(
    points,
    normals,
    num_rays,
    max_ray_distance,
    ao_neighbor_radius,
    num_steps,
    point_indices_sorted,
    unique_hashes,
    starts,
    ends,
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

                                # Хэш ячейки для доступа к данным (int64)
                                cell_hash = (
                                    check_coords[0] * grid_dims[1] * grid_dims[2]
                                    + check_coords[1] * grid_dims[2]
                                    + check_coords[2]
                                )

                                # Поиск ячейки в разреженном списке уникальных хэшей
                                pos = np.searchsorted(unique_hashes, np.int64(cell_hash))
                                if pos < unique_hashes.shape[0] and unique_hashes[pos] == np.int64(cell_hash):
                                    start = starts[pos]
                                    end = ends[pos]
                                else:
                                    start = 0
                                    end = 0

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
