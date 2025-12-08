from numba import njit, prange
import numpy as np
from numba import njit, prange


def _generate_hemisphere_directions_fibonacci(num_dirs: int) -> np.ndarray:
    """
    Генерирует равномерно распределённые направления на полусфере (z>=0)
    с использованием распределения по Фибоначчи.

    Возвращает массив формы (num_dirs, 3), dtype float32.
    """
    if num_dirs < 1:
        return np.zeros((0, 3), dtype=np.float32)

    # Золотое сечение
    phi = (1.0 + np.sqrt(5.0)) * 0.5
    two_pi_over_phi = 2.0 * np.pi / phi

    dirs = np.zeros((num_dirs, 3), dtype=np.float32)
    # Равномерное распределение по z на [0, 1] для полусферы
    for i in range(num_dirs):
        z = (i + 0.5) / float(num_dirs)  # z in (0,1)
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = two_pi_over_phi * i
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        dirs[i, 0] = np.float32(x)
        dirs[i, 1] = np.float32(y)
        dirs[i, 2] = np.float32(z)
    return dirs


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


@njit(parallel=True, fastmath=True)
def _illuminance_pcv_bins_numba(
    points,
    normals,
    ao_neighbor_radius,
    point_indices_sorted,
    unique_hashes,
    starts,
    ends,
    min_bound,
    grid_dims,
    grid_cell_size,
    directions_local,  # (M,3) локальные направления на полусфере z>=0
    cos_aperture,      # косинус угла апертуры конуса для биннинга
    cell_radius        # радиус (в ячейках) для перебора соседних вокселей
):
    """
    Быстрый аналог CloudCompare PCV: бининг полусферы и пометка "закрытых" направлений
    соседними точками в пределах радиуса. Освещенность = доля "открытых" направлений.
    """
    num_points = points.shape[0]
    num_dirs = directions_local.shape[0]
    radius_sq = ao_neighbor_radius * ao_neighbor_radius

    illuminance = np.zeros(num_points, dtype=np.float32)

    for i in prange(num_points):
        p = points[i]
        n = normals[i]

        # Проверка нормали
        norm_n = np.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])
        if norm_n < 1e-6:
            illuminance[i] = 0.5
            continue
        ni = n / norm_n

        # Строим ортонормированный базис (t, b, n)
        # Выбираем вспомогательный вектор
        if np.abs(ni[2]) < 0.999:
            a0 = 0.0
            a1 = 0.0
            a2 = 1.0
        else:
            a0 = 0.0
            a1 = 1.0
            a2 = 0.0

        # t = normalize(cross(ni, a))
        tx = ni[1]*a2 - ni[2]*a1
        ty = ni[2]*a0 - ni[0]*a2
        tz = ni[0]*a1 - ni[1]*a0
        t_norm = np.sqrt(tx*tx + ty*ty + tz*tz)
        if t_norm < 1e-12:
            illuminance[i] = 0.5
            continue
        tx /= t_norm
        ty /= t_norm
        tz /= t_norm

        # b = cross(ni, t)
        bx = ni[1]*tz - ni[2]*ty
        by = ni[2]*tx - ni[0]*tz
        bz = ni[0]*ty - ni[1]*tx

        # Массив пометок для направлений
        occluded = np.zeros(num_dirs, dtype=np.uint8)

        # Определяем координаты ячейки для точки p
        base_cell = np.floor((p - min_bound) / grid_cell_size).astype(np.int32)

        # Перебираем окрестность вокселей
        for dz in range(-cell_radius, cell_radius + 1):
            cz = base_cell[2] + dz
            if cz < 0 or cz >= grid_dims[2]:
                continue
            for dy in range(-cell_radius, cell_radius + 1):
                cy = base_cell[1] + dy
                if cy < 0 or cy >= grid_dims[1]:
                    continue
                for dx in range(-cell_radius, cell_radius + 1):
                    cx = base_cell[0] + dx
                    if cx < 0 or cx >= grid_dims[0]:
                        continue

                    cell_hash = (
                        cx * grid_dims[1] * grid_dims[2]
                        + cy * grid_dims[2]
                        + cz
                    )

                    pos = np.searchsorted(unique_hashes, np.int64(cell_hash))
                    if not (pos < unique_hashes.shape[0] and unique_hashes[pos] == np.int64(cell_hash)):
                        continue

                    start = starts[pos]
                    end = ends[pos]

                    # Обходим точки в ячейке
                    for idx_in_sorted in range(start, end):
                        j = point_indices_sorted[idx_in_sorted]
                        if j == i:
                            continue

                        vx = points[j, 0] - p[0]
                        vy = points[j, 1] - p[1]
                        vz = points[j, 2] - p[2]
                        dist_sq = vx*vx + vy*vy + vz*vz
                        if dist_sq > radius_sq or dist_sq < 1e-18:
                            continue

                        inv_dist = 1.0 / np.sqrt(dist_sq)
                        ux = vx * inv_dist
                        uy = vy * inv_dist
                        uz = vz * inv_dist

                        # Только полусфера по нормали
                        if ux*ni[0] + uy*ni[1] + uz*ni[2] <= 0.0:
                            continue

                        # Проецируем направление соседа в локальный базис точки
                        u_local_x = ux*tx + uy*ty + uz*tz
                        u_local_y = ux*bx + uy*by + uz*bz
                        u_local_z = ux*ni[0] + uy*ni[1] + uz*ni[2]

                        # Находим ближайший биновый вектор: argmax(dot(u_local, dir))
                        best_dot = -1.0
                        best_idx = -1
                        for d_idx in range(num_dirs):
                            dlx = directions_local[d_idx, 0]
                            dly = directions_local[d_idx, 1]
                            dlz = directions_local[d_idx, 2]
                            dp = u_local_x*dlx + u_local_y*dly + u_local_z*dlz
                            if dp > best_dot:
                                best_dot = dp
                                best_idx = d_idx

                        if best_dot >= cos_aperture and best_idx >= 0:
                            occluded[best_idx] = 1

        # Доля открытых направлений
        num_closed = 0
        for d_idx in range(num_dirs):
            if occluded[d_idx] != 0:
                num_closed += 1
        open_fraction = 1.0 - (num_closed / float(num_dirs))
        illuminance[i] = open_fraction

    return illuminance


@njit(parallel=True, fastmath=True)
def _illuminance_pcv_dda_numba(
    points,
    normals,
    max_ray_distance,
    ao_neighbor_radius,
    point_indices_sorted,
    unique_hashes,
    starts,
    ends,
    min_bound,
    grid_dims,
    grid_cell_size,
    directions_local,  # (M,3) локальные направления (z>=0)
    cos_aperture,      # косинус апертуры
    tan_half_aperture  # тангенс половины апертуры
):
    """
    PCV с фиксированными направлениями и шаганием по воксельной сетке (DDA-подобно).
    Для каждого направления идем от точки вдоль луча до max_ray_distance,
    на каждом шаге проверяем точки в окрестных ячейках.
    """
    num_points = points.shape[0]
    num_dirs = directions_local.shape[0]
    ao_r_sq = ao_neighbor_radius * ao_neighbor_radius

    illuminance = np.zeros(num_points, dtype=np.float32)

    # Шаг по расстоянию: по ячейке
    step_len = grid_cell_size
    max_steps = 1
    if max_ray_distance > 1e-9 and step_len > 1e-9:
        max_steps = int(max_ray_distance / step_len)
        if max_steps < 1:
            max_steps = 1

    for i in prange(num_points):
        p = points[i]
        n = normals[i]

        # нормаль
        n_norm = np.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])
        if n_norm < 1e-6:
            illuminance[i] = 0.5
            continue
        ni = n / n_norm

        # локальный базис (t, b, n)
        if np.abs(ni[2]) < 0.999:
            ax, ay, az = 0.0, 0.0, 1.0
        else:
            ax, ay, az = 0.0, 1.0, 0.0

        tx = ni[1]*az - ni[2]*ay
        ty = ni[2]*ax - ni[0]*az
        tz = ni[0]*ay - ni[1]*ax
        t_norm = np.sqrt(tx*tx + ty*ty + tz*tz)
        if t_norm < 1e-12:
            illuminance[i] = 0.5
            continue
        tx /= t_norm
        ty /= t_norm
        tz /= t_norm

        bx = ni[1]*tz - ni[2]*ty
        by = ni[2]*tx - ni[0]*tz
        bz = ni[0]*ty - ni[1]*tx

        num_closed = 0

        for d_idx in range(num_dirs):
            # локальное направление -> мировое
            dlx = directions_local[d_idx, 0]
            dly = directions_local[d_idx, 1]
            dlz = directions_local[d_idx, 2]

            # world_dir = dlx*t + dly*b + dlz*n
            wx = dlx*tx + dly*bx + dlz*ni[0]
            wy = dlx*ty + dly*by + dlz*ni[1]
            wz = dlx*tz + dly*bz + dlz*ni[2]

            # нормируем (на всякий случай)
            w_norm = np.sqrt(wx*wx + wy*wy + wz*wz)
            if w_norm < 1e-12:
                continue
            inv_w = 1.0 / w_norm
            wx *= inv_w
            wy *= inv_w
            wz *= inv_w

            occluded = False

            for s in range(1, max_steps + 1):
                tp_x = p[0] + wx * (s * step_len)
                tp_y = p[1] + wy * (s * step_len)
                tp_z = p[2] + wz * (s * step_len)

                cx = int(np.floor((tp_x - min_bound[0]) / grid_cell_size))
                cy = int(np.floor((tp_y - min_bound[1]) / grid_cell_size))
                cz = int(np.floor((tp_z - min_bound[2]) / grid_cell_size))

                # проверяем текущую ячейку и соседние (3x3x3)
                for dz in range(-1, 2):
                    zc = cz + dz
                    if zc < 0 or zc >= grid_dims[2]:
                        continue
                    for dy in range(-1, 2):
                        yc = cy + dy
                        if yc < 0 or yc >= grid_dims[1]:
                            continue
                        for dx in range(-1, 2):
                            xc = cx + dx
                            if xc < 0 or xc >= grid_dims[0]:
                                continue

                            cell_hash = xc * grid_dims[1] * grid_dims[2] + yc * grid_dims[2] + zc
                            pos = np.searchsorted(unique_hashes, np.int64(cell_hash))
                            if not (pos < unique_hashes.shape[0] and unique_hashes[pos] == np.int64(cell_hash)):
                                continue

                            start = starts[pos]
                            end = ends[pos]

                            for idx_in_sorted in range(start, end):
                                j = point_indices_sorted[idx_in_sorted]
                                if j == i:
                                    continue
                                # Вектор от точки p до кандидата q
                                ux = points[j, 0] - p[0]
                                uy = points[j, 1] - p[1]
                                uz = points[j, 2] - p[2]
                                # Проекция на направление луча
                                t = ux*wx + uy*wy + uz*wz
                                if t <= 0.0 or t > max_ray_distance:
                                    continue
                                # Перпендикулярное расстояние до оси луча
                                rx = ux - wx * t
                                ry = uy - wy * t
                                rz = uz - wz * t
                                r2 = rx*rx + ry*ry + rz*rz
                                # Радиус конуса на расстоянии t (с «толщиной» ядра)
                                cone_r = t * tan_half_aperture
                                if cone_r < ao_neighbor_radius:
                                    cone_r = ao_neighbor_radius
                                cone_r2 = cone_r * cone_r
                                if r2 <= cone_r2:
                                    occluded = True
                                    break
                            if occluded:
                                break
                        if occluded:
                            break
                    if occluded:
                        break

                if occluded:
                    num_closed += 1
                    break

        open_fraction = 1.0 - (num_closed / float(num_dirs))
        illuminance[i] = open_fraction

    return illuminance
