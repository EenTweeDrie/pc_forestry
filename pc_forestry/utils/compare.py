import numpy as np
from scipy.spatial import cKDTree


def diff(src_points: np.ndarray, tgt_points: np.ndarray, tolerance: float = 0.01) -> np.ndarray:
    """
    Возвращает массив numpy из 0/1 длиной N (N = число точек в `src_points`).

    Логика: для каждой точки из `src_points` проверяется, существует ли точка в
    `tgt_points` на расстоянии не более `tolerance` (евклидова метрика). Если да —
    считаем, что точка "вычлась" (метка 1), иначе — 0.

    Параметры
    - src_points: массив numpy (Nx3)
    - tgt_points: массив numpy (Mx3)
    - tolerance: допустимая погрешность по координатам (в тех же единицах, что и точки)

    Возвращает
    - np.ndarray формы (N,), dtype=np.uint8: 1 — точка вычлась, 0 — не вычлась
    """
    if src_points.size == 0:
        return np.empty((0,), dtype=np.uint8)
    if tgt_points.size == 0:
        return np.zeros((src_points.shape[0],), dtype=np.uint8)

    # KD-дерево по целевому облаку
    tree = cKDTree(tgt_points)

    # Ищем ближайшего соседа для каждой точки источника с ограничением по радиусу
    distances, _ = tree.query(src_points, k=1, distance_upper_bound=tolerance, workers=-1)

    # Если совпадение не найдено, расстояние будет np.inf
    removed_mask = np.isfinite(distances)
    return removed_mask


def diff_area(src_points: np.ndarray, tgt_points: np.ndarray, tolerance: float = 0.01) -> np.ndarray:
    """
    Возвращает массив numpy из 0/1 длиной N (N = число точек в `src_points`).

    Логика: для каждой точки из `src_points` проверяется, попадает ли ХОТЯ БЫ ОДНА
    точка из `tgt_points` в шар радиуса `tolerance` вокруг неё. Если да — метка 1,
    иначе — 0. Внутри используется радиусный поиск (учитываются все соседи),
    однако результирующая метка — это факт наличия хотя бы одного соседа.

    Параметры
    - src_points: массив numpy (Nx3)
    - tgt_points: массив numpy (Mx3)
    - tolerance: радиус шара в тех же единицах, что и точки

    Возвращает
    - np.ndarray формы (N,), dtype=np.uint8: 1 — точка вычлась, 0 — не вычлась
    """
    if src_points.size == 0:
        return np.empty((0,), dtype=np.uint8)
    if tgt_points.size == 0:
        return np.zeros((src_points.shape[0],), dtype=np.uint8)

    # KD-дерево по целевому облаку: считаем количество соседей для каждой точки
    # источника в пределах радиуса tolerance
    tree = cKDTree(tgt_points)
    counts = tree.query_ball_point(src_points, r=tolerance, workers=-1, return_length=True)
    removed_mask = (counts > 0)
    return removed_mask


def segment_vertical_planes_mask(
    points: np.ndarray,
    distance_threshold: float = 0.1,
    min_points_for_plane: int = 1000,
    verticality_threshold: float = 0.15,
    num_iterations: int = 1000,
) -> list[bool]:
    """
    Сегментирует вертикальные плоскости (стены) в облаке точек с помощью RANSAC.

    Функция итеративно находит наибольшие плоскости в облаке. Для каждой
    найденной плоскости проверяется ее вертикальность. Точки, принадлежащие
    вертикальным плоскостям, помечаются в итоговой маске.

    Параметры
    ----------
    points : np.ndarray
        Входное облако точек (Nx3).
    distance_threshold : float, optional
        Максимальное расстояние от точки до плоскости для RANSAC, by default 0.1.
    min_points_for_plane : int, optional
        Минимальное количество точек для поиска новой плоскости, by default 1000.
    verticality_threshold : float, optional
        Порог для Z-компоненты нормали. Вектор нормали нормирован, поэтому
        abs(c) - это косинус угла между нормалью и осью Z. Для вертикальной
        плоскости этот угол близок к 90 градусам, а косинус - к 0.
        Значение 0.15 соответствует углу > 81 градуса. By default 0.15.
    num_iterations : int, optional
        Количество итераций RANSAC, by default 1000.

    Возвращает
    -------
    list[bool]
        Маска (N,), где True - точка принадлежит вертикальной
        плоскости, False - в противном случае.
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "Open3D library is required. Please install it using 'pip install open3d'"
        )

    if points.shape[0] < min_points_for_plane:
        return [False] * points.shape[0]

    # Преобразуем в формат Open3D
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(points)

    # Маска для результата и индексы для итеративной обработки
    wall_mask = [False] * points.shape[0]
    remaining_indices = np.arange(points.shape[0])

    while len(remaining_indices) > min_points_for_plane:
        # Работаем с под-облаком на каждой итерации
        remaining_pcd = pcd_o3d.select_by_index(remaining_indices)

        # Ищем наибольшую плоскость в оставшемся облаке
        plane_model, inliers_local = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold, ransac_n=3, num_iterations=num_iterations
        )

        # Если найдено слишком мало точек, прекращаем поиск
        if len(inliers_local) < min_points_for_plane:
            break

        # Получаем параметры плоскости: ax + by + cz + d = 0
        # (a, b, c) - это вектор нормали к плоскости
        [a, b, c, d] = plane_model

        # Проверяем, является ли плоскость вертикальной
        if abs(c) < verticality_threshold:
            # Это вертикальная плоскость.
            # Находим глобальные индексы инлаеров и помечаем их в маске
            inliers_global = remaining_indices[inliers_local]
            for idx in inliers_global:
                wall_mask[idx] = True

        # Удаляем найденную плоскость (инлаеры) из рассмотрения для следующей итерации,
        # независимо от того, вертикальная она или нет.
        remaining_indices = np.delete(remaining_indices, inliers_local)

    return np.array(wall_mask)


def select_n_largest_clusters(labels, n):
    """
    Выбирает n самых больших кластеров по количеству точек.

    Args:
        labels (list[int] or np.ndarray): Индексы (метки) кластеров для набора точек.
                                          Метка 0 игнорируется, так как обычно обозначает фон или шум.
        n (int): Количество самых больших кластеров для выбора.

    Returns:
        list[int]: Список меток n самых больших кластеров, отсортированный по убыванию их размера.
    """
    if n <= 0:
        return []

    # np.unique эффективно подсчитывает количество вхождений каждой метки
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Создаем словарь {метка: количество}, исключая метку 0
    cluster_counts = {label: count for label, count in zip(unique_labels, counts) if label != 0}

    # Если не найдено ни одного кластера (кроме фона), возвращаем пустой список
    if not cluster_counts:
        return []

    # Сортируем метки кластеров по убыванию их размера
    sorted_labels = sorted(cluster_counts, key=cluster_counts.get, reverse=True)

    # Возвращаем n самых больших, или все, если их меньше n
    return sorted_labels[:n]
