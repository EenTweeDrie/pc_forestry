import os
import heapq
from typing import List, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors
import hdbscan

from ..pcd.TREE import TREE
from ..pcd.PCD import PCD


def _detect_trunk_seeds_xy(trunk_pcd: PCD,
                           z_slice_height: float = 1.5,
                           min_cluster_size: int = 40) -> np.ndarray:
    """
    Выделяет центры стволов в XY из облака `trunk_pcd` через кластеризацию HDBSCAN
    по нижнему срезу по высоте.

    Returns:
        np.ndarray формы (N, 2): координаты центров кластеров стволов в XY.
    """
    points = trunk_pcd.points
    if points is None or points.size == 0:
        return np.empty((0, 2))

    z_min = float(np.min(points[:, 2]))
    mask = points[:, 2] <= (z_min + z_slice_height)
    lower = points[mask]
    if lower.shape[0] < min_cluster_size:
        return np.empty((0, 2))

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, core_dist_n_jobs=1)
    labels = clusterer.fit_predict(lower[:, :2])
    uniq = np.unique(labels[labels >= 0])
    if uniq.size == 0:
        return np.empty((0, 2))

    centers = []
    for lab in uniq:
        pts = lower[labels == lab, :2]
        if pts.shape[0] == 0:
            continue
        centers.append(np.mean(pts, axis=0))
    if not centers:
        return np.empty((0, 2))
    return np.vstack(centers)


def _prepare_seeds_xy(tree: TREE,
                      known_tree_coords_xy: Optional[np.ndarray],
                      max_match_dist: float = 1.5,
                      z_slice_height: float = 1.5,
                      min_cluster_size: int = 40) -> np.ndarray:
    """
    Объединяет детектированные центры стволов из `tree.trunk` и заданные координаты деревьев.
    Каждая известная координата сопоставляется ближайшему детектированному стволу в радиусе
    `max_match_dist`; если не найден, используется сама координата как сид.

    Args:
        tree: Объект TREE с заполненным `tree.trunk`.
        known_tree_coords_xy: np.ndarray формы (M, 2) или None.
    Returns:
        np.ndarray формы (K, 2): набор сидов в XY.
    """
    seeds_detected = np.empty((0, 2))
    if tree.trunk is not None and tree.trunk.points is not None and tree.trunk.points.size > 0:
        seeds_detected = _detect_trunk_seeds_xy(tree.trunk,
                                                z_slice_height=z_slice_height,
                                                min_cluster_size=min_cluster_size)

    if known_tree_coords_xy is None or known_tree_coords_xy.size == 0:
        return seeds_detected

    # Свести сиды из координат к центрам детектированных стволов, если они близко
    if seeds_detected.size > 0:
        from sklearn.neighbors import KDTree
        tree_kd = KDTree(seeds_detected)
        dists, idxs = tree_kd.query(known_tree_coords_xy, k=1)
        dists = dists.ravel()
        idxs = idxs.ravel()

        snapped = []
        for i, (d, j) in enumerate(zip(dists, idxs)):
            if d <= max_match_dist:
                snapped.append(seeds_detected[j])
            else:
                snapped.append(known_tree_coords_xy[i])
        snapped = np.asarray(snapped)
        # Удалим дубли
        if snapped.shape[0] > 1:
            snapped = np.unique(np.round(snapped, 4), axis=0)
        # Добавим невостребованные детектированные сиды (на случай лишних стволов)
        all_seeds = snapped
        if seeds_detected.shape[0] > 0:
            all_seeds = np.vstack([all_seeds, seeds_detected])
            all_seeds = np.unique(np.round(all_seeds, 4), axis=0)
        return all_seeds
    else:
        # Нет детекции – используем только заданные координаты
        return known_tree_coords_xy


def _build_knn_graph_xy(points_xy: np.ndarray,
                        n_neighbors: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Строит kNN граф по XY.

    Returns:
        indices: (N, n_neighbors) индексы соседей
        distances: (N, n_neighbors) евклидовы расстояния до соседей
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    nn.fit(points_xy)
    distances, indices = nn.kneighbors(points_xy)
    return indices, distances


def _estimate_local_density(distances: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Оценивает локальную плотность как 1 / (среднее расстояние до k соседей + eps).
    distances: (N, k)
    """
    mean_dist = np.maximum(distances.mean(axis=1), eps)
    return 1.0 / mean_dist


def _multi_source_dijkstra(labels_seeds_idx: List[int],
                           neighbor_indices: np.ndarray,
                           neighbor_distances: np.ndarray,
                           point_density: np.ndarray,
                           beta: float = 1.0) -> np.ndarray:
    """
    Многосеменной Дейкстра по графу kNN.

    Стоимость перехода i->j: d_ij * ((rho_i + rho_j)/2)^beta,
    где rho – оценка локальной плотности (чем выше, тем дороже),
    чтобы границы проходили через области минимальной плотности.

    Returns:
        labels: (N,) метка сида для каждой точки [0..S-1]
    """
    num_points = neighbor_indices.shape[0]
    dist = np.full(num_points, np.inf, dtype=np.float64)
    label = np.full(num_points, -1, dtype=np.int32)
    visited = np.zeros(num_points, dtype=bool)

    heap = []
    for sid, idx in enumerate(labels_seeds_idx):
        dist[idx] = 0.0
        label[idx] = sid
        heapq.heappush(heap, (0.0, int(idx), sid))

    while heap:
        cur_d, i, sid = heapq.heappop(heap)
        if visited[i]:
            continue
        visited[i] = True

        nbrs = neighbor_indices[i]
        nbrd = neighbor_distances[i]
        rho_i = point_density[i]
        for j, d_ij in zip(nbrs, nbrd):
            if visited[j]:
                continue
            rho_j = point_density[j]
            step_cost = float(d_ij) * float(((rho_i + rho_j) * 0.5) ** beta)
            nd = cur_d + step_cost
            if nd < dist[j]:
                dist[j] = nd
                label[j] = sid
                heapq.heappush(heap, (nd, int(j), sid))

    return label


def split_by_trunks_and_coords(tree: TREE,
                               known_tree_coords_xy: Optional[np.ndarray] = None,
                               params: Optional[dict] = None) -> List[PCD]:
    """
    Разделяет кластер `tree` на отдельные деревья, используя стволы (`tree.trunk`)
    и известные координаты деревьев. Разделение происходит по наименее плотной
    границе (геодезическое многоисточниковое разрастание с весами от плотности).

    Args:
        tree: Объект TREE, для которого уже вызван `tree.find_trunk_ml(...)`.
        known_tree_coords_xy: np.ndarray формы (M, 2) в системе координат облака (метры).
        params: Словарь параметров:
            - n_neighbors (int): размер kNN графа, по умолчанию 16
            - beta (float): степень влияния плотности, по умолчанию 1.0
            - max_match_dist (float): радиус прилипания известной координаты к детектированному стволу, м
            - z_slice_height (float): высота нижнего среза для поиска стволов, м
            - min_cluster_size (int): мин. размер кластера HDBSCAN для стволов
    Returns:
        Список под-облаков PCD, соответствующих отдельным деревьям.
    """
    if params is None:
        params = {}
    n_neighbors = int(params.get('n_neighbors', 16))
    beta = float(params.get('beta', 1.0))
    max_match_dist = float(params.get('max_match_dist', 1.5))
    z_slice_height = float(params.get('z_slice_height', 1.5))
    min_cluster_size = int(params.get('min_cluster_size', 40))

    if tree.points is None or tree.points.size == 0:
        return []

    # Подготовим сиды в XY
    seeds_xy = _prepare_seeds_xy(
        tree,
        known_tree_coords_xy=known_tree_coords_xy,
        max_match_dist=max_match_dist,
        z_slice_height=z_slice_height,
        min_cluster_size=min_cluster_size,
    )
    if seeds_xy.size == 0:
        # Нет сидов – нечего разделять, вернуть весь кластер как один
        return [tree.clone()]

    # Построим граф по XY всем точкам кластера
    points_xy = tree.points[:, :2]
    nbr_idx, nbr_dist = _build_knn_graph_xy(points_xy, n_neighbors=n_neighbors)
    density = _estimate_local_density(nbr_dist)

    # Привязка сидов к ближайшим точкам облака
    from sklearn.neighbors import KDTree
    kd_all = KDTree(points_xy)
    _, seed_point_idx = kd_all.query(seeds_xy, k=1)
    seed_point_idx = seed_point_idx.ravel().tolist()

    labels = _multi_source_dijkstra(seed_point_idx, nbr_idx, nbr_dist, density, beta=beta)

    # Соберём под-облака по меткам
    result: List[PCD] = []
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        mask = labels == lab
        sub = tree.clone()
        sub.index_cut(mask)
        result.append(sub)
    return result


def load_known_coords_from_csv(file_path: str) -> np.ndarray:
    """
    Загрузка координат деревьев из CSV/TXT. Ожидаются как минимум два столбца (X, Y).
    Пытается автоматически определить разделитель.
    """
    if not os.path.exists(file_path):
        return np.empty((0, 2))
    try:
        import pandas as pd
        # sep=None + engine='python' позволяет авто-определить разделитель (запятая, таб, пробел, ;)
        df = pd.read_csv(file_path, sep=None, engine='python', comment='#')
        # Если есть названия столбцов X/Y (в любом регистре), используем их
        cols_lower = {c.lower(): c for c in df.columns}
        if 'x' in cols_lower and 'y' in cols_lower:
            xy = df[[cols_lower['x'], cols_lower['y']]].to_numpy(dtype=float)
        else:
            # Иначе берем первые два столбца
            if df.shape[1] < 2:
                return np.empty((0, 2))
            xy = df.iloc[:, :2].to_numpy(dtype=float)
        if xy.ndim == 1:
            xy = xy.reshape(1, -1)
        if xy.shape[1] >= 2:
            return xy[:, :2]
        return np.empty((0, 2))
    except Exception:
        # Фоллбек к numpy.loadtxt с авто-разделителем по пробелам/табам
        try:
            data = np.loadtxt(file_path)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] >= 2:
                return data[:, :2]
            return np.empty((0, 2))
        except Exception:
            return np.empty((0, 2))
