from __future__ import annotations

import numpy as np
from typing import Any
from .base import VoxelFeature
from .registry import register_feature, registry


def _safe_counts(grid) -> np.ndarray:
    if getattr(grid, "_counts", None) is None:
        return np.zeros(len(grid), dtype=np.int64)
    return np.maximum(grid._counts, 1)


def _mode_per_voxel_int(
    inverse: np.ndarray,
    values: np.ndarray,
    num_voxels: int,
    *,
    default: int = 0,
) -> np.ndarray:
    """
    Векторизованный подсчёт "моды" (самого частого значения) для каждой группы.

    Вход:
    - inverse: (N,) int, voxel-id для каждой точки
    - values: (N,) int, значение для каждой точки (например, label/mask)
    - num_voxels: число вокселей (len(grid))

    Выход:
    - (num_voxels,) int64 с наиболее частым значением по каждой группе.

    Реализовано через сортировку по (voxel_id, value) и run-length, без Python-циклов по вокселям.
    """
    inv = np.asarray(inverse)
    vals = np.asarray(values)
    if inv.size == 0 or vals.size == 0 or num_voxels <= 0:
        return np.full((int(num_voxels),), int(default), dtype=np.int64)

    inv = inv.astype(np.int64, copy=False).ravel()
    vals = vals.astype(np.int64, copy=False).ravel()
    n = inv.shape[0]
    if vals.shape[0] != n:
        raise ValueError("inverse и values должны иметь одинаковую длину")

    # Упорядочим точки по (voxel_id, value)
    order = np.lexsort((vals, inv))
    inv_s = inv[order]
    val_s = vals[order]

    # Границы "ранов" где меняется (voxel_id,value)
    # run_start[0]=0, дальше там, где пара отличается от предыдущей
    diff = (inv_s[1:] != inv_s[:-1]) | (val_s[1:] != val_s[:-1])
    run_starts = np.concatenate([np.array([0], dtype=np.int64), np.nonzero(diff)[0].astype(np.int64) + 1])
    run_ends = np.concatenate([run_starts[1:], np.array([n], dtype=np.int64)])
    run_counts = (run_ends - run_starts).astype(np.int64, copy=False)
    run_vox = inv_s[run_starts]
    run_val = val_s[run_starts]

    # Выберем для каждого voxel run с максимальным count.
    # Сортируем раны по (voxel_id, -count), и берём первый для каждого voxel_id.
    order_runs = np.lexsort((run_val, -run_counts, run_vox))
    run_vox_o = run_vox[order_runs]
    run_val_o = run_val[order_runs]
    # индексы первого вхождения каждого voxel_id в order_runs
    _, first_idx = np.unique(run_vox_o, return_index=True)
    vox_ids = run_vox_o[first_idx]
    modes = run_val_o[first_idx]

    out = np.full((int(num_voxels),), int(default), dtype=np.int64)
    # Защитимся от мусорных voxel id
    m = (vox_ids >= 0) & (vox_ids < num_voxels)
    out[vox_ids[m]] = modes[m]
    return out


class NumPoints(VoxelFeature):
    name = "num_points"
    dim = 1
    doc = "Количество точек в вокселе"

    def compute(self, grid, **kwargs) -> np.ndarray:
        if getattr(grid, "_counts", None) is None:
            return np.zeros(len(grid), dtype=np.float64)
        return grid._counts.astype(np.float64, copy=False)


class MeanIntensity(VoxelFeature):
    name = "mean_intensity"
    dim = 1
    doc = "Средняя интенсивность по вокселю"

    def compute(self, grid, **kwargs) -> np.ndarray:
        if grid.PC.intensity.size == 0 or grid._inverse is None or grid._counts is None:
            return np.zeros(len(grid), dtype=np.float64)
        sums = np.bincount(grid._inverse, weights=grid.PC.intensity, minlength=len(grid))
        means = sums / _safe_counts(grid)
        return means.astype(np.float64, copy=False)


class StdIntensity(VoxelFeature):
    name = "std_intensity"
    dim = 1
    doc = "Ст. отклонение интенсивности по вокселю"

    def compute(self, grid, **kwargs) -> np.ndarray:
        if grid.PC.intensity.size == 0 or grid._inverse is None or grid._counts is None:
            return np.zeros(len(grid), dtype=np.float64)
        counts = _safe_counts(grid)
        x = grid.PC.intensity
        sum_x = np.bincount(grid._inverse, weights=x, minlength=len(grid))
        sum_x2 = np.bincount(grid._inverse, weights=x * x, minlength=len(grid))
        mean_x = sum_x / counts
        mean_x2 = sum_x2 / counts
        var = np.clip(mean_x2 - mean_x * mean_x, 0.0, None)
        return np.sqrt(var, dtype=np.float64)


class MeanRGB(VoxelFeature):
    name = "mean_rgb"
    dim = 3
    doc = "Средний RGB по вокселю"

    def compute(self, grid, **kwargs) -> np.ndarray:
        out = np.zeros((len(grid), 3), dtype=np.float64)
        if grid.PC.rgb.size == 0 or grid._inverse is None or grid._counts is None:
            return out
        counts = _safe_counts(grid)
        for ch in range(3):
            sums = np.bincount(grid._inverse, weights=grid.PC.rgb[:, ch], minlength=len(grid))
            out[:, ch] = sums / counts
        return out


class StdRGB(VoxelFeature):
    name = "std_rgb"
    dim = 3
    doc = "Ст. отклонение RGB по вокселю"

    def compute(self, grid, **kwargs) -> np.ndarray:
        out = np.zeros((len(grid), 3), dtype=np.float64)
        if grid.PC.rgb.size == 0 or grid._inverse is None or grid._counts is None:
            return out
        counts = _safe_counts(grid)
        for ch in range(3):
            x = grid.PC.rgb[:, ch]
            sum_x = np.bincount(grid._inverse, weights=x, minlength=len(grid))
            sum_x2 = np.bincount(grid._inverse, weights=x * x, minlength=len(grid))
            mean_x = sum_x / counts
            mean_x2 = sum_x2 / counts
            var = np.clip(mean_x2 - mean_x * mean_x, 0.0, None)
            out[:, ch] = np.sqrt(var, dtype=np.float64)
        return out


class MeanChromaticity(VoxelFeature):
    name = "mean_chromaticity"
    dim = 3
    doc = "Средняя хроматичность RGB по вокселю"

    def compute(self, grid, **kwargs) -> np.ndarray:
        out = np.zeros((len(grid), 3), dtype=np.float64)
        if grid.PC.rgb.size == 0 or grid._inverse is None or grid._counts is None:
            return out
        rgb = grid.PC.rgb
        s = rgb.sum(axis=1)
        s[s == 0] = 1.0
        chrom = rgb / s[:, None]
        counts = _safe_counts(grid)
        for ch in range(3):
            sums = np.bincount(grid._inverse, weights=chrom[:, ch], minlength=len(grid))
            out[:, ch] = sums / counts
        return out


class MeanNormals(VoxelFeature):
    name = "mean_normals"
    dim = 3
    doc = "Средние компоненты нормали (nx, ny, nz)"

    def compute(self, grid, **kwargs) -> np.ndarray:
        out = np.zeros((len(grid), 3), dtype=np.float64)
        if grid.PC.normals.size == 0 or grid._inverse is None or grid._counts is None:
            return out
        counts = _safe_counts(grid)
        for ch in range(3):
            sums = np.bincount(grid._inverse, weights=grid.PC.normals[:, ch], minlength=len(grid))
            out[:, ch] = sums / counts
        return out


class MeanNormalsX(VoxelFeature):
    name = "mean_normals_x"
    dim = 1
    doc = "Средний nx"

    def compute(self, grid, **kwargs) -> np.ndarray:
        arr = grid._features.get("mean_normals")
        if arr is None:
            arr = registry.get("mean_normals").compute(grid)
            grid._features["mean_normals"] = arr
        return arr[:, 0]


class MeanNormalsY(VoxelFeature):
    name = "mean_normals_y"
    dim = 1
    doc = "Средний ny"

    def compute(self, grid, **kwargs) -> np.ndarray:
        arr = grid._features.get("mean_normals")
        if arr is None:
            arr = registry.get("mean_normals").compute(grid)
            grid._features["mean_normals"] = arr
        return arr[:, 1]


class MeanNormalsZ(VoxelFeature):
    name = "mean_normals_z"
    dim = 1
    doc = "Средний nz"

    def compute(self, grid, **kwargs) -> np.ndarray:
        arr = grid._features.get("mean_normals")
        if arr is None:
            arr = registry.get("mean_normals").compute(grid)
            grid._features["mean_normals"] = arr
        return arr[:, 2]


class MeanIlluminanceRay(VoxelFeature):
    name = "mean_illuminance_ray"
    dim = 1
    doc = "Средняя освещённость (RAY) по вокселю"

    def compute(self, grid, **kwargs) -> np.ndarray:
        if grid.PC.illuminance_ray.size == 0 or grid._inverse is None or grid._counts is None:
            return np.zeros(len(grid), dtype=np.float64)
        sums = np.bincount(grid._inverse, weights=grid.PC.illuminance_ray, minlength=len(grid))
        means = sums / _safe_counts(grid)
        return means.astype(np.float64, copy=False)


class MeanIlluminancePCV(VoxelFeature):
    name = "mean_illuminance_pcv"
    dim = 1
    doc = "Средняя освещённость (PCV) по вокселю"

    def compute(self, grid, **kwargs) -> np.ndarray:
        if grid.PC.illuminance_pcv.size == 0 or grid._inverse is None or grid._counts is None:
            return np.zeros(len(grid), dtype=np.float64)
        sums = np.bincount(grid._inverse, weights=grid.PC.illuminance_pcv, minlength=len(grid))
        means = sums / _safe_counts(grid)
        return means.astype(np.float64, copy=False)


class MeanIlluminanceCC(VoxelFeature):
    name = "mean_illuminance_cc"
    dim = 1
    doc = "Средняя освещённость (CC) по вокселю"

    def compute(self, grid, **kwargs) -> np.ndarray:
        if grid.PC.illuminance_cc.size == 0 or grid._inverse is None or grid._counts is None:
            return np.zeros(len(grid), dtype=np.float64)
        sums = np.bincount(grid._inverse, weights=grid.PC.illuminance_cc, minlength=len(grid))
        means = sums / _safe_counts(grid)
        return means.astype(np.float64, copy=False)


class MeanGpsTime(VoxelFeature):
    name = "mean_gps_time"
    dim = 1
    doc = "Среднее GPS-время по вокселю"

    def compute(self, grid, **kwargs) -> np.ndarray:
        if grid.PC.gps_time.size == 0 or grid._inverse is None or grid._counts is None:
            return np.zeros(len(grid), dtype=np.float64)
        sums = np.bincount(grid._inverse, weights=grid.PC.gps_time, minlength=len(grid))
        means = sums / _safe_counts(grid)
        return means.astype(np.float64, copy=False)


class CenterCoords(VoxelFeature):
    name = "center_coords"
    dim = 3
    doc = "Координаты центра вокселя (x,y,z)"

    def compute(self, grid, **kwargs) -> np.ndarray:
        return grid.centers.astype(np.float64, copy=False)


class HeightNorm(VoxelFeature):
    name = "height_norm"
    dim = 1
    doc = "Нормированная высота центра вокселя в [0,1]"

    def compute(self, grid, **kwargs) -> np.ndarray:
        z = grid.centers[:, 2]
        z_min = float(z.min()) if z.size > 0 else 0.0
        z_max = float(z.max()) if z.size > 0 else 1.0
        denom = (z_max - z_min) if (z_max - z_min) > 1e-12 else 1.0
        return (z - z_min) / denom


class MeanAbsNz(VoxelFeature):
    name = "mean_abs_nz"
    dim = 1
    doc = "Среднее |nz| по нормалям точек в вокселе"

    def compute(self, grid, **kwargs) -> np.ndarray:
        if grid.PC.normals.size == 0 or grid._inverse is None or grid._counts is None:
            return np.zeros(len(grid), dtype=np.float64)
        abs_nz = np.abs(grid.PC.normals[:, 2])
        sums = np.bincount(grid._inverse, weights=abs_nz, minlength=len(grid))
        means = sums / _safe_counts(grid)
        return means.astype(np.float64, copy=False)


class MeanAbsNy(VoxelFeature):
    name = "mean_abs_ny"
    dim = 1
    doc = "Среднее |ny| по нормалям точек в вокселе"

    def compute(self, grid, **kwargs) -> np.ndarray:
        if grid.PC.normals.size == 0 or grid._inverse is None or grid._counts is None:
            return np.zeros(len(grid), dtype=np.float64)
        abs_ny = np.abs(grid.PC.normals[:, 1])
        sums = np.bincount(grid._inverse, weights=abs_ny, minlength=len(grid))
        means = sums / _safe_counts(grid)
        return means.astype(np.float64, copy=False)


class MeanAbsNx(VoxelFeature):
    name = "mean_abs_nx"
    dim = 1
    doc = "Среднее |nx| по нормалям точек в вокселе"

    def compute(self, grid, **kwargs) -> np.ndarray:
        if grid.PC.normals.size == 0 or grid._inverse is None or grid._counts is None:
            return np.zeros(len(grid), dtype=np.float64)
        abs_nx = np.abs(grid.PC.normals[:, 0])
        sums = np.bincount(grid._inverse, weights=abs_nx, minlength=len(grid))
        means = sums / _safe_counts(grid)
        return means.astype(np.float64, copy=False)


class DistanceToCoord(VoxelFeature):
    name = "distance_to_coord"
    dim = 1
    doc = "Мин. 3D расстояние до ближайшей из заданных координат"

    def compute(self, grid, **kwargs) -> np.ndarray:
        coords = kwargs.get("coordinates", None)
        coordinate = kwargs.get("coordinate", None)
        if coords is None:
            if coordinate is None:
                return np.zeros(len(grid), dtype=np.float64)
            if np.ndim(coordinate) == 1:
                c = np.asarray([
                    coordinate[0],
                    coordinate[1],
                    coordinate[2] if len(coordinate) > 2 else 0.0
                ], dtype=np.float64)
                diff = grid.centers - c.reshape(1, 3)
                return np.linalg.norm(diff, axis=1)
            else:
                coords = coordinate
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        if coords.shape[1] == 2:
            coords = np.concatenate([coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)], axis=1)
        # Быстрая ветка: cKDTree если доступен
        try:
            from scipy.spatial import cKDTree  # type: ignore
            tree = cKDTree(coords)
            dist, _ = tree.query(grid.centers, k=1, workers=-1)
            return dist.astype(np.float64, copy=False)
        except Exception:
            try:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
                nn.fit(coords)
                dist, _ = nn.kneighbors(grid.centers, return_distance=True)
                return dist.reshape(-1)
            except Exception:
                d = np.linalg.norm(grid.centers[:, None, :] - coords[None, :, :], axis=2)
                return np.min(d, axis=1) if d.size > 0 else np.zeros(len(grid), dtype=np.float64)


class DistanceToCoordXY(VoxelFeature):
    name = "distance_to_coord_XY"
    dim = 1
    doc = "Мин. 2D (XY) расстояние до ближайшей координаты"

    def compute(self, grid, **kwargs) -> np.ndarray:
        coords = kwargs.get("coordinates", None)
        coordinate = kwargs.get("coordinate", None)
        if coords is None:
            if coordinate is None:
                return np.zeros(len(grid), dtype=np.float64)
            if np.ndim(coordinate) == 1:
                c = np.asarray([coordinate[0], coordinate[1]], dtype=np.float64)
                diff = grid.centers[:, :2] - c.reshape(1, 2)
                return np.linalg.norm(diff, axis=1)
            else:
                coords = coordinate
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        if coords.shape[1] >= 2:
            coords_xy = coords[:, :2]
        else:
            return np.zeros(len(grid), dtype=np.float64)
        try:
            from scipy.spatial import cKDTree  # type: ignore
            tree = cKDTree(coords_xy)
            dist, _ = tree.query(grid.centers[:, :2], k=1, workers=-1)
            return dist.astype(np.float64, copy=False)
        except Exception:
            try:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
                nn.fit(coords_xy)
                dist, _ = nn.kneighbors(grid.centers[:, :2], return_distance=True)
                return dist.reshape(-1)
            except Exception:
                d = np.linalg.norm(grid.centers[:, None, :2] - coords_xy[None, :, :], axis=2)
                return np.min(d, axis=1) if d.size > 0 else np.zeros(len(grid), dtype=np.float64)


class Label(VoxelFeature):
    name = "label"
    dim = 1
    doc = "Мода original_cloud_index по точкам вокселя"

    def compute(self, grid, **kwargs) -> np.ndarray:
        if grid.PC.original_cloud_index.size == 0 or grid._inverse is None:
            return np.zeros(len(grid), dtype=np.int64)
        return _mode_per_voxel_int(grid._inverse, grid.PC.original_cloud_index, len(grid), default=0)


class NXFilteringMask(VoxelFeature):
    name = "nx_filtering_mask"
    dim = 1
    doc = "Мода nx_filtering_mask по точкам вокселя"

    def compute(self, grid, **kwargs) -> np.ndarray:
        if grid.PC.nx_filtering_mask.size == 0 or grid._inverse is None:
            return np.zeros(len(grid), dtype=np.int64)
        return _mode_per_voxel_int(grid._inverse, grid.PC.nx_filtering_mask, len(grid), default=0)


class NYFilteringMask(VoxelFeature):
    name = "ny_filtering_mask"
    dim = 1
    doc = "Мода ny_filtering_mask по точкам вокселя"

    def compute(self, grid, **kwargs) -> np.ndarray:
        if grid.PC.ny_filtering_mask.size == 0 or grid._inverse is None:
            return np.zeros(len(grid), dtype=np.int64)
        return _mode_per_voxel_int(grid._inverse, grid.PC.ny_filtering_mask, len(grid), default=0)


class NZFilteringMask(VoxelFeature):
    name = "nz_filtering_mask"
    dim = 1
    doc = "Мода nz_filtering_mask по точкам вокселя"

    def compute(self, grid, **kwargs) -> np.ndarray:
        if grid.PC.nz_filtering_mask.size == 0 or grid._inverse is None:
            return np.zeros(len(grid), dtype=np.int64)
        return _mode_per_voxel_int(grid._inverse, grid.PC.nz_filtering_mask, len(grid), default=0)


class NFilteringMask(VoxelFeature):
    name = "n_filtering_mask"
    dim = 1
    doc = "Мода n_filtering_mask по точкам вокселя"

    def compute(self, grid, **kwargs) -> np.ndarray:
        if grid.PC.n_filtering_mask.size == 0 or grid._inverse is None:
            return np.zeros(len(grid), dtype=np.int64)
        return _mode_per_voxel_int(grid._inverse, grid.PC.n_filtering_mask, len(grid), default=0)


class ExpandFilteringMask(VoxelFeature):
    name = "expand_filtering_mask"
    dim = 1
    doc = "Мода expand_filtering_mask по точкам вокселя"

    def compute(self, grid, **kwargs) -> np.ndarray:
        if grid.PC.expand_filtering_mask.size == 0 or grid._inverse is None:
            return np.zeros(len(grid), dtype=np.int64)
        return _mode_per_voxel_int(grid._inverse, grid.PC.expand_filtering_mask, len(grid), default=0)


class DistanceToPrevLayer(VoxelFeature):
    name = "distance_to_prev_layer"
    dim = 1
    doc = "3D до ближайшего 'положительного' в нижних слоях (mask или label)"

    def compute(self, grid, **kwargs) -> np.ndarray:
        centers = grid.centers
        z = grid.index_array[:, 2]
        current_layer = kwargs.get("current_layer", None)
        restrict = kwargs.get("restrict_to_layer", False)
        prev_pred_mask = kwargs.get("prev_pred_mask", None)

        # Кандидаты: предсказанные ранее (или fallback label==0)
        if prev_pred_mask is not None:
            cand_mask_global = np.asarray(prev_pred_mask, dtype=bool)
        else:
            labels = grid._features.get("label")
            if labels is None:
                labels = registry.get("label").compute(grid)
                grid._features["label"] = labels
            cand_mask_global = (labels == 0)

        # Быстрый путь: считать только для текущего слоя
        if restrict and current_layer is not None:
            layer = int(current_layer)
            idx_cur = grid.layer_to_indices.get(layer, np.array([], dtype=np.int64))
            if idx_cur.size == 0:
                return np.zeros((0,), dtype=np.float64)
            mask_cand = (z < layer) & cand_mask_global
            idx_cand = np.where(mask_cand)[0]
            if idx_cand.size == 0:
                # базовый слой или нет кандидатов — нули
                return np.zeros((idx_cur.size,), dtype=np.float64)
            cur = centers[idx_cur]
            cand = centers[idx_cand]
            try:
                from scipy.spatial import cKDTree  # type: ignore
                tree = cKDTree(cand)
                dist, _ = tree.query(cur, k=1, workers=-1)
                return dist.astype(np.float64, copy=False)
            except Exception:
                try:
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
                    nn.fit(cand)
                    dist, _ = nn.kneighbors(cur, return_distance=True)
                    return dist.reshape(-1)
                except Exception:
                    d = np.linalg.norm(cur[:, None, :] - cand[None, :, :], axis=2)
                    return np.min(d, axis=1) if d.size > 0 else np.zeros((idx_cur.size,), dtype=np.float64)

        # Полный расчёт на все слои
        res = np.full(len(grid), np.inf, dtype=np.float64)
        layers = sorted(grid.layer_to_indices.keys())
        z0_idx = grid.layer_to_indices.get(0, np.array([], dtype=np.int64))
        if z0_idx.size > 0:
            res[z0_idx] = 0.0
        for layer in layers:
            if layer == 0:
                continue
            idx_cur = grid.layer_to_indices[layer]
            mask_cand = (z < layer) & cand_mask_global
            idx_cand = np.where(mask_cand)[0]
            if idx_cand.size == 0 or idx_cur.size == 0:
                continue
            cur = centers[idx_cur]
            cand = centers[idx_cand]
            try:
                from scipy.spatial import cKDTree  # type: ignore
                tree = cKDTree(cand)
                dist, _ = tree.query(cur, k=1, workers=-1)
                res[idx_cur] = dist.astype(np.float64, copy=False)
            except Exception:
                try:
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
                    nn.fit(cand)
                    dist, _ = nn.kneighbors(cur, return_distance=True)
                    res[idx_cur] = dist.reshape(-1)
                except Exception:
                    d = np.linalg.norm(cur[:, None, :] - cand[None, :, :], axis=2)
                    res[idx_cur] = np.min(d, axis=1)
        return res


class DistanceToPrevLayerXY(VoxelFeature):
    name = "distance_to_prev_layer_XY"
    dim = 1
    doc = "XY до ближайшего 'положительного' в нижних слоях (mask или label)"

    def compute(self, grid, **kwargs) -> np.ndarray:
        centers = grid.centers
        z = grid.index_array[:, 2]
        current_layer = kwargs.get("current_layer", None)
        restrict = kwargs.get("restrict_to_layer", False)
        prev_pred_mask = kwargs.get("prev_pred_mask", None)

        if prev_pred_mask is not None:
            cand_mask_global = np.asarray(prev_pred_mask, dtype=bool)
        else:
            labels = grid._features.get("label")
            if labels is None:
                labels = registry.get("label").compute(grid)
                grid._features["label"] = labels
            cand_mask_global = (labels == 0)

        if restrict and current_layer is not None:
            layer = int(current_layer)
            idx_cur = grid.layer_to_indices.get(layer, np.array([], dtype=np.int64))
            if idx_cur.size == 0:
                return np.zeros((0,), dtype=np.float64)
            mask_cand = (z < layer) & cand_mask_global
            idx_cand = np.where(mask_cand)[0]
            if idx_cand.size == 0:
                return np.zeros((idx_cur.size,), dtype=np.float64)
            cur = centers[idx_cur, :2]
            cand = centers[idx_cand, :2]
            try:
                from scipy.spatial import cKDTree  # type: ignore
                tree = cKDTree(cand)
                dist, _ = tree.query(cur, k=1, workers=-1)
                return dist.astype(np.float64, copy=False)
            except Exception:
                try:
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
                    nn.fit(cand)
                    dist, _ = nn.kneighbors(cur, return_distance=True)
                    return dist.reshape(-1)
                except Exception:
                    d = np.linalg.norm(cur[:, None, :] - cand[None, :, :], axis=2)
                    return np.min(d, axis=1) if d.size > 0 else np.zeros((idx_cur.size,), dtype=np.float64)

        res = np.full(len(grid), np.inf, dtype=np.float64)
        layers = sorted(grid.layer_to_indices.keys())
        z0_idx = grid.layer_to_indices.get(0, np.array([], dtype=np.int64))
        if z0_idx.size > 0:
            res[z0_idx] = 0.0
        for layer in layers:
            if layer == 0:
                continue
            idx_cur = grid.layer_to_indices[layer]
            mask_cand = (z < layer) & cand_mask_global
            idx_cand = np.where(mask_cand)[0]
            if idx_cand.size == 0 or idx_cur.size == 0:
                continue
            cur = centers[idx_cur, :2]
            cand = centers[idx_cand, :2]
            try:
                from scipy.spatial import cKDTree  # type: ignore
                tree = cKDTree(cand)
                dist, _ = tree.query(cur, k=1, workers=-1)
                res[idx_cur] = dist.astype(np.float64, copy=False)
            except Exception:
                try:
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
                    nn.fit(cand)
                    dist, _ = nn.kneighbors(cur, return_distance=True)
                    res[idx_cur] = dist.reshape(-1)
                except Exception:
                    d = np.linalg.norm(cur[:, None, :] - cand[None, :, :], axis=2)
                    res[idx_cur] = np.min(d, axis=1)
        return res


class DistanceAlias3D(VoxelFeature):
    name = "distance"
    dim = 1
    doc = "Алиас: distance_to_prev_layer"

    def compute(self, grid, **kwargs) -> np.ndarray:
        arr = grid._features.get("distance_to_prev_layer")
        if arr is None:
            arr = registry.get("distance_to_prev_layer").compute(grid, **kwargs)
            grid._features["distance_to_prev_layer"] = arr
        return arr


class DistanceAliasXY(VoxelFeature):
    name = "distance_XY"
    dim = 1
    doc = "Алиас: distance_to_prev_layer_XY"

    def compute(self, grid, **kwargs) -> np.ndarray:
        arr = grid._features.get("distance_to_prev_layer_XY")
        if arr is None:
            arr = registry.get("distance_to_prev_layer_XY").compute(grid, **kwargs)
            grid._features["distance_to_prev_layer_XY"] = arr
        return arr


# Регистрация встроенных фич (побочный эффект импортов)
register_feature(NumPoints())
register_feature(MeanIntensity())
register_feature(StdIntensity())
register_feature(MeanRGB())
register_feature(StdRGB())
register_feature(MeanChromaticity())
register_feature(MeanNormals())
register_feature(MeanNormalsX())
register_feature(MeanNormalsY())
register_feature(MeanNormalsZ())
register_feature(MeanIlluminanceRay())
register_feature(MeanIlluminancePCV())
register_feature(MeanIlluminanceCC())
register_feature(MeanGpsTime())
register_feature(CenterCoords())
register_feature(HeightNorm())
register_feature(MeanAbsNz())
register_feature(MeanAbsNy())
register_feature(MeanAbsNx())
register_feature(DistanceToCoord())
register_feature(DistanceToCoordXY())
register_feature(Label())
register_feature(DistanceToPrevLayer())
register_feature(DistanceToPrevLayerXY())
register_feature(DistanceAlias3D())
register_feature(DistanceAliasXY())
register_feature(NXFilteringMask())
register_feature(NYFilteringMask())
register_feature(NZFilteringMask())
register_feature(NFilteringMask())
register_feature(ExpandFilteringMask())
