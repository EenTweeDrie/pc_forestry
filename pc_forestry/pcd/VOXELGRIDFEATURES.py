from .VOXEL import VOXEL
from .PCD import PCD
import numpy as np
import pandas as pd
from tqdm import tqdm
from .features import registry as _feature_registry  # реестр фич
from . import features as _features_pkg  # noqa: F401 - подтягивает встроенные фичи (регистрация по импорту)
# Явно подтягиваем модуль builtin, чтобы гарантировать регистрацию всех встроенных признаков
from .features import builtin as _builtin  # noqa: F401


class VOXELGRIDFEATURES:
    """
    Простой и быстрый класс для работы с вокселями и признаками.
    - Быстро переводит точки из PCD в воксельную структуру и обратно
    - Считает фичи через внешний реестр фич (плагины)
    Добавление новой фичи не требует изменений в этом классе.
    """

    def __init__(self, PC: PCD, voxel_size: float, voxels: list[VOXEL] = None):
        self.PC = PC
        self.voxel_size = voxel_size
        self.voxels = voxels if voxels is not None else []
        # Кэши
        self._centers = None
        self._index_array = None
        self._layer_to_indices = None
        # Групповые индексы (для векторизации по группам)
        self._inverse = None    # per-point -> voxel id
        self._counts = None     # points per voxel (len == num_voxels)
        # Вычисленные фичи (кэш результатов)
        self._features = {}      # name -> np.ndarray (N,) или (N,k)

    def __len__(self) -> int:
        return len(self.voxels)

    def __getitem__(self, index: int) -> VOXEL:
        return self.voxels[index]

    @classmethod
    def from_pcd(cls, PC: PCD, voxel_size: float, verbose: bool = False):
        voxel_indices = np.floor(PC.points / voxel_size).astype(np.int32)
        unique_indices, inverse, counts = np.unique(
            voxel_indices, axis=0, return_inverse=True, return_counts=True
        )
        order = np.argsort(inverse, kind='mergesort')
        splits = np.cumsum(counts)[:-1]
        groups = np.split(order, splits)

        voxels = []
        has_intensity = PC.intensity.size > 0
        has_rgb = PC.rgb.size > 0
        has_oci = PC.original_cloud_index.size > 0
        has_gps = PC.gps_time.size > 0
        has_illum_ray = PC.illuminance_ray.size > 0
        has_illum_pcv = PC.illuminance_pcv.size > 0
        has_illum_cc = PC.illuminance_cc.size > 0
        has_normals = PC.normals.size > 0

        iterator = range(len(groups))
        if verbose:
            iterator = tqdm(iterator, desc="Creating voxel grid (fast)")

        for i in iterator:
            idx_tuple = tuple(int(v) for v in unique_indices[i])
            sel = groups[i]
            voxel = VOXEL(idx_tuple)
            voxel.points = PC.points[sel]
            if has_intensity:
                voxel.intensity = PC.intensity[sel]
            if has_rgb:
                voxel.rgb = PC.rgb[sel]
            if has_oci:
                voxel.original_cloud_index = PC.original_cloud_index[sel]
            if has_gps:
                voxel.gps_time = PC.gps_time[sel]
            if has_illum_ray:
                voxel.illuminance_ray = PC.illuminance_ray[sel]
            if has_illum_pcv:
                voxel.illuminance_pcv = PC.illuminance_pcv[sel]
            if has_illum_cc:
                voxel.illuminance_cc = PC.illuminance_cc[sel]
            if has_normals:
                voxel.normals = PC.normals[sel]
            voxels.append(voxel)

        inst = cls(PC, voxel_size, voxels)
        inst._inverse = inverse
        inst._counts = counts.astype(np.int64)
        return inst

    # -------- Кэш индексов/центров/слоёв --------
    @property
    def index_array(self) -> np.ndarray:
        if self._index_array is None:
            self._index_array = np.array([v.index for v in self.voxels], dtype=np.int32)
        return self._index_array

    @property
    def centers(self) -> np.ndarray:
        if self._centers is None:
            self._centers = (self.index_array.astype(np.float64) + 0.5) * float(self.voxel_size)
        return self._centers

    @property
    def layer_to_indices(self) -> dict:
        if self._layer_to_indices is None:
            z = self.index_array[:, 2]
            self._layer_to_indices = {}
            for layer in np.unique(z):
                self._layer_to_indices[int(layer)] = np.where(z == layer)[0]
        return self._layer_to_indices

    # -------- Обратные преобразования --------
    def to_pcd(self) -> PCD:
        """Собрать исходный PCD из точек всех вокселей."""
        pcd = PCD()
        if len(self.voxels) == 0:
            return pcd
        # Собираем списки массивов и конкатенируем разом (быстрее, чем append в цикле)
        points_list = [v.points for v in self.voxels if getattr(v, 'points', None) is not None and v.points.size > 0]
        pcd.points = np.concatenate(points_list, axis=0) if points_list else np.empty((0, 3), dtype=float)

        if getattr(self.voxels[0], 'intensity', None) is not None:
            intensity_list = [v.intensity for v in self.voxels if v.intensity.size > 0]
            pcd.intensity = np.concatenate(intensity_list, axis=0) if intensity_list else np.empty((0,), dtype=float)
        if getattr(self.voxels[0], 'rgb', None) is not None:
            rgb_list = [v.rgb for v in self.voxels if v.rgb.size > 0]
            pcd.rgb = np.concatenate(rgb_list, axis=0) if rgb_list else np.empty((0, 3), dtype=float)
        if getattr(self.voxels[0], 'original_cloud_index', None) is not None:
            oci_list = [v.original_cloud_index for v in self.voxels if v.original_cloud_index.size > 0]
            pcd.original_cloud_index = np.concatenate(oci_list, axis=0) if oci_list else np.empty((0,), dtype=float)
        if getattr(self.voxels[0], 'gps_time', None) is not None:
            gps_list = [v.gps_time for v in self.voxels if v.gps_time.size > 0]
            pcd.gps_time = np.concatenate(gps_list, axis=0) if gps_list else np.empty((0,), dtype=float)
        if getattr(self.voxels[0], 'illuminance', None) is not None:
            illum_list = [v.illuminance for v in self.voxels if v.illuminance.size > 0]
            pcd.illuminance = np.concatenate(illum_list, axis=0) if illum_list else np.empty((0,), dtype=float)
        if getattr(self.voxels[0], 'normals', None) is not None:
            normals_list = [v.normals for v in self.voxels if v.normals.size > 0]
            pcd.normals = np.concatenate(normals_list, axis=0) if normals_list else np.empty((0, 3), dtype=float)
        return pcd

    def to_pcd_centers(self, color_field: str = 'intensity') -> PCD:
        """PCD из центров вокселей, серый цвет по выбранному полю."""
        pcd = PCD()
        pcd.points = self.centers.copy()
        if len(self.voxels) > 0 and hasattr(self.voxels[0], color_field):
            values = []
            for voxel in self.voxels:
                val = getattr(voxel, color_field)
                if isinstance(val, np.ndarray) and val.ndim > 0:
                    values.append(val.mean() if val.size > 0 else 0.0)
                else:
                    values.append(float(val) if val is not None else 0.0)
            values = np.asarray(values, dtype=np.float64)
            if values.size > 0:
                vmin, vmax = float(values.min()), float(values.max())
                denom = (vmax - vmin) if (vmax - vmin) > 1e-12 else 1.0
                norm = (values - vmin) / denom
                pcd.rgb = np.stack([norm, norm, norm], axis=1)
        return pcd

    def available_features(self) -> list[str]:
        """Список доступных фич по именам (из внешнего реестра)."""
        return _feature_registry.names()

    def feature_info(self) -> dict:
        """Метаданные фич: размерность и описание (из внешнего реестра)."""
        return {name: {"dim": feat.dim, "doc": getattr(feat, "doc", "")}
                for name, feat in _feature_registry.items()}

    def compute_features(self, names: list[str] | None = None, apply_to_voxels: bool = True, force_recompute: bool = False, **kwargs) -> dict:
        """Вычислить фичи из внешнего реестра и (опционально) сохранить в атрибуты вокселей."""
        if names is None:
            names = _feature_registry.names()
        results = {}

        def _set_voxel_attr_safe(voxel: VOXEL, attr_name: str, value):
            try:
                setattr(voxel, attr_name, value)
            except AttributeError:
                setattr(voxel, f"feature_{attr_name}", value)

        for name in names:
            if not force_recompute and name in self._features:
                arr = self._features[name]
                feat = _feature_registry.get(name)
                dim = int(getattr(feat, "dim", 1))
            else:
                feat = _feature_registry.get(name)
                arr = feat.compute(self, **kwargs)
                arr = np.asarray(arr)
                dim = int(getattr(feat, "dim", 1))
                if dim == 1:
                    if arr.shape[0] != len(self):
                        raise ValueError(f"Неверная форма фичи '{name}', ожидалось (N,), получено {arr.shape}")
                    arr = arr.reshape(-1)
                else:
                    if arr.shape != (len(self), dim):
                        raise ValueError(f"Неверная форма фичи '{name}', ожидалось (N,{dim}), получено {arr.shape}")
                self._features[name] = arr
            results[name] = arr
            if apply_to_voxels:
                if dim == 1:
                    for i, voxel in enumerate(self.voxels):
                        _set_voxel_attr_safe(voxel, name, arr[i])
                else:
                    for i, voxel in enumerate(self.voxels):
                        _set_voxel_attr_safe(voxel, name, arr[i])
        return results

    def get_feature_matrix(self, names: list[str]) -> np.ndarray:
        """Матрица признаков (N, sum(dims)). Вычисляет недостающие по мере необходимости."""
        parts = []
        for name in names:
            if name not in self._features:
                _ = _feature_registry.get(name)  # проверка наличия
                self.compute_features([name])
            arr = self._features[name]
            if arr.ndim == 1:
                parts.append(arr.reshape(-1, 1))
            else:
                parts.append(arr)
        if not parts:
            return np.zeros((len(self), 0), dtype=np.float64)
        return np.concatenate(parts, axis=1)

    def get_features_df(self, names: list[str]) -> pd.DataFrame:
        """DataFrame по выбранным признакам + индексы вокселей."""
        mat = self.get_feature_matrix(names)
        cols = []
        for name in names:
            dim = _feature_registry.get(name).dim
            if dim == 1:
                cols.append(name)
            else:
                cols.extend([f"{name}_{i}" for i in range(dim)])
        df = pd.DataFrame(mat, columns=cols)
        idx = self.index_array
        df.insert(0, 'z', idx[:, 2])
        df.insert(0, 'y', idx[:, 1])
        df.insert(0, 'x', idx[:, 0])
        return df

    def get_labels(self) -> np.ndarray:
        """Вектор целевых меток (label) длины N (число вокселей)."""
        self.compute_features(['label'], apply_to_voxels=False)
        return self._features['label'].astype(np.int64, copy=False)

    # -------- Расчёт фич только для выбранного слоя Z --------
    def compute_features_for_layer(self, z_layer: int, names: list[str] | None = None,
                                   apply_to_voxels: bool = True, **kwargs) -> dict:
        """
        Вычислить фичи ТОЛЬКО для вокселей слоя z_layer.
        Плагины считают фичи на весь массив, после чего берётся срез нужного слоя.
        Возвращает dict: имя -> массив значений для слоя.
        """
        if names is None:
            names = _feature_registry.names()
        idx = self.layer_to_indices.get(int(z_layer), np.array([], dtype=np.int64))
        results = {}
        if idx.size == 0:
            # вернуть пустые массивы корректной формы
            for name in names:
                feat = _feature_registry.get(name)
                dim = int(getattr(feat, "dim", 1))
                if dim == 1:
                    results[name] = np.zeros((0,), dtype=np.float64)
                else:
                    results[name] = np.zeros((0, dim), dtype=np.float64)
            return results

        # Разделим фичи на статические (считаем на весь объём с кэшем) и динамические (только текущий слой)
        dynamic_features = kwargs.get("dynamic_features", []) or []
        dynamic_set = set(dynamic_features)
        static_names = [n for n in names if n not in dynamic_set]
        dyn_names = [n for n in names if n in dynamic_set]

        results_full = {}
        if static_names:
            # Считаем и кэшируем недостающие фичи полностью, затем режем по слою
            results_full = self.compute_features(static_names, apply_to_voxels=False, **kwargs)

        def _set_voxel_attr_safe(voxel: VOXEL, attr_name: str, value):
            try:
                setattr(voxel, attr_name, value)
            except AttributeError:
                setattr(voxel, f"feature_{attr_name}", value)

        # Статические фичи: берём срез слоя
        for name in static_names:
            arr = results_full[name]
            if arr.ndim == 1:
                part = arr[idx]
            else:
                part = arr[idx, :]
            results[name] = part
            if apply_to_voxels:
                if arr.ndim == 1:
                    for j, i_vox in enumerate(idx):
                        _set_voxel_attr_safe(self.voxels[i_vox], name, part[j])
                else:
                    for j, i_vox in enumerate(idx):
                        _set_voxel_attr_safe(self.voxels[i_vox], name, part[j])

        # Динамические фичи: считаем только для текущего слоя
        if dyn_names:
            from .features import registry as _feature_registry  # локальный импорт уже есть выше
            for name in dyn_names:
                feat = _feature_registry.get(name)
                # compute может поддерживать current_layer/restrict_to_layer
                arr_cur = feat.compute(self,
                                       current_layer=int(z_layer),
                                       restrict_to_layer=True,
                                       **kwargs)
                arr_cur = np.asarray(arr_cur)
                dim = int(getattr(feat, "dim", 1))
                if dim == 1:
                    arr_cur = arr_cur.reshape(-1)
                else:
                    if arr_cur.ndim == 1:
                        # допустим, плагин вернул (M,) для многомерной — приведём к (M, dim) нулями
                        tmp = np.zeros((arr_cur.shape[0], dim), dtype=arr_cur.dtype)
                        tmp[:, 0] = arr_cur
                        arr_cur = tmp
                results[name] = arr_cur
                if apply_to_voxels:
                    if dim == 1:
                        for j, i_vox in enumerate(idx):
                            _set_voxel_attr_safe(self.voxels[i_vox], name, arr_cur[j])
                    else:
                        for j, i_vox in enumerate(idx):
                            _set_voxel_attr_safe(self.voxels[i_vox], name, arr_cur[j])
        return results

    def get_feature_matrix_for_layer(self, names: list[str], z_layer: int, **kwargs) -> np.ndarray:
        """
        Матрица признаков по выбранному слою (M, sum(dims)), где M — число вокселей в слое.
        """
        parts = []
        # Индексы вокселей текущего слоя
        idx = self.layer_to_indices.get(int(z_layer), np.array([], dtype=np.int64))
        if idx.size == 0:
            return np.zeros((0, 0), dtype=np.float64)
        layer_vals = self.compute_features_for_layer(z_layer, names, apply_to_voxels=False, **kwargs)
        for name in names:
            arr = layer_vals[name]
            # Защита от плагинов, вернувших массив длины N вместо M (режем по слою)
            if arr.ndim == 1:
                if arr.shape[0] != idx.size and arr.shape[0] == len(self):
                    arr = arr[idx]
                parts.append(arr.reshape(-1, 1))
            else:
                if arr.shape[0] != idx.size and arr.shape[0] == len(self):
                    arr = arr[idx, :]
                parts.append(arr)
        if not parts:
            return np.zeros((0, 0), dtype=np.float64)
        return np.concatenate(parts, axis=1)

    def get_features_df_for_layer(self, names: list[str], z_layer: int, **kwargs) -> pd.DataFrame:
        """
        DataFrame по выбранным признакам только для заданного слоя.
        Содержит колонки x,y,z и выбранные фичи.
        """
        idx = self.layer_to_indices.get(int(z_layer), np.array([], dtype=np.int64))
        if idx.size == 0:
            # пустой слой
            cols = []
            for name in names:
                dim = _feature_registry.get(name).dim
                if dim == 1:
                    cols.append(name)
                else:
                    cols.extend([f"{name}_{i}" for i in range(dim)])
            return pd.DataFrame(columns=["x", "y", "z"] + cols)

        mat = self.get_feature_matrix_for_layer(names, z_layer, **kwargs)
        cols = []
        for name in names:
            dim = _feature_registry.get(name).dim
            if dim == 1:
                cols.append(name)
            else:
                cols.extend([f"{name}_{i}" for i in range(dim)])
        df = pd.DataFrame(mat, columns=cols)
        idx_all = self.index_array
        df.insert(0, 'z', idx_all[idx, 2])
        df.insert(0, 'y', idx_all[idx, 1])
        df.insert(0, 'x', idx_all[idx, 0])
        return df

    # @staticmethod
    # def from_df(df: pd.DataFrame, voxel_size: float) -> 'VOXELGRIDFEATURES':
    #     """
    #     Создать VOXELGRIDFEATURES из DataFrame.
    #     """
    #     voxels = []
    #     for i in range(len(df)):
    #         idx_tuple = (int(df.iloc[i]['x']), int(df.iloc[i]['y']), int(df.iloc[i]['z']))
    #         voxel = VOXEL(idx_tuple)
    #         voxels.append(voxel)
    #     return VOXELGRIDFEATURES(None, voxel_size, voxels)
