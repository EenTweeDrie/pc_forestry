
from abc import ABC, abstractmethod
from itertools import product
import re

import numpy as np
import open3d as o3d
from loguru import logger

from ..utils import pypcd
from .illuminance.illuminance import (
    _create_voxel_grid_fast,
    _illuminance_kernel_numba,
    _illuminance_pcv_bins_numba,
    _generate_hemisphere_directions_fibonacci,
    _illuminance_pcv_dda_numba,
)
import laspy
from ..utils.timer import Timer
from .lcc import LCC
from ..utils.compare import diff_area


class Field(ABC):
    """Абстрактный базовый класс для поля облака точек."""

    _NAME_VARIANTS_CACHE = {}

    def __init__(self, data=None):
        self._data = self.initialize_data(data)

    @property
    @abstractmethod
    def name(self) -> str:
        """Имя поля (например, 'points', 'intensity')."""
        pass

    @property
    @abstractmethod
    def default_value(self) -> np.ndarray:
        """Значение по умолчанию для данных поля."""
        pass

    def initialize_data(self, data):
        if data is not None:
            return np.asarray(data)
        return self.default_value.copy()

    @staticmethod
    def _sanitize_alias(name: str) -> str:
        return re.sub(r'\W+', '_', name)

    @classmethod
    def _tokenize_name(cls, name: str):
        if not name:
            return []
        sanitized = name.replace('-', '_')
        tokens = [t for t in sanitized.split('_') if t]
        if len(tokens) <= 1:
            camel_tokens = re.findall(r'[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+', name)
            if camel_tokens:
                tokens = camel_tokens
        return tokens if tokens else [name]

    @classmethod
    def generate_name_variants(cls, base_name: str):
        if not base_name:
            return []
        if base_name in cls._NAME_VARIANTS_CACHE:
            return cls._NAME_VARIANTS_CACHE[base_name]

        variants = []
        seen = set()

        def add(value: str):
            if value and value not in seen:
                seen.add(value)
                variants.append(value)

        add(base_name)
        sanitized_base = cls._sanitize_alias(base_name)
        add(sanitized_base)

        tokens = cls._tokenize_name(base_name)
        token_forms = []
        for token in tokens:
            forms = [token, token.lower(), token.upper(), token.capitalize()]
            token_forms.append(list(dict.fromkeys(forms)))

        for combo in product(*token_forms):
            underscore_form = '_'.join(combo)
            add(underscore_form)
            add(cls._sanitize_alias(underscore_form))

            concatenated = ''.join(combo)
            add(concatenated)
            add(cls._sanitize_alias(concatenated))
            if concatenated:
                camel_case = concatenated[0].lower() + concatenated[1:]
                add(camel_case)
                add(cls._sanitize_alias(camel_case))

        cls._NAME_VARIANTS_CACHE[base_name] = variants
        return variants

    def resolve_name_variant(self, preferred_name: str, available_names):
        if not preferred_name or not available_names:
            return None

        if not isinstance(available_names, (list, tuple)):
            try:
                available_names = list(available_names)
            except TypeError:
                available_names = [available_names]

        available_strings = [name for name in available_names if isinstance(name, str)]
        if not available_strings:
            return None

        direct_lookup = {name: name for name in available_strings}
        sanitized_lookup = {self._sanitize_alias(name): name for name in available_strings}
        lower_lookup = {name.lower(): name for name in available_strings}

        variants = self.generate_name_variants(preferred_name)
        for variant in variants:
            if variant in direct_lookup:
                return direct_lookup[variant]
        for variant in variants:
            sanitized_variant = self._sanitize_alias(variant)
            if sanitized_variant in sanitized_lookup:
                return sanitized_lookup[sanitized_variant]
        for variant in variants:
            lower_variant = variant.lower()
            if lower_variant in lower_lookup:
                return lower_lookup[lower_variant]
        return None

    def resolve_name_list(self, preferred_names, available_names):
        resolved = []
        for name in preferred_names:
            actual = self.resolve_name_variant(name, available_names)
            if actual is None:
                return None
            resolved.append(actual)
        return resolved

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        self._data = np.asarray(value)

    def __len__(self):
        if self.data.ndim == 0:
            return 0
        return self.data.shape[0]

    @property
    def size(self):
        return self.data.size

    def compute(self, pcd, **kwargs):
        """Вычисляет данные для поля. Базовая реализация ничего не делает."""
        pass

    # --- Format-specific handlers ---

    @property
    def df_column_names(self):
        return [self.name]

    @property
    def pcd_field_names(self):
        # By default, use capitalized name for PCD field
        return [self.name.capitalize()]

    def pack_pcd_data(self):
        return self.data

    def unpack_pcd_data(self, pcd_data):
        self.data = np.nan_to_num(np.asarray(pcd_data))

    @property
    def las_attrs(self):
        attrs = {}
        for variant in self.generate_name_variants(self.name):
            attrs[variant] = lambda l, attr=variant: getattr(l, attr)
        return attrs

    @property
    def txt_column_map(self):
        # CloudCompare-style capitalized names
        return {self.name: self.name.capitalize()}

    def add_las_extra_dims(self, las):
        """Добавляет необходимые Extra Dimensions в заголовок LAS."""
        # Fallback for simple scalar fields
        if isinstance(self, ScalarField) and self.name not in ['intensity', 'gps_time']:
            try:
                las.add_extra_dim(laspy.ExtraBytesParams(name=self.name, type=np.float32))
            except Exception:  # Already exists
                pass

    def pack_las_data(self, las):
        """Упаковывает данные в объект laspy.LasData."""
        # Fallback for attributes with the same name as the field
        if hasattr(las, self.name):
            setattr(las, self.name, self.data)


class ScalarField(Field):
    """Базовый класс для скалярных полей (1 значение на точку)."""

    @property
    def default_value(self) -> np.ndarray:
        return np.empty(0)

    def unpack_pcd_data(self, pcd_data):
        self.data = np.nan_to_num(np.asarray(pcd_data).ravel())


class VectorField(Field):
    """Базовый класс для векторных полей (N значений на точку)."""

    def __init__(self, data=None, num_columns: int = 3):
        self.num_columns = num_columns
        super().__init__(data)

    @property
    def default_value(self) -> np.ndarray:
        return np.empty((0, self.num_columns))


# --- Конкретные классы полей ---

class Points(VectorField):
    @property
    def name(self) -> str: return 'points'

    @property
    def df_column_names(self): return ['x', 'y', 'z']

    @property
    def pcd_field_names(self): return ['x', 'y', 'z']

    @property
    def las_attrs(self): return {'points': lambda l: np.vstack([l.x, l.y, l.z]).transpose()}

    @property
    def txt_column_map(self): return {'x': 'X', 'y': 'Y', 'z': 'Z'}

    def unpack_pcd_data(self, pcd_data):
        # for 'points', pcd_data comes from 'x' field, but it is Nx3
        self.data = np.asarray(pcd_data)

    def add_las_extra_dims(self, las):
        """Точки являются базовым полем, не требуют extra dims."""
        pass

    def pack_las_data(self, las):
        """Записывает данные точек в las объект."""
        las.x = self.data[:, 0]
        las.y = self.data[:, 1]
        las.z = self.data[:, 2]


class Intensity(ScalarField):
    @property
    def name(self) -> str: return 'intensity'

    @property
    def txt_column_map(self): return {'intensity': 'Intensity'}


class RGB(VectorField):
    @property
    def name(self) -> str: return 'rgb'

    @property
    def default_value(self) -> np.ndarray:
        return np.empty((0, self.num_columns), dtype=np.uint8)

    @property
    def df_column_names(self): return ['r', 'g', 'b']

    @property
    def pcd_field_names(self): return ['rgb']

    @property
    def txt_column_map(self): return {'r': 'R', 'g': 'G', 'b': 'B'}

    def pack_pcd_data(self):
        return pypcd.encode_rgb_for_pcl(np.uint8(self.data))

    def unpack_pcd_data(self, pcd_data):
        self.data = np.nan_to_num(pypcd.decode_rgb_from_pcl(pcd_data))

    @property
    def las_attrs(self):
        return {'rgb': lambda l: (np.vstack([l.red, l.green, l.blue]).transpose() // 256).astype(np.uint8)}

    def add_las_extra_dims(self, las):
        pass  # RGB is part of the base point format

    def pack_las_data(self, las):
        las.colors = (self.data.astype(np.uint16) * 256)


class Normals(VectorField):
    @property
    def name(self) -> str: return 'normals'

    @property
    def df_column_names(self): return ['nx', 'ny', 'nz']

    @property
    def pcd_field_names(self): return ['normal_x', 'normal_y', 'normal_z']

    @property
    def txt_column_map(self): return {'nx': 'Nx', 'ny': 'Ny', 'nz': 'Nz'}

    @property
    def las_attrs(self):
        return {'normals': lambda l: np.vstack([l.nx, l.ny, l.nz]).transpose()}

    def add_las_extra_dims(self, las):
        try:
            las.add_extra_dim(laspy.ExtraBytesParams(name="nx", type=np.float32))
            las.add_extra_dim(laspy.ExtraBytesParams(name="ny", type=np.float32))
            las.add_extra_dim(laspy.ExtraBytesParams(name="nz", type=np.float32))
        except Exception:  # already exists
            pass

    def pack_las_data(self, las):
        las.nx = self.data[:, 0]
        las.ny = self.data[:, 1]
        las.nz = self.data[:, 2]

    def compute(self, pcd, radius: float = 0.1, max_nn: int = 30):
        # with Timer("estimate normals"):
        if len(pcd.points) == 0:
            return

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd.points)
        o3d_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        self.data = np.asarray(o3d_pcd.normals)


class OriginalCloudIndex(ScalarField):
    @property
    def name(self) -> str: return 'original_cloud_index'

    @property
    def default_value(self) -> np.ndarray:
        return np.empty(0, dtype=np.int32)

    @property
    def pcd_field_names(self): return ['Original_cloud_index']

    @property
    def txt_column_map(self): return {'original_cloud_index': 'Original_cloud_index'}


class TreeID(ScalarField):
    @property
    def name(self) -> str: return 'tree_id'

    @property
    def default_value(self) -> np.ndarray:
        return np.empty(0, dtype=np.int32)

    @property
    def pcd_field_names(self): return ['Tree_ID']

    @property
    def txt_column_map(self): return {'tree_id': 'Tree_ID'}


class GPSTime(ScalarField):
    @property
    def name(self) -> str: return 'gps_time'

    @property
    def pcd_field_names(self): return ['GpsTime']

    @property
    def txt_column_map(self): return {'gps_time': 'GpsTime'}


class IlluminanceRay(ScalarField):
    @property
    def name(self) -> str: return 'illuminance_ray'

    @property
    def pcd_field_names(self): return ['Illuminance_(RAY)']

    @property
    def txt_column_map(self): return {'illuminance_ray': 'Illuminance_(RAY)'}

    def compute(self,
                pcd,
                num_rays: int = 8,
                max_ray_distance: float = 0.5,
                ao_neighbor_radius: float = 0.02,
                normal_est_radius: float = None,
                normal_est_max_nn: int = 30,
                force_normal_recalculation: bool = False,
                pcv_cone_angle_deg: float = 15.0):

        with Timer("compute illuminance_ray"):
            num_points = len(pcd.points)
            if num_points == 0:
                return

            if normal_est_radius is None:
                normal_est_radius = max_ray_distance / 2

            if 'normals' not in pcd._fields or pcd.normals.shape[0] != num_points or force_normal_recalculation:
                logger.debug("Estimating normals for illuminance calculation (ray).")
                pcd._fields['normals'].compute(pcd, radius=normal_est_radius, max_nn=normal_est_max_nn)

            points = pcd.points.astype(np.float32)
            normals = pcd.normals.astype(np.float32)

            grid_cell_size = ao_neighbor_radius
            (
                point_indices_sorted,
                unique_hashes,
                starts,
                ends,
                min_bound,
                grid_dims,
            ) = _create_voxel_grid_fast(points, grid_cell_size)

            num_steps = 10
            illuminance = _illuminance_kernel_numba(
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
                grid_cell_size,
            )

            self.data = illuminance


class IlluminancePCV(ScalarField):
    @property
    def name(self) -> str: return 'illuminance_pcv'

    @property
    def pcd_field_names(self): return ['Illuminance_(PCV)']

    @property
    def txt_column_map(self): return {'illuminance_pcv': 'Illuminance_(PCV)'}

    def compute(self,
                pcd,
                num_rays: int = 8,
                max_ray_distance: float = 0.5,
                ao_neighbor_radius: float = 0.02,
                normal_est_radius: float = None,
                normal_est_max_nn: int = 30,
                force_normal_recalculation: bool = False,
                pcv_cone_angle_deg: float = 15.0):

        with Timer("compute illuminance_pcv"):
            num_points = len(pcd.points)
            if num_points == 0:
                return

            if normal_est_radius is None:
                normal_est_radius = max_ray_distance / 2

            if 'normals' not in pcd._fields or pcd.normals.shape[0] != num_points or force_normal_recalculation:
                logger.debug("Estimating normals for illuminance calculation (pcv).")
                pcd._fields['normals'].compute(pcd, radius=normal_est_radius, max_nn=normal_est_max_nn)

            points = pcd.points.astype(np.float32)
            normals = pcd.normals.astype(np.float32)

            grid_cell_size = ao_neighbor_radius
            (
                point_indices_sorted,
                unique_hashes,
                starts,
                ends,
                min_bound,
                grid_dims,
            ) = _create_voxel_grid_fast(points, grid_cell_size)

            num_dirs = max(4, int(num_rays))
            directions_local = _generate_hemisphere_directions_fibonacci(num_dirs).astype(np.float32)
            cos_aperture = np.float32(np.cos(np.deg2rad(pcv_cone_angle_deg)))
            tan_half_aperture = np.float32(np.sqrt(max(0.0, (1.0 - float(cos_aperture)) / (1.0 + float(cos_aperture)))))

            illuminance = _illuminance_pcv_dda_numba(
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
                directions_local,
                cos_aperture,
                tan_half_aperture
            )

            self.data = illuminance


class IlluminanceCC(ScalarField):
    @property
    def name(self) -> str: return 'illuminance_cc'

    @property
    def pcd_field_names(self): return ['Illuminance']

    @property
    def txt_column_map(self): return {'illuminance_cc': 'Illuminance'}


# Совместимость: старый класс 'Illuminance' оставляем как алиас PCV-реализации
class Illuminance(IlluminancePCV):
    @property
    def name(self) -> str: return 'illuminance'

    @property
    def pcd_field_names(self): return ['Illuminance']

    @property
    def txt_column_map(self): return {'illuminance': 'Illuminance_(PCV)'}


class NYFilteringMask(ScalarField):
    @property
    def name(self) -> str: return 'ny_filtering_mask'

    @property
    def default_value(self) -> np.ndarray:
        return np.empty(0, dtype=np.float32)

    @property
    def pcd_field_names(self): return ['Ny_filtering_mask']

    @property
    def txt_column_map(self): return {'ny_filtering_mask': 'Ny_filtering_mask'}

    def compute(self,
                pcd,
                ny_threshold: float = 0.9,
                voxel_size: float = 0.1,
                connectivity: int = 26,
                normal_est_radius: float = 0.1,
                normal_est_max_nn: int = 30,
                force_normal_recalculation: bool = False,
                show_debug: bool = False):
        # Нормали
        num_points = len(pcd.points)
        if num_points == 0:
            self.data = np.empty(0, dtype=bool)
            return

        if 'normals' not in pcd._fields or pcd.normals.shape[0] != num_points or force_normal_recalculation:
            pcd._fields['normals'].compute(pcd, radius=normal_est_radius, max_nn=normal_est_max_nn)

        # Фильтр по ny
        mask_ny = (pcd.ny > ny_threshold) | (pcd.ny < -ny_threshold)
        original_indices = np.where(mask_ny)[0]

        mask_pc = np.zeros(num_points, dtype=bool)
        if original_indices.size == 0:
            self.data = mask_pc
            return

        # Клонирование и обрезка
        pc_cl = pcd.clone()
        pc_cl.index_cut(original_indices)

        # Кластеризация LCC
        clustering = LCC(voxel_size=voxel_size, connectivity=connectivity).fit(pc_cl.points)
        labels_stumps = clustering.labels_.astype(np.int64)

        # Выбор крупных кластеров
        unique_labels, counts = np.unique(labels_stumps, return_counts=True)
        valid_clusters_mask = unique_labels != -1
        if np.any(valid_clusters_mask):
            valid_labels = unique_labels[valid_clusters_mask]
            valid_counts = counts[valid_clusters_mask]
            max_size = np.max(valid_counts) if valid_counts.size > 0 else 0
            median_size = np.median(valid_counts) if valid_counts.size > 0 else 0
            threshold = max(0.1 * max_size, median_size)
            large_cluster_mask = valid_counts > threshold
            large_cluster_labels = valid_labels[large_cluster_mask]
        else:
            large_cluster_labels = np.array([], dtype=unique_labels.dtype)

        mask_cl = np.isin(labels_stumps, large_cluster_labels)

        # Отображение маски обратно на исходный pc
        if mask_cl.any():
            mask_pc[original_indices[mask_cl]] = True

        if show_debug:
            try:
                pc_cl.show(labels=labels_stumps)
                pc_cl.show(labels=mask_cl)
                pcd.show(labels=mask_pc)
            except Exception:
                pass

        self.data = np.asarray(mask_pc, dtype=np.float32)


class NFilteringMask(ScalarField):
    @property
    def name(self) -> str: return 'n_filtering_mask'

    @property
    def default_value(self) -> np.ndarray:
        return np.empty(0, dtype=np.float32)

    @property
    def pcd_field_names(self): return ['N_filtering_mask']

    @property
    def txt_column_map(self): return {'n_filtering_mask': 'N_filtering_mask'}

    def compute(self,
                pcd,
                **kwargs):
        num_points = len(pcd.points)
        if num_points == 0:
            self.data = np.empty(0, dtype=np.float32)
            return

        def _safe_arr(arr):
            if arr is None or len(arr) != num_points:
                return np.zeros(num_points, dtype=np.float32)
            return np.asarray(arr, dtype=np.float32).ravel()

        nx = _safe_arr(getattr(pcd, 'nx_filtering_mask', None))
        ny = _safe_arr(getattr(pcd, 'ny_filtering_mask', None))
        nz = _safe_arr(getattr(pcd, 'nz_filtering_mask', None))

        self.data = np.minimum(nx + ny + nz, 1.0).astype(np.float32)


class NXFilteringMask(ScalarField):
    @property
    def name(self) -> str: return 'nx_filtering_mask'

    @property
    def default_value(self) -> np.ndarray:
        return np.empty(0, dtype=np.float32)

    @property
    def pcd_field_names(self): return ['Nx_filtering_mask']

    @property
    def txt_column_map(self): return {'nx_filtering_mask': 'Nx_filtering_mask'}

    def compute(self,
                pcd,
                nx_threshold: float = 0.9,
                voxel_size: float = 0.1,
                connectivity: int = 26,
                normal_est_radius: float = 0.1,
                normal_est_max_nn: int = 30,
                force_normal_recalculation: bool = False,
                show_debug: bool = False):
        num_points = len(pcd.points)
        if num_points == 0:
            self.data = np.empty(0, dtype=bool)
            return

        if 'normals' not in pcd._fields or pcd.normals.shape[0] != num_points or force_normal_recalculation:
            pcd._fields['normals'].compute(pcd, radius=normal_est_radius, max_nn=normal_est_max_nn)

        mask_nx = (pcd.nx > nx_threshold) | (pcd.nx < -nx_threshold)
        original_indices = np.where(mask_nx)[0]

        mask_pc = np.zeros(num_points, dtype=bool)
        if original_indices.size == 0:
            self.data = mask_pc
            return

        pc_cl = pcd.clone()
        pc_cl.index_cut(original_indices)

        clustering = LCC(voxel_size=voxel_size, connectivity=connectivity).fit(pc_cl.points)
        labels_stumps = clustering.labels_.astype(np.int64)

        unique_labels, counts = np.unique(labels_stumps, return_counts=True)
        valid_clusters_mask = unique_labels != -1
        if np.any(valid_clusters_mask):
            valid_labels = unique_labels[valid_clusters_mask]
            valid_counts = counts[valid_clusters_mask]
            max_size = np.max(valid_counts) if valid_counts.size > 0 else 0
            median_size = np.median(valid_counts) if valid_counts.size > 0 else 0
            threshold = max(0.1 * max_size, median_size)
            large_cluster_mask = valid_counts > threshold
            large_cluster_labels = valid_labels[large_cluster_mask]
        else:
            large_cluster_labels = np.array([], dtype=unique_labels.dtype)

        mask_cl = np.isin(labels_stumps, large_cluster_labels)

        if mask_cl.any():
            mask_pc[original_indices[mask_cl]] = True

        if show_debug:
            try:
                pc_cl.show(labels=labels_stumps)
                pc_cl.show(labels=mask_cl)
                pcd.show(labels=mask_pc)
            except Exception:
                pass

        self.data = np.asarray(mask_pc, dtype=np.float32)


class ExpandFilteringMask(ScalarField):
    @property
    def name(self) -> str: return 'expand_filtering_mask'

    @property
    def default_value(self) -> np.ndarray:
        return np.empty(0, dtype=np.float32)

    @property
    def pcd_field_names(self): return ['Expand_filtering_mask']

    @property
    def txt_column_map(self): return {'expand_filtering_mask': 'Expand_filtering_mask'}

    def compute(self,
                pcd,
                tolerance: float = 0.05):
        num_points = len(pcd.points)
        if num_points == 0:
            self.data = np.empty(0, dtype=np.float32)
            return

        # Гарантируем наличие объединённой маски
        if ('n_filtering_mask' not in pcd._fields or
                getattr(pcd._fields['n_filtering_mask'], 'size', 0) != num_points):
            if 'n_filtering_mask' in pcd._fields:
                pcd._fields['n_filtering_mask'].compute(pcd)
            else:
                # Если поле отсутствует вообще — считаем пустую маску
                self.data = np.zeros(num_points, dtype=np.float32)
                return

        # Часть облака по маске n_filtering_mask == 1
        idx_mask = np.where(pcd.n_filtering_mask == 1)[0]
        pc_cl = pcd.clone()
        pc_cl.index_cut(idx_mask)

        # Расширение области: отмечаем точки исходного облака,
        # попадающие в окрестность tol от выбранного поднабора
        arr = diff_area(pcd.points, pc_cl.points, tolerance=tolerance)

        # Можно получить подмножество при необходимости:
        # pc_cl1 = pcd.clone().index_cut(arr)

        self.data = np.asarray(arr, dtype=np.float32)


class NZFilteringMask(ScalarField):
    @property
    def name(self) -> str: return 'nz_filtering_mask'

    @property
    def default_value(self) -> np.ndarray:
        return np.empty(0, dtype=np.float32)

    @property
    def pcd_field_names(self): return ['Nz_filtering_mask']

    @property
    def txt_column_map(self): return {'nz_filtering_mask': 'Nz_filtering_mask'}

    def compute(self,
                pcd,
                nz_min: float = -0.05,
                nz_max: float = 0.05,
                voxel_size: float = 0.1,
                connectivity: int = 26,
                normal_est_radius: float = 0.1,
                normal_est_max_nn: int = 30,
                force_normal_recalculation: bool = False,
                show_debug: bool = False):
        num_points = len(pcd.points)
        if num_points == 0:
            self.data = np.empty(0, dtype=bool)
            return

        if 'normals' not in pcd._fields or pcd.normals.shape[0] != num_points or force_normal_recalculation:
            pcd._fields['normals'].compute(pcd, radius=normal_est_radius, max_nn=normal_est_max_nn)

        # Диапазон по nz: (nz_min, nz_max)
        mask_nz = (pcd.nz > nz_min) & (pcd.nz < nz_max)
        original_indices = np.where(mask_nz)[0]

        mask_pc = np.zeros(num_points, dtype=bool)
        if original_indices.size == 0:
            self.data = mask_pc
            return

        pc_cl = pcd.clone()
        pc_cl.index_cut(original_indices)

        clustering = LCC(voxel_size=voxel_size, connectivity=connectivity).fit(pc_cl.points)
        labels_stumps = clustering.labels_.astype(np.int64)

        unique_labels, counts = np.unique(labels_stumps, return_counts=True)
        valid_clusters_mask = unique_labels != -1
        if np.any(valid_clusters_mask):
            valid_labels = unique_labels[valid_clusters_mask]
            valid_counts = counts[valid_clusters_mask]
            max_size = np.max(valid_counts) if valid_counts.size > 0 else 0
            median_size = np.median(valid_counts) if valid_counts.size > 0 else 0
            threshold = max(0.1 * max_size, median_size)
            large_cluster_mask = valid_counts > threshold
            large_cluster_labels = valid_labels[large_cluster_mask]
        else:
            large_cluster_labels = np.array([], dtype=unique_labels.dtype)

        mask_cl = np.isin(labels_stumps, large_cluster_labels)

        if mask_cl.any():
            mask_pc[original_indices[mask_cl]] = True

        if show_debug:
            try:
                pc_cl.show(labels=labels_stumps)
                pc_cl.show(labels=mask_cl)
                pcd.show(labels=mask_pc)
            except Exception:
                pass

        self.data = np.asarray(mask_pc, dtype=np.float32)
