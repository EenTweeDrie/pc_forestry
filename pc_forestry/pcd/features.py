
import numpy as np
from abc import ABC, abstractmethod

from ..utils import pypcd
import open3d as o3d
from loguru import logger
from .illuminance.illuminance import _create_voxel_grid_fast, _illuminance_kernel_numba
import laspy
from ..utils.timer import Timer


class Feature(ABC):
    """Абстрактный базовый класс для признака облака точек."""

    def __init__(self, data=None):
        self._data = self.initialize_data(data)

    @property
    @abstractmethod
    def name(self) -> str:
        """Имя признака (например, 'points', 'intensity')."""
        pass

    @property
    @abstractmethod
    def default_value(self) -> np.ndarray:
        """Значение по умолчанию для данных признака."""
        pass

    def initialize_data(self, data):
        if data is not None:
            return np.asarray(data)
        return self.default_value.copy()

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
        """Вычисляет данные для признака. Базовая реализация ничего не делает."""
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
        return {self.name: lambda l: getattr(l, self.name)}

    @property
    def txt_column_map(self):
        # CloudCompare-style capitalized names
        return {self.name: self.name.capitalize()}

    def add_las_extra_dims(self, las):
        """Добавляет необходимые Extra Dimensions в заголовок LAS."""
        # Fallback for simple scalar features
        if isinstance(self, ScalarFeature) and self.name not in ['intensity', 'gps_time']:
            try:
                las.add_extra_dim(laspy.ExtraBytesParams(name=self.name, type=np.float32))
            except Exception:  # Already exists
                pass

    def pack_las_data(self, las):
        """Упаковывает данные в объект laspy.LasData."""
        # Fallback for attributes with the same name as the feature
        if hasattr(las, self.name):
            setattr(las, self.name, self.data)


class ScalarFeature(Feature):
    """Базовый класс для скалярных признаков (1 значение на точку)."""

    @property
    def default_value(self) -> np.ndarray:
        return np.empty(0)

    def unpack_pcd_data(self, pcd_data):
        self.data = np.nan_to_num(np.asarray(pcd_data).ravel())


class VectorFeature(Feature):
    """Базовый класс для векторных признаков (N значений на точку)."""

    def __init__(self, data=None, num_columns: int = 3):
        self.num_columns = num_columns
        super().__init__(data)

    @property
    def default_value(self) -> np.ndarray:
        return np.empty((0, self.num_columns))


# --- Конкретные классы признаков ---

class Points(VectorFeature):
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


class Intensity(ScalarFeature):
    @property
    def name(self) -> str: return 'intensity'

    @property
    def txt_column_map(self): return {'intensity': 'Intensity'}


class RGB(VectorFeature):
    @property
    def name(self) -> str: return 'rgb'

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


class Normals(VectorFeature):
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
        with Timer("estimate normals"):
            if len(pcd.points) == 0:
                return

            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(pcd.points)
            o3d_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        self.data = np.asarray(o3d_pcd.normals)


class OriginalCloudIndex(ScalarFeature):
    @property
    def name(self) -> str: return 'original_cloud_index'

    @property
    def pcd_field_names(self): return ['Original_cloud_index']

    @property
    def txt_column_map(self): return {'original_cloud_index': 'Original_cloud_index'}


class GPSTime(ScalarFeature):
    @property
    def name(self) -> str: return 'gps_time'

    @property
    def pcd_field_names(self): return ['GpsTime']

    @property
    def txt_column_map(self): return {'gps_time': 'GpsTime'}


class Illuminance(ScalarFeature):
    @property
    def name(self) -> str: return 'illuminance'

    @property
    def pcd_field_names(self): return ['Illuminance']

    @property
    def txt_column_map(self): return {'illuminance': 'Illuminance_(PCV)'}

    def compute(self,
                pcd,
                num_rays: int = 32,
                max_ray_distance: float = 0.5,
                ao_neighbor_radius: float = 0.02,
                normal_est_radius: float = None,
                normal_est_max_nn: int = 30,
                force_normal_recalculation: bool = False):

        with Timer("compute illuminance"):
            num_points = len(pcd.points)
            if num_points == 0:
                return

            if normal_est_radius is None:
                normal_est_radius = max_ray_distance / 2

            # Check for normals and compute if necessary
            if 'normals' not in pcd._features or pcd.normals.shape[0] != num_points or force_normal_recalculation:
                logger.debug("Estimating normals for illuminance calculation.")
                pcd._features['normals'].compute(pcd, radius=normal_est_radius, max_nn=normal_est_max_nn)

            points = pcd.points.astype(np.float32)
            normals = pcd.normals.astype(np.float32)

            grid_cell_size = ao_neighbor_radius
            point_indices_sorted, cell_starts_ends, min_bound, grid_dims = _create_voxel_grid_fast(
                points, grid_cell_size)

            num_steps = 10

            illuminance = _illuminance_kernel_numba(
                points, normals, num_rays, max_ray_distance, ao_neighbor_radius, num_steps,
                point_indices_sorted, cell_starts_ends, min_bound, grid_dims, grid_cell_size
            )

        self.data = illuminance
