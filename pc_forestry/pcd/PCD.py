import torch
import numpy as np
from time import time
from ..utils import pypcd, fps
import open3d as o3d
import laspy
import pyvista
import h5py
import pandas as pd
import copy
from tqdm import tqdm
from loguru import logger
from numba import njit, prange
from .illuminance.illuminance import _create_voxel_grid_fast, _illuminance_kernel_numba
from ..utils.timer import Timer


class PCD:
    def __init__(self,
                 points=None,
                 intensity=None,
                 rgb=None,
                 original_cloud_index=None,
                 gps_time=None,
                 illuminance=None,
                 normals=None):
        self._data = {
            'points': points if points is not None else np.empty((0, 3)),
            'intensity': intensity if intensity is not None else np.empty(0),
            'rgb': rgb if rgb is not None else np.empty((0, 3)),
            'original_cloud_index': original_cloud_index if original_cloud_index is not None else np.empty(0),
            'gps_time': gps_time if gps_time is not None else np.empty(0),
            'illuminance': illuminance if illuminance is not None else np.empty(0),
        }
        self._normals = normals if normals is not None else np.empty((0, 3))

    @property
    def df(self) -> pd.DataFrame:
        """ merge all fields in DataFrame """
        data = {}
        # Special handling for multi-column fields
        if 'points' in self._data and self._data['points'].size > 0:
            data['x'] = self.x
            data['y'] = self.y
            data['z'] = self.z
        if 'rgb' in self._data and self._data['rgb'].size > 0:
            data['r'] = self.r
            data['g'] = self.g
            data['b'] = self.b

        # Add all other (scalar) fields
        for name, values in self._data.items():
            if name not in ['points', 'rgb'] and values.size > 0:
                data[name] = values

        return pd.DataFrame(data)

    def save(self, file_path: str, verbose: bool = False) -> None:
        """Saves the point cloud to a file, dispatching to the correct format handler."""
        format = file_path.split('.')[-1]

        # Format-specific handlers for saving data
        # This makes adding new formats or changing existing ones much cleaner.
        pcd_handler = {
            'fields': ['x', 'y', 'z', 'rgb', 'GpsTime', 'Original_cloud_index', 'Intensity', 'Illuminance'],
            'data_map': {
                'points': (lambda p: p, slice(0, 3)),
                'rgb': (lambda c: pypcd.encode_rgb_for_pcl(np.uint8(c)), 3),
                'gps_time': (lambda g: g, 4),
                'original_cloud_index': (lambda o: o, 5),
                'intensity': (lambda i: i, 6),
                'illuminance': (lambda i: i, 7),
            }
        }
        las_handler = {
            'extra_dims': [
                laspy.ExtraBytesParams(name="illuminance", type=np.float32),
                laspy.ExtraBytesParams(name="original_cloud_index", type=np.float32)
            ],
            'attr_map': {
                'points': lambda las, p: setattr(las, 'points', p),
                'rgb': lambda las, c: setattr(las, 'colors', (c.astype(np.uint16) * 256)),
                'intensity': lambda las, i: setattr(las, 'intensity', i),
                'illuminance': lambda las, i: setattr(las, 'illuminance', i),
                'gps_time': lambda las, g: setattr(las, 'gps_time', g),
                'original_cloud_index': lambda las, o: setattr(las, 'original_cloud_index', o)
            }
        }

        @Timer(f"Сохранение файла {file_path}")
        def save_pcd(file_path, verbose=False):
            num_points = len(self.points)
            dt = np.zeros((num_points, len(pcd_handler['fields'])), dtype=np.float32)

            for field, (func, col_slice) in pcd_handler['data_map'].items():
                if self._data.get(field, np.array([])).size > 0:
                    dt[:, col_slice] = func(self._data[field])

            md = {'version': .7, 'fields': pcd_handler['fields'],
                  'count': [1] * len(pcd_handler['fields']), 'width': num_points, 'height': 1,
                  'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 'points': num_points,
                  'type': ['F'] * len(pcd_handler['fields']), 'size': [4] * len(pcd_handler['fields']), 'data': 'binary'}

            dtype_list = [(name, np.float32) for name in pcd_handler['fields']]
            pc_data = dt.view(np.dtype(dtype_list)).squeeze()

            new_cloud = pypcd.PointCloud(md, pc_data)
            new_cloud.save_pcd(file_path, 'binary')

        @Timer(f"Сохранение файла {file_path}")
        def save_las_laz(file_path, verbose=False):
            header = laspy.LasHeader(point_format=3, version="1.4")
            header.point_count = len(self.points)
            las = laspy.LasData(header)

            for extra_dim in las_handler['extra_dims']:
                las.add_extra_dim(extra_dim)

            for field, func in las_handler['attr_map'].items():
                if self._data.get(field, np.array([])).size > 0:
                    func(las, self._data[field])

            las.write(file_path)

        @Timer(f"Сохранение файла {file_path}")
        def save_csv(file_path, verbose=False):
            self.df.to_csv(file_path, index=False)

        @Timer(f"Сохранение файла {file_path}")
        def save_txt(file_path, verbose=False):
            df = self.df
            # Remap columns for txt format
            df = df.rename(columns={
                'x': 'X', 'y': 'Y', 'z': 'Z',
                'intensity': 'Intensity', 'r': 'R', 'g': 'G', 'b': 'B',
                'original_cloud_index': 'Original_cloud_index',
                'gps_time': 'GpsTime', 'illuminance': 'Illuminance_(PCV)'
            })

            with open(file_path, 'w') as f:
                f.write('//' + ' '.join(df.columns) + '\n')
                df.to_csv(f, sep=' ', index=False, header=False, lineterminator='\n')

        @Timer(f"Сохранение файла {file_path}")
        def save_h5(file_path, verbose=False):
            with h5py.File(file_path, 'w') as h5f:
                for name, data in self._data.items():
                    if data.size > 0:
                        h5f.create_dataset(name, data=data)

        # Dispatch table
        savers = {
            'pcd': save_pcd,
            'las': save_las_laz,
            'laz': save_las_laz,
            'csv': save_csv,
            'txt': save_txt,
            'h5': save_h5,
        }

        if format in savers:
            savers[format](file_path, verbose=verbose)
        else:
            print("invalid format")

    @classmethod
    def read(cls, file_path: str, verbose: bool = False) -> 'PCD':
        instance = cls()
        instance.open(file_path, verbose=verbose)
        return instance

    def open(self, file_path: str, verbose: bool = False) -> None:
        @Timer(f"Открытие файла {file_path}")
        def open_pcd(self, file_path, verbose=False):
            """ open .pcd """
            cloud = pypcd.PointCloud.from_path(file_path)
            data = cloud.pc_data.view(np.float32).reshape(
                cloud.pc_data.shape + (-1,))

            # Mapping from PCL field names to PCD attribute names
            field_map = {
                'x': ('points', slice(0, 3)),
                'Intensity': ('intensity', None),
                'Illuminance': ('illuminance', None),
                'rgb': ('rgb', None),
                'GpsTime': ('gps_time', None),
                'Original_cloud_index': ('original_cloud_index', None)
            }

            metadata_fields = cloud.get_metadata()["fields"]
            for pcl_name, (pcd_name, col_slice) in field_map.items():
                try:
                    idx = metadata_fields.index(pcl_name)
                    if pcd_name == 'points':
                        self.points = data[:, idx:idx+3]
                    elif pcd_name == 'rgb':
                        self.rgb = np.nan_to_num(pypcd.decode_rgb_from_pcl(data[:, idx]))
                    else:
                        setattr(self, pcd_name, np.nan_to_num(np.asarray(data[:, idx])))
                except ValueError:
                    # Field not found in file, will be empty
                    pass

        @Timer(f"Открытие файла {file_path}")
        def open_h5(self, file_path, verbose=False):
            """ open .h5 """
            with h5py.File(file_path, 'r') as h5f:
                for name in self._data.keys():
                    if name in h5f:
                        self._data[name] = np.asarray(h5f.get(name))

        @Timer(f"Открытие файла {file_path}")
        def open_las_laz(self, file_path, verbose=False):
            """ open .las or .laz """
            las = laspy.read(file_path)

            # Mapping from laspy attributes to PCD attributes
            attr_map = {
                'points': lambda l: np.vstack([l.x, l.y, l.z]).transpose(),
                'intensity': lambda l: l.intensity,
                'illuminance': lambda l: l.illuminance,
                'rgb': lambda l: (np.vstack([l.red, l.green, l.blue]).transpose() // 256).astype(np.uint8),
                'original_cloud_index': lambda l: l.original_cloud_index,
                'gps_time': lambda l: l.gps_time
            }

            for pcd_name, func in attr_map.items():
                try:
                    setattr(self, pcd_name, func(las))
                except (AttributeError, IndexError):
                    pass  # Field does not exist in LAS file

        @Timer(f"Открытие файла {file_path}")
        def open_csv(self, file_path, verbose=False):
            """ open .csv """
            df = pd.read_csv(file_path)

            col_map = {
                'points': (['x', 'y', 'z'], lambda d: d.values),
                'rgb': (['red', 'green', 'blue'], lambda d: d.values),
                'intensity': (['Intensity'], lambda d: d.values.ravel()),
                'gps_time': (['GpsTime'], lambda d: d.values.ravel()),
                'original_cloud_index': (['Original_cloud_index'], lambda d: d.values.ravel()),
                'illuminance': (['Illuminance'], lambda d: d.values.ravel()),
            }

            for pcd_name, (cols, func) in col_map.items():
                if all(c in df.columns for c in cols):
                    setattr(self, pcd_name, func(df[cols]))

        @Timer(f"Открытие файла {file_path}")
        def open_txt(self, file_path, verbose=False):
            """ open .txt """
            with open(file_path, 'r') as file:
                header_line = file.readline().strip()

            if header_line.startswith('//'):
                header = [col.strip('/') for col in header_line.split()]
            else:
                # Basic fallback if no header
                data_preview = np.loadtxt(file_path, max_rows=1)
                num_cols = data_preview.shape[0]
                default_headers = ['X', 'Y', 'Z', 'Intensity', 'R', 'G', 'B']  # common order
                header = default_headers[:num_cols]
                if verbose:
                    print(f"No header found, guessing columns: {header}")

            df = pd.read_csv(file_path, sep=r'\s+', comment='/', names=header, header=None)

            col_map = {
                'points': (['X', 'Y', 'Z'], lambda d: d.values),
                'rgb': (['R', 'G', 'B'], lambda d: d.values),
                'intensity': (['Intensity'], lambda d: d.values.ravel()),
                'gps_time': (['GpsTime', 'Gps_Time'], lambda d: d.values.ravel()),
                'original_cloud_index': (['Original_cloud_index'], lambda d: d.values.ravel()),
                'illuminance': (['Illuminance_(PCV)'], lambda d: d.values.ravel()),
            }

            for pcd_name, (cols, func) in col_map.items():
                # Try all possible column names for a field
                for col_alias in cols:
                    if col_alias in df.columns and pcd_name == 'points':
                        if all(c in df.columns for c in cols):
                            self.points = func(df[cols])
                            break
                    elif col_alias in df.columns:
                        setattr(self, pcd_name, func(df[[col_alias]]))
                        break

        # Dispatch table for opening files
        openers = {
            ".h5": open_h5,
            '.pcd': open_pcd,
            '.las': open_las_laz,
            '.laz': open_las_laz,
            '.csv': open_csv,
            '.txt': open_txt,
        }

        file_ext = "." + file_path.split('.')[-1]
        if file_ext in openers:
            openers[file_ext](self, file_path, verbose=verbose)
        else:
            print("invalid format")

        self.check_and_pad_fields()

    def check_and_pad_fields(self):
        """ check if all fields have the same length, and pad with zeros if not """
        num_points = len(self.points)
        if num_points == 0:
            lengths = [len(v) for v in self._data.values() if hasattr(v, '__len__')]
            if not lengths:
                return
            num_points = max(lengths) if lengths else 0
            if num_points == 0:
                return

        # Set points length first if it's zero
        if len(self.points) == 0 and num_points > 0:
            self.points = np.zeros((num_points, 3))

        for name, values in self._data.items():
            if values is None:
                continue

            current_len = len(values)
            if current_len == num_points:
                continue

            # Determine shape for padding
            if values.ndim > 1:
                pad_shape = (num_points - current_len, values.shape[1])
                pad_func = np.vstack
            else:
                pad_shape = (num_points - current_len,)
                pad_func = np.hstack

            padding = np.zeros(pad_shape, dtype=values.dtype)

            if current_len > 0:
                self._data[name] = pad_func((values, padding))
            else:
                self._data[name] = padding

    def clone(self) -> 'PCD':
        """ clone PCD object """
        return copy.deepcopy(self)

    @Timer("Сэмплирование точек (FPS)")
    def sample_fps(self, num_sample: int, verbose: bool = False) -> None:
        """ sampling 'num_sample' points from 'PCD' class via farthest point sampling algorithm """
        np_points = np.asarray([self.points])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        points_torch = torch.Tensor(np_points).to(device)
        centroids = fps.farthest_point_sample(points_torch, num_sample).cpu().data.numpy()[0]

        for name, values in self._data.items():
            if values.size > 0:
                self._data[name] = values[centroids]
        if self._normals.size > 0:
            self._normals = self._normals[centroids]

    def index_cut(self, idx_labels: np.ndarray) -> None:
        """ cut points and all other fields using indexes """
        for name, values in self._data.items():
            if values.size > 0 and len(values) > 0:
                try:
                    self._data[name] = values[idx_labels]
                except IndexError:
                    # This can happen if a field was not correctly sized.
                    # We create an empty array of the correct shape.
                    shape = (len(idx_labels), values.shape[1]) if values.ndim > 1 else (len(idx_labels),)
                    self._data[name] = np.empty(shape)

        if self._normals.size > 0:
            try:
                self._normals = self._normals[idx_labels]
            except IndexError:
                self._normals = np.empty((len(idx_labels), 3))

    def _generate_hemisphere_rays(self, normal_p: np.ndarray, num_rays: int) -> np.ndarray:
        """Helper method to generate random rays on a hemisphere oriented by a normal vector."""
        rays = []
        while len(rays) < num_rays:
            # Generate a random vector in a 3D space
            v = np.random.normal(size=3)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-6:
                continue
            v /= norm_v

            # Ensure the vector is in the hemisphere defined by the normal
            if np.dot(v, normal_p) < 0:
                v = -v
            rays.append(v)
        return np.array(rays)

    @Timer("Быстрый расчет освещенности")
    def calculate_illuminance(self,
                              num_rays: int = 32,
                              max_ray_distance: float = 0.5,
                              ao_neighbor_radius: float = 0.02,
                              normal_est_radius: float = None,
                              normal_est_max_nn: int = 30,
                              force_normal_recalculation: bool = False) -> None:
        """
        Быстрый и полностью параллельный расчет Ambient Occlusion с использованием numba
        и пространственной сетки (Voxel Grid).
        """
        num_points = len(self.points)
        if num_points == 0:
            return

        if normal_est_radius is None:
            normal_est_radius = max_ray_distance / 2

        # Проверка и расчет нормалей
        if self.normals.shape[0] != num_points or force_normal_recalculation:
            self.estimate_normals(radius=normal_est_radius,
                                  max_nn=normal_est_max_nn)

        points = self.points.astype(np.float32)
        normals = self.normals.astype(np.float32)

        grid_cell_size = ao_neighbor_radius
        point_indices_sorted, cell_starts_ends, min_bound, grid_dims = _create_voxel_grid_fast(
            points, grid_cell_size)

        # Количество шагов вдоль луча. 10-20 обычно достаточно.
        num_steps = 10

        # tqdm можно обернуть вокруг вызова ядра, если хочется видеть общий прогресс,
        # но это не покажет прогресс внутри параллельного цикла.
        # Для отладки можно убрать `parallel=True` и обернуть `prange` в `tqdm`.

        illuminance = _illuminance_kernel_numba(
            points, normals, num_rays, max_ray_distance, ao_neighbor_radius, num_steps,
            point_indices_sorted, cell_starts_ends, min_bound, grid_dims, grid_cell_size
        )

        self.illuminance = illuminance

    @Timer("Оценка нормалей")
    def estimate_normals(self, radius: float = 0.1, max_nn: int = 30) -> None:
        """ estimate normals """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        self._normals = np.asarray(pcd.normals)

    def get_normals(self, radius: float = 0.1, max_nn: int = 30) -> np.ndarray:
        """ get normals """
        if self._normals is None or self._normals.shape[0] != len(self.points):
            logger.debug("Estimating normals")
            self.estimate_normals(radius=radius, max_nn=max_nn)
        return self._normals

    @property
    def normals(self) -> np.ndarray:
        return self.get_normals()

    @normals.setter
    def normals(self, value):
        self._normals = value

    def unique(self) -> None:
        """ leaves only unique point values """
        if self.points.size == 0:
            return
        _, unique_indices = np.unique(
            self.points, axis=0, return_index=True)
        self.index_cut(unique_indices)

    def append(self, other: 'PCD') -> None:
        """ append PCD object """
        if not isinstance(other, PCD):
            raise TypeError("Argument must be an instance of PCD")

        # Ensure the other PCD has the same fields, padding if necessary
        other.check_and_pad_fields()
        self.check_and_pad_fields()

        num_points_self = len(self.points)
        num_points_other = len(other.points)

        # If one cloud is empty, just copy the other
        if num_points_self == 0:
            self._data = copy.deepcopy(other._data)
            self._normals = copy.deepcopy(other._normals)
            return
        if num_points_other == 0:
            return

        for name, self_values in self._data.items():
            other_values = other._data.get(name)
            if other_values is not None and other_values.size > 0:
                self._data[name] = np.concatenate((self_values, other_values), axis=0)

        if self._normals.size > 0 and other._normals.size > 0:
            self.normals = np.concatenate((self.normals, other.normals), axis=0)

    def show(self, color_field: str = 'intensity') -> None:
        """ show PCD object """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        if color_field == 'rgb' and self.rgb.size > 0:
            colors = np.asarray(self.rgb)
            colors = colors / 255.0  # normalize RGB values
            pcd.colors = o3d.utility.Vector3dVector(colors)
        elif hasattr(self, color_field) and getattr(self, color_field).size > 0:
            field_values = np.asarray(getattr(self, color_field))
            field_values = (field_values - field_values.min()) / \
                (field_values.max() - field_values.min())
            colors = np.zeros((field_values.shape[0], 3))
            colors[:, 0] = field_values  # r
            colors[:, 1] = field_values  # g
            colors[:, 2] = field_values  # b
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # o3d.visualization.draw_geometries([pcd])
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.get_render_option().background_color = [0.25, 0.25, 0.25]
        vis.add_geometry(pcd)
        vis.run()

    def normalize_fields(self) -> None:
        """ normalize all numeric fields to range [0, 1] """
        def normalize(array: np.ndarray) -> np.ndarray:
            if array.size > 0:
                min_val, max_val = array.min(), array.max()
                if max_val - min_val > 1e-6:
                    return (array - min_val) / (max_val - min_val)
            return array

        for name, values in self._data.items():
            self._data[name] = normalize(values)
        self.nan_to_zero()

    def shift_to_origin(self) -> None:
        """ shift points to origin (center of mass at zero) """
        if self.points.size > 0:
            self.points -= self.points.mean(axis=0)

    def shift_to_zero(self) -> None:
        """ shift points so that min values for all axes are zero """
        if self.points.size > 0:
            self.points -= self.points.min(axis=0)

    def nan_to_zero(self) -> None:
        """ replace NaN to 0 in all fields """
        for name, values in self._data.items():
            self._data[name] = np.nan_to_num(values)
        self._normals = np.nan_to_num(self._normals)

    @Timer("Визуализация PCD как gif")
    def visual_gif(self, path_gif: str, zoom: float = 0.4, point_size: float = 4.0, color_field: str = 'rgb') -> None:
        """ Визуализировать объект PCD как gif с цветовой схемой blue > green > yellow > red """
        import pyvista
        import numpy as np

        cloud = pyvista.PointSet(self.points)

        def colormap_bgyr(values: np.ndarray) -> np.ndarray:
            """
            Кастомная цветовая карта: blue -> green -> yellow -> red
            values: нормализованный массив [0, 1]
            """
            # Градиент: 0.0 - синий, 0.33 - зеленый, 0.66 - желтый, 1.0 - красный
            colors = np.zeros((values.shape[0], 3))
            for i, v in enumerate(values):
                if v <= 0.33:
                    # от синего (0,0,1) к зеленому (0,1,0)
                    t = v / 0.33
                    colors[i] = [0 * (1-t) + 0 * t, 0 *
                                 (1-t) + 1 * t, 1 * (1-t) + 0 * t]
                elif v <= 0.66:
                    # от зеленого (0,1,0) к желтому (1,1,0)
                    t = (v - 0.33) / (0.66 - 0.33)
                    colors[i] = [0 * (1-t) + 1 * t, 1, 0]
                else:
                    # от желтого (1,1,0) к красному (1,0,0)
                    t = (v - 0.66) / (1.0 - 0.66)
                    colors[i] = [1, 1 * (1-t) + 0 * t, 0]
            return colors

        # Определяем цвета
        if color_field == 'rgb' and self.rgb.size > 0:
            # Если rgb, используем как есть, но нормализуем
            colors = np.asarray(self.rgb)
            colors = colors / 255.0  # нормализация RGB
        elif hasattr(self, color_field) and getattr(self, color_field).size > 0:
            field_values = np.asarray(getattr(self, color_field))
            # Нормализация в [0, 1]
            field_values = (field_values - field_values.min()) / \
                (field_values.max() - field_values.min() + 1e-8)
            colors = colormap_bgyr(field_values)
        else:
            # Цвет по умолчанию — синий
            colors = np.ones(
                (self.points.shape[0], 3)) * np.array([0.0, 0.0, 1.0])

        pl = pyvista.Plotter(off_screen=True)
        pl.add_mesh(
            cloud,
            scalars=colors,
            rgb=True,
            opacity=1,
            point_size=point_size,
            show_scalar_bar=False,
        )
        pl.background_color = (0.5, 0.5, 0.5)  # светло-серый фон
        pl.show(auto_close=False)
        pl.camera.zoom(zoom)
        path = pl.generate_orbital_path(
            n_points=36, shift=cloud.length/3, factor=3.0)
        pl.open_gif(path_gif)
        pl.orbit_on_path(path, write_frames=True)
        pl.close()

    @property
    def points(self):
        return self._data['points']

    @points.setter
    def points(self, value):
        self._data['points'] = np.asarray(value)

    @property
    def x(self):
        return self.points[:, 0]

    @x.setter
    def x(self, value):
        self.points[:, 0] = value

    @property
    def y(self):
        return self.points[:, 1]

    @y.setter
    def y(self, value):
        self.points[:, 1] = value

    @property
    def z(self):
        return self.points[:, 2]

    @z.setter
    def z(self, value):
        self.points[:, 2] = value

    @property
    def intensity(self):
        return self._data['intensity']

    @intensity.setter
    def intensity(self, value):
        self._data['intensity'] = np.asarray(value)

    @property
    def original_cloud_index(self):
        return self._data['original_cloud_index']

    @original_cloud_index.setter
    def original_cloud_index(self, value):
        self._data['original_cloud_index'] = np.asarray(value)

    @property
    def gps_time(self):
        return self._data['gps_time']

    @gps_time.setter
    def gps_time(self, value):
        self._data['gps_time'] = np.asarray(value)

    @property
    def illuminance(self):
        return self._data['illuminance']

    @illuminance.setter
    def illuminance(self, value):
        self._data['illuminance'] = np.asarray(value)

    @property
    def rgb(self):
        return self._data['rgb']

    @rgb.setter
    def rgb(self, value):
        self._data['rgb'] = np.asarray(value)

    @property
    def r(self):
        return self.rgb[:, 0]

    @r.setter
    def r(self, value):
        self.rgb[:, 0] = value

    @property
    def g(self):
        return self.rgb[:, 1]

    @g.setter
    def g(self, value):
        self.rgb[:, 1] = value

    @property
    def b(self):
        return self.rgb[:, 2]

    @b.setter
    def b(self, value):
        self.rgb[:, 2] = value
