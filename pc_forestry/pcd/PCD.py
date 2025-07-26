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
from .features import (
    Points, Intensity, RGB, Normals, OriginalCloudIndex, GPSTime, Illuminance,
    Feature, ScalarFeature, VectorFeature
)


class PCD:
    def __init__(self, features=None, **kwargs):
        if features is None:
            self._features = {
                'points': Points(),
                'intensity': Intensity(),
                'rgb': RGB(),
                'normals': Normals(),
                'original_cloud_index': OriginalCloudIndex(),
                'gps_time': GPSTime(),
                'illuminance': Illuminance(),
            }
        else:
            self._features = features

        # Populate data from kwargs
        for name, data in kwargs.items():
            if name in self._features:
                self._features[name].data = data

        self._create_properties()

    def _create_properties(self):
        """Dynamically create properties for each feature."""
        for name, feature in self._features.items():
            # Create main property for the feature data
            setattr(PCD, name, property(
                fget=lambda self, n=name: self._features[n].data,
                fset=lambda self, value, n=name: setattr(self._features[n], 'data', value)
            ))

            # Create properties for vector components (e.g., x, y, z for points)
            if isinstance(feature, VectorFeature):
                for i, col_name in enumerate(feature.df_column_names):
                    # Use a new variable in the lambda's scope
                    prop_name = col_name.lower()
                    if not hasattr(PCD, prop_name):
                        setattr(PCD, prop_name, property(
                            fget=lambda self, n=name, idx=i: self._features[n].data[:, idx],
                            fset=lambda self, value, n=name, idx=i: self._features[n].data.__setitem__(
                                (slice(None), idx), value)
                        ))

    @property
    def df(self) -> pd.DataFrame:
        """ merge all fields in DataFrame """
        df_data = {}
        for feature in self._features.values():
            if feature.size > 0:
                if isinstance(feature, VectorFeature):
                    for i, col_name in enumerate(feature.df_column_names):
                        df_data[col_name] = feature.data[:, i]
                else:
                    df_data[feature.name] = feature.data
        return pd.DataFrame(df_data)

    def save(self, file_path: str, verbose: bool = False) -> None:
        """Saves the point cloud to a file, dispatching to the correct format handler."""
        file_format = file_path.split('.')[-1]

        @Timer(f"Сохранение файла {file_path}")
        def save_pcd(file_path, verbose=False):
            num_points = len(self.points)
            pcd_fields = []
            pcd_data_list = []

            for feature in self._features.values():
                if feature.size > 0:
                    pcd_fields.extend(feature.pcd_field_names)
                    packed_data = feature.pack_pcd_data()
                    if packed_data.ndim == 1:
                        packed_data = packed_data.reshape(-1, 1)
                    pcd_data_list.append(packed_data)

            if not pcd_data_list:
                return

            dt = np.hstack(pcd_data_list).astype(np.float32)

            md = {'version': .7, 'fields': pcd_fields,
                  'count': [1] * len(pcd_fields), 'width': num_points, 'height': 1,
                  'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 'points': num_points,
                  'type': ['F'] * len(pcd_fields), 'size': [4] * len(pcd_fields), 'data': 'binary'}

            dtype_list = [(name, np.float32) for name in pcd_fields]
            pc_data = dt.view(np.dtype(dtype_list)).squeeze()

            new_cloud = pypcd.PointCloud(md, pc_data)
            new_cloud.save_pcd(file_path, 'binary')

        @Timer(f"Сохранение файла {file_path}")
        def save_las_laz(file_path, verbose=False):
            header = laspy.LasHeader(point_format=3, version="1.4")
            if self.points.size > 0:
                header.point_count = len(self.points)
            las = laspy.LasData(header)

            # Let each feature add its required dimensions to the header
            for feature in self._features.values():
                if feature.size > 0:
                    feature.add_las_extra_dims(las)

            # Let each feature pack its data into the las object
            for feature in self._features.values():
                if feature.size > 0:
                    feature.pack_las_data(las)

            las.write(file_path)

        @Timer(f"Сохранение файла {file_path}")
        def save_csv(file_path, verbose=False):
            self.df.to_csv(file_path, index=False)

        @Timer(f"Сохранение файла {file_path}")
        def save_txt(file_path, verbose=False):
            df = self.df
            # Dynamically rename columns for txt format
            rename_map = {}
            for feature in self._features.values():
                if feature.size > 0:
                    rename_map.update(feature.txt_column_map)
            df = df.rename(columns=rename_map)

            with open(file_path, 'w') as f:
                f.write('//' + ' '.join(df.columns) + '\n')
                df.to_csv(f, sep=' ', index=False, header=False, lineterminator='\n')

        @Timer(f"Сохранение файла {file_path}")
        def save_h5(file_path, verbose=False):
            with h5py.File(file_path, 'w') as h5f:
                for name, feature in self._features.items():
                    if feature.size > 0:
                        h5f.create_dataset(name, data=feature.data)

        # Dispatch table
        savers = {
            'pcd': save_pcd,
            'las': save_las_laz,
            'laz': save_las_laz,
            'csv': save_csv,
            'txt': save_txt,
            'h5': save_h5,
        }

        if file_format in savers:
            savers[file_format](file_path, verbose=verbose)
        else:
            print("invalid format")

    @classmethod
    def read(cls, file_path: str, verbose: bool = False, features=None) -> 'PCD':
        instance = cls(features)
        instance.open(file_path, verbose=verbose)
        return instance

    def open(self, file_path: str, verbose: bool = False) -> None:
        file_ext = "." + file_path.split('.')[-1]

        @Timer(f"Открытие файла {file_path}")
        def open_pcd(self, file_path, verbose=False):
            cloud = pypcd.PointCloud.from_path(file_path)
            cloud_data = cloud.pc_data
            metadata_fields = cloud.get_metadata()["fields"]

            for feature in self._features.values():
                pcd_fields = feature.pcd_field_names
                try:
                    # Handle single field features (like rgb, intensity)
                    if len(pcd_fields) == 1 and pcd_fields[0] in metadata_fields:
                        feature.unpack_pcd_data(cloud_data[pcd_fields[0]])
                    # Handle multi-field features (like points, normals)
                    elif all(f in metadata_fields for f in pcd_fields):
                        data_slice = np.array([cloud_data[f] for f in pcd_fields]).T
                        feature.unpack_pcd_data(data_slice)

                except (ValueError, IndexError, KeyError):
                    pass  # Field not found

        @Timer(f"Открытие файла {file_path}")
        def open_h5(self, file_path, verbose=False):
            with h5py.File(file_path, 'r') as h5f:
                for name, feature in self._features.items():
                    if name in h5f:
                        feature.data = np.asarray(h5f.get(name))

        @Timer(f"Открытие файла {file_path}")
        def open_las_laz(self, file_path, verbose=False):
            las = laspy.read(file_path)
            for feature in self._features.values():
                try:
                    # Uses the first (and likely only) attr from the feature
                    attr_name, loader_func = next(iter(feature.las_attrs.items()))
                    feature.data = loader_func(las)
                except (AttributeError, IndexError, StopIteration):
                    pass  # Field not in las file

        @Timer(f"Открытие файла {file_path}")
        def open_csv(self, file_path, verbose=False):
            df = pd.read_csv(file_path)
            for feature in self._features.values():
                cols = feature.df_column_names
                if all(c in df.columns for c in cols):
                    data = df[cols].values
                    if isinstance(feature, ScalarFeature):
                        data = data.ravel()
                    feature.data = data

        @Timer(f"Открытие файла {file_path}")
        def open_txt(self, file_path, verbose=False):
            with open(file_path, 'r') as file:
                header_line = file.readline().strip()

            if header_line.startswith('//'):
                header = [col.strip('/') for col in header_line.split()]
            else:  # Basic fallback if no header
                data_preview = np.loadtxt(file_path, max_rows=1)
                num_cols = data_preview.shape[0] if data_preview.ndim > 0 else 0
                header = [f'col_{i}' for i in range(num_cols)]
                if verbose:
                    print(f"No header found, creating generic column names.")

            df = pd.read_csv(file_path, sep=r'\s+', comment='/', names=header, header=None)

            for feature in self._features.values():
                # Map from feature standard names to txt header names
                column_map = feature.txt_column_map  # e.g. {'nx': 'Nx', 'ny': 'Ny', 'nz': 'Nz'}
                # Find which of the feature's columns are present in the txt header
                present_txt_cols = [
                    column_map[df_col]
                    for df_col in feature.df_column_names
                    if df_col in column_map and column_map[df_col] in header
                ]

                if present_txt_cols:
                    data = df[present_txt_cols].values
                    if isinstance(feature, ScalarFeature):
                        data = data.ravel()
                    feature.data = data

        # Dispatch table for opening files
        openers = {
            ".h5": open_h5,
            '.pcd': open_pcd,
            '.las': open_las_laz,
            '.laz': open_las_laz,
            '.csv': open_csv,
            '.txt': open_txt,
        }

        if file_ext in openers:
            openers[file_ext](self, file_path, verbose=verbose)
        else:
            print("invalid format")

        self.check_and_pad_fields()

    def check_and_pad_fields(self):
        """ check if all fields have the same length, and pad with zeros if not """
        num_points = len(self.points)
        if num_points == 0:
            lengths = [len(f.data) for f in self._features.values() if hasattr(f.data, '__len__') and f.size > 0]
            if not lengths:
                return
            num_points = max(lengths) if lengths else 0
            if num_points == 0:
                return

        # Set points length first if it's zero
        if len(self.points) == 0 and num_points > 0:
            self.points = np.zeros((num_points, 3))

        for name, feature in self._features.items():
            values = feature.data
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
                feature.data = pad_func((values, padding))
            else:
                feature.data = padding

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

        for name, feature in self._features.items():
            if feature.size > 0:
                feature.data = feature.data[centroids]

    def index_cut(self, idx_labels: np.ndarray) -> None:
        """ cut points and all other fields using indexes """
        for name, feature in self._features.items():
            if feature.size > 0 and len(feature.data) > 0:
                try:
                    feature.data = feature.data[idx_labels]
                except IndexError:
                    # This can happen if a field was not correctly sized.
                    # We create an empty array of the correct shape.
                    shape = (len(idx_labels), feature.data.shape[1]) if feature.data.ndim > 1 else (len(idx_labels),)
                    feature.data = np.empty(shape)

    def compute_feature(self, name: str, **kwargs):
        """
        Вычисляет данные для указанного признака.

        :param name: Имя признака для вычисления (например, 'normals', 'illuminance').
        :param kwargs: Дополнительные аргументы, передаваемые в метод compute() признака.
        """
        if name in self._features:
            feature = self._features[name]
            feature.compute(self, **kwargs)
        else:
            raise ValueError(f"Feature '{name}' not found in PCD object.")

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
            self._features = copy.deepcopy(other._features)
            self._create_properties()
            return
        if num_points_other == 0:
            return

        for name, self_feature in self._features.items():
            other_feature = other._features.get(name)
            if other_feature is not None and other_feature.size > 0:
                # Ensure self_feature has data to concatenate with
                if self_feature.size == 0 and num_points_self > 0:
                    # Initialize with default-like empty data of the correct length
                    if isinstance(self_feature, VectorFeature):
                        self_feature.data = np.zeros((num_points_self, self_feature.num_columns))
                    else:
                        self_feature.data = np.zeros(num_points_self)

                self_feature.data = np.concatenate((self_feature.data, other_feature.data), axis=0)
        self._create_properties()

    def show(self, color_field: str = 'intensity') -> None:
        """ show PCD object """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        if color_field == 'rgb' and self.rgb.size > 0:
            colors = np.asarray(self.rgb)
            colors = colors / 255.0  # normalize RGB values
            pcd.colors = o3d.utility.Vector3dVector(colors)
        elif color_field in self._features and self._features[color_field].size > 0:
            field_values = np.asarray(self._features[color_field].data)
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

        for name, feature in self._features.items():
            feature.data = normalize(feature.data)
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
        for name, feature in self._features.items():
            feature.data = np.nan_to_num(feature.data)

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
        elif color_field in self._features and self._features[color_field].size > 0:
            field_values = np.asarray(self._features[color_field].data)
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
