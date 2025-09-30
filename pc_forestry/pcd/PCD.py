import torch
import numpy as np
from ..utils import pypcd, fps
import open3d as o3d
import laspy
import h5py
import pandas as pd
import copy
from ..utils.timer import Timer
from .fields import (
    Points, Intensity, RGB, Normals, OriginalCloudIndex, GPSTime, Illuminance,
    Field, ScalarField, VectorField
)
from .is_inside import is_inside_sm_parallel, parallelpointinpolygon, ray_tracing_numpy_numba, is_inside_postgis_parallel


class PCD:
    def __init__(self, fields=None, **kwargs):
        if fields is None:
            self._fields = {
                'points': Points(),
                'intensity': Intensity(),
                'rgb': RGB(),
                'normals': Normals(),
                'original_cloud_index': OriginalCloudIndex(),
                'gps_time': GPSTime(),
                'illuminance': Illuminance(),
            }
        else:
            self._fields = fields

        # Populate data from kwargs
        for name, data in kwargs.items():
            if name in self._fields:
                self._fields[name].data = data

        self._create_properties()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_properties(self):
        """Dynamically create properties for each field."""
        for name, field in self._fields.items():
            # Create main property for the field data
            setattr(PCD, name, property(
                fget=lambda self, n=name: self._fields[n].data,
                fset=lambda self, value, n=name: setattr(self._fields[n], 'data', value)
            ))

            # Create properties for vector components (e.g., x, y, z for points)
            if isinstance(field, VectorField):
                for i, col_name in enumerate(field.df_column_names):
                    # Use a new variable in the lambda's scope
                    prop_name = col_name.lower()
                    if not hasattr(PCD, prop_name):
                        setattr(PCD, prop_name, property(
                            fget=lambda self, n=name, idx=i: self._fields[n].data[:, idx],
                            fset=lambda self, value, n=name, idx=i: self._fields[n].data.__setitem__(
                                (slice(None), idx), value)
                        ))

    @property
    def df(self) -> pd.DataFrame:
        """ merge all fields in DataFrame """
        df_data = {}
        for field in self._fields.values():
            if field.size > 0:
                if isinstance(field, VectorField):
                    if hasattr(field.data, 'ndim') and field.data.ndim > 1:
                        for i, col_name in enumerate(field.df_column_names):
                            df_data[col_name] = field.data[:, i]
                else:
                    df_data[field.name] = field.data
        return pd.DataFrame(df_data)

    def save(self, file_path: str) -> None:
        """Saves the point cloud to a file, dispatching to the correct format handler."""
        file_format = file_path.split('.')[-1]

        @Timer(f"Сохранение файла {file_path}")
        def save_pcd(file_path):
            num_points = len(self.points) if hasattr(self.points, '__len__') else 0
            pcd_fields = []
            pcd_data_list = []

            for field in self._fields.values():
                if field.size > 0:
                    pcd_fields.extend(field.pcd_field_names)
                    packed_data = field.pack_pcd_data()
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
        def save_las_laz(file_path):
            header = laspy.LasHeader(point_format=3, version="1.4")
            if self.points.size > 0:
                header.point_count = len(self.points)
            las = laspy.LasData(header)

            # Let each field add its required dimensions to the header
            for field in self._fields.values():
                if field.size > 0:
                    field.add_las_extra_dims(las)

            # Let each field pack its data into the las object
            for field in self._fields.values():
                if field.size > 0:
                    field.pack_las_data(las)

            las.write(file_path)

        @Timer(f"Сохранение файла {file_path}")
        def save_csv(file_pathe):
            self.df.to_csv(file_path, index=False)

        @Timer(f"Сохранение файла {file_path}")
        def save_txt(file_path):
            df = self.df
            # Dynamically rename columns for txt format
            rename_map = {}
            for field in self._fields.values():
                if field.size > 0:
                    rename_map.update(field.txt_column_map)
            df = df.rename(columns=rename_map)

            with open(file_path, 'w') as f:
                f.write('//' + ' '.join(df.columns) + '\n')
                df.to_csv(f, sep=' ', index=False, header=False, lineterminator='\n')

        @Timer(f"Сохранение файла {file_path}")
        def save_h5(file_path):
            with h5py.File(file_path, 'w') as h5f:
                for name, field in self._fields.items():
                    if field.size > 0:
                        h5f.create_dataset(name, data=field.data)

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
            savers[file_format](file_path)
        else:
            print("invalid format")

    @classmethod
    def read(cls, file_path: str, fields=None) -> 'PCD':
        instance = cls(fields)
        instance.open(file_path,)
        return instance

    def open(self, file_path: str) -> None:
        file_ext = "." + file_path.split('.')[-1]

        @Timer(f"Открытие файла {file_path}")
        def open_pcd(self, file_path):
            cloud = pypcd.PointCloud.from_path(file_path)
            cloud_data = cloud.pc_data
            metadata_fields = cloud.get_metadata()["fields"]

            for field in self._fields.values():
                pcd_fields = field.pcd_field_names
                try:
                    # Handle single field fields (like rgb, intensity)
                    if len(pcd_fields) == 1 and pcd_fields[0] in metadata_fields:
                        field.unpack_pcd_data(cloud_data[pcd_fields[0]])
                    # Handle multi-field fields (like points, normals)
                    elif all(f in metadata_fields for f in pcd_fields):
                        data_slice = np.array([cloud_data[f] for f in pcd_fields]).T
                        field.unpack_pcd_data(data_slice)

                except (ValueError, IndexError, KeyError):
                    pass  # Field not found

        @Timer(f"Открытие файла {file_path}")
        def open_h5(self, file_path):
            with h5py.File(file_path, 'r') as h5f:
                for name, field in self._fields.items():
                    if name in h5f:
                        field.data = np.asarray(h5f.get(name))

        @Timer(f"Открытие файла {file_path}")
        def open_las_laz(self, file_path):
            las = laspy.read(file_path)
            for field in self._fields.values():
                try:
                    # Uses the first (and likely only) attr from the field
                    attr_name, loader_func = next(iter(field.las_attrs.items()))
                    field.data = loader_func(las)
                except (AttributeError, IndexError, StopIteration):
                    pass  # Field not in las file

        @Timer(f"Открытие файла {file_path}")
        def open_csv(self, file_path):
            df = pd.read_csv(file_path)
            for field in self._fields.values():
                cols = field.df_column_names
                if all(c in df.columns for c in cols):
                    data = df[cols].values
                    if isinstance(field, ScalarField):
                        data = data.ravel()
                    field.data = data

        @Timer(f"Открытие файла {file_path}")
        def open_txt(self, file_path):
            with open(file_path, 'r') as file:
                header_line = file.readline().strip()

            if header_line.startswith('//'):
                header = [col.strip('/') for col in header_line.split()]
            else:
                raise ValueError(
                    f"Заголовок не найден в файле {file_path}. "
                    f"Ожидалась строка, начинающаяся с '//'."
                )

            # Параметр `names` в pandas требует уникальных имен. Чтобы обработать
            # возможные дубликаты в заголовке файла, мы читаем данные без заголовка
            # и затем выбираем столбцы по их целочисленному индексу.
            df = pd.read_csv(file_path, sep=r'\s+', comment='/', header=None)

            # Мы создаем отображение каждого уникального имени столбца на индекс его
            # первого появления. Это гарантирует, что если имя столбца дублируется,
            # мы будем рассматривать только первое из них, выполняя требование
            # "читать только уникальные".
            unique_header_map = {name: i for i, name in reversed(list(enumerate(header)))}

            for field in self._fields.values():
                # Отображение стандартных имен полей на имена в заголовке txt
                column_map = field.txt_column_map  # например, {'nx': 'Nx', 'ny': 'Ny', 'nz': 'Nz'}

                # Находим целочисленные индексы столбцов, необходимых для этого поля
                col_indices = []
                for df_col in field.df_column_names:
                    txt_col_name = column_map.get(df_col)
                    if txt_col_name and txt_col_name in unique_header_map:
                        col_indices.append(unique_header_map[txt_col_name])

                if col_indices:
                    # Удаляем дубликаты индексов, сохраняя порядок, на случай, если
                    # несколько столбцов поля отображаются на один и тот же
                    # исходный столбец в txt-файле.
                    unique_indices = list(dict.fromkeys(col_indices))

                    data = df.iloc[:, unique_indices].values
                    if isinstance(field, ScalarField):
                        data = data.ravel()
                    field.data = data

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
            openers[file_ext](self, file_path)
        else:
            print("invalid format")

        self.check_and_pad_fields()

    def check_and_pad_fields(self):
        """
        Проверяет, все ли поля имеют одинаковую длину. Если нет, то для полей с
        нулевой длиной создаются массивы нулей, равные по длине максимальной
        длине среди всех полей.
        """
        # Сначала находим максимальную длину среди всех полей
        num_points = 0
        all_lengths = []
        for f in self._fields.values():
            # Убедимся, что у данных есть размерность (не 0-d array) перед вызовом len()
            if hasattr(f.data, 'ndim') and f.data.ndim > 0:
                all_lengths.append(len(f.data))

        if all_lengths:
            num_points = max(all_lengths)

        if num_points == 0:
            # Если все поля пустые, делать нечего
            return

        # Убедимся, что points инициализированы, если нужно
        if self._fields['points'].size == 0:
            self.points = np.zeros((num_points, 3))

        # Теперь проходим по всем полям и дополняем те, что пусты
        for name, field in self._fields.items():
            if field.size < num_points:
                if isinstance(field, VectorField):
                    shape = (num_points, field.num_columns)
                else:  # ScalarField
                    shape = (num_points,)

                # Используем тип данных по умолчанию или float32
                dtype = field.default_value.dtype
                field.data = np.zeros(shape, dtype=dtype)

    def clone(self) -> 'PCD':
        """ clone PCD object """
        return copy.deepcopy(self)

    @Timer("Сэмплирование точек (FPS)")
    def sample_fps(self, num_sample: int) -> None:
        """ sampling 'num_sample' points from 'PCD' class via farthest point sampling algorithm """
        np_points = np.asarray([self.points])
        device = self.device
        points_torch = torch.Tensor(np_points).to(device)
        centroids = fps.farthest_point_sample(points_torch, num_sample).cpu().data.numpy()[0]

        for name, field in self._fields.items():
            if field.size > 0:
                field.data = field.data[centroids]

    def index_cut(self, idx_labels: np.ndarray) -> None:
        """ cut points and all other fields using indexes """
        for name, field in self._fields.items():
            if field.size > 0 and hasattr(field.data, '__len__') and len(field.data) > 0:
                try:
                    field.data = field.data[idx_labels]
                except IndexError:
                    # This can happen if a field was not correctly sized.
                    # We create an empty array of the correct shape.
                    shape = (len(idx_labels), field.data.shape[1]) if field.data.ndim > 1 else (len(idx_labels),)
                    field.data = np.empty(shape)

    def compute_field(self, name: str, **kwargs):
        """
        Вычисляет данные для указанного поля.

        :param name: Имя поля для вычисления (например, 'normals', 'illuminance').
        :param kwargs: Дополнительные аргументы, передаваемые в метод compute() поля.
        """
        if name in self._fields:
            field = self._fields[name]
            field.compute(self, **kwargs)
        else:
            raise ValueError(f"Field '{name}' not found in PCD object.")

    def append(self, other: 'PCD') -> None:
        """ append PCD object """
        if not isinstance(other, PCD):
            raise TypeError("Argument must be an instance of PCD")

        # Ensure the other PCD has the same fields, padding if necessary
        other.check_and_pad_fields()
        self.check_and_pad_fields()

        num_points_self = len(self.points) if hasattr(self.points, '__len__') else 0
        num_points_other = len(other.points) if hasattr(other.points, '__len__') else 0

        # If one cloud is empty, just copy the other
        if num_points_self == 0:
            self._fields = copy.deepcopy(other._fields)
            self._create_properties()
            return
        if num_points_other == 0:
            return

        for name, self_field in self._fields.items():
            other_field = other._fields.get(name)
            if other_field is not None and other_field.size > 0:
                # Ensure self_field has data to concatenate with
                if self_field.size == 0 and num_points_self > 0:
                    # Initialize with default-like empty data of the correct length
                    if isinstance(self_field, VectorField):
                        self_field.data = np.zeros((num_points_self, self_field.num_columns))
                    else:
                        self_field.data = np.zeros(num_points_self)

                self_field.data = np.concatenate((self_field.data, other_field.data), axis=0)
        self._create_properties()

    def show(self, color_field: str = 'intensity') -> None:
        """ show PCD object """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        if color_field == 'rgb' and self.rgb.size > 0:
            colors = np.asarray(self.rgb)
            colors = colors / 255.0  # normalize RGB values
            pcd.colors = o3d.utility.Vector3dVector(colors)
        elif color_field in self._fields and self._fields[color_field].size > 0:
            field_values = np.asarray(self._fields[color_field].data)
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

        for name, field in self._fields.items():
            field.data = normalize(field.data)
        self.nan_to_zero()

    def shift_to_origin(self) -> None:
        """ shift points to origin (center of mass at zero) """
        if self.points.size > 0:
            self.points -= self.points.mean(axis=0)

    def shift_to_zero(self) -> None:
        """ shift points so that min values for all axes are zero """
        if self.points.size > 0:
            self.points -= self.points.min(axis=0)

    def shift_with_vector(self, shift_vector: np.ndarray) -> None:
        """ shift points by shift_vector """
        self.points = self.points - shift_vector
        if hasattr(self, 'shift'):
            self.shift += shift_vector
        else:
            self.shift = shift_vector

    def calculate_auto_shift_vector(self) -> np.ndarray:
        """ calculate auto shift vector for centering points cloud near zero """
        centroid = np.mean(self.points, axis=0)
        shift_vector = -centroid
        return shift_vector

    def nan_to_zero(self) -> None:
        """ replace NaN to 0 in all fields """
        for name, field in self._fields.items():
            field.data = np.nan_to_num(field.data)

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
        elif color_field in self._fields and self._fields[color_field].size > 0:
            field_values = np.asarray(self._fields[color_field].data)
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

    def poly_cut(self, polygon, algo='cm_parallel') -> 'PCD':
        idx_labels = np.where((self.points[:, 0] > min(polygon[:, 0])) & (self.points[:, 0] < max(polygon[:, 0])) &
                              (self.points[:, 1] > min(polygon[:, 1])) & (self.points[:, 1] < max(polygon[:, 1])))
        pc_part = self.clone()
        pc_part.index_cut(idx_labels)

        if algo == 'cm_parallel':
            idx_labels = is_inside_sm_parallel(pc_part.points, polygon)

        if algo == 'inpoly_parallel':
            idx_labels = parallelpointinpolygon(pc_part.points, polygon)

        if algo == 'ray_tracing':
            idx_labels = ray_tracing_numpy_numba(pc_part.points, polygon)

        if algo == 'postgis_parallel':
            idx_labels = is_inside_postgis_parallel(pc_part.points, polygon)

        pc_part.index_cut(idx_labels)
        return pc_part
