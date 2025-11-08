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
    Points, Intensity, RGB, Normals, OriginalCloudIndex, GPSTime, Illuminance, TreeID,
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
                'tree_id': TreeID()
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

    def save(self, file_path: str, split: int = None) -> None:
        """Saves the point cloud to a file, dispatching to the correct format handler."""
        file_format = file_path.split('.')[-1]

        @Timer(f"Сохранение файла {file_path}")
        def save_pcd(file_path, _np=np):
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

            dt = _np.hstack(pcd_data_list).astype(_np.float32)
            # Обеспечим C-смежность перед представлением как структурированный dtype
            dt = _np.ascontiguousarray(dt)
            num_points = dt.shape[0]

            md = {'version': .7, 'fields': pcd_fields,
                  'count': [1] * len(pcd_fields), 'width': num_points, 'height': 1,
                  'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 'points': num_points,
                  'type': ['F'] * len(pcd_fields), 'size': [4] * len(pcd_fields), 'data': 'binary'}

            dtype_list = [(name, _np.float32) for name in pcd_fields]
            pc_data = dt.view(_np.dtype(dtype_list)).squeeze()

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
    def read(cls, file_path: str, fields=None, txt_header: str = None, auto_oci: bool = False) -> 'PCD':
        """
        Читает файл и возвращает объект PCD.

        :param file_path: Путь до файла.
        :param fields: Набор полей для инициализации объекта.
        :param txt_header: Кастомная строка заголовка для txt (например, "// X Y Z ..."),
                            используется, если в файле отсутствует заголовок.
        :param auto_oci: Если True, автоматически определить колонку с 0/1 как original_cloud_index.
        """
        instance = cls(fields)
        instance.open(file_path, txt_header=txt_header, auto_oci=auto_oci)
        return instance

    def open(self, file_path: str, txt_header: str = None, auto_oci: bool = False) -> None:
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
                loaded = False
                # Пробуем все возможные маппинги поля на LAS-атрибуты
                for _, loader_func in field.las_attrs.items():
                    try:
                        field.data = loader_func(las)
                        loaded = True
                        break
                    except Exception:
                        continue
                # Если ничего не загрузилось — оставляем поле как есть (дополним позже нулями)

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

        # @Timer(f"Открытие файла {file_path}")
        def open_txt(self, file_path):
            with open(file_path, 'r') as file:
                header_line = file.readline().strip()

            # Определяем заголовок: из файла, либо из переданного txt_header
            if header_line.startswith('//'):
                header = [col.strip('/') for col in header_line.split()]
            elif txt_header is not None:
                provided = txt_header.strip()
                if provided.startswith('//'):
                    header = [col.strip('/') for col in provided.split()]
                else:
                    header = [col.strip('/') for col in provided.split()]
            else:
                raise ValueError(
                    f"Заголовок не найден в файле {file_path}. "
                    f"Ожидалась строка, начинающаяся с '//', либо задайте txt_header."
                )

            # Параметр `names` в pandas требует уникальных имен. Чтобы обработать
            # возможные дубликаты в заголовке файла, мы читаем данные без заголовка
            # и затем выбираем столбцы по их целочисленному индексу.
            # Используем движок 'python' и on_bad_lines='skip' для устойчивости к
            # строкам с некорректным форматом и ошибкам токенизации (ParserError).
            df = pd.read_csv(
                file_path,
                sep=r'\s+',
                comment='/',
                header=None,
                engine='python',
                on_bad_lines='skip'
            )

            # Ограничиваем заголовок реальным количеством столбцов данных
            num_cols = df.shape[1]
            if len(header) > num_cols:
                header = header[:num_cols]

            # Мы создаем отображение каждого уникального имени столбца на индекс его
            # первого появления. Это гарантирует, что если имя столбца дублируется,
            # мы будем рассматривать только первое из них, выполняя требование
            # "читать только уникальные".
            unique_header_map = {name: i for i, name in reversed(list(enumerate(header)))}

            used_indices = set()
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
                    unique_indices = [i for i in dict.fromkeys(col_indices) if i < num_cols]
                    # Проверяем, что собрали ровно необходимое количество столбцов
                    expected_cols = getattr(field, 'num_columns', 1)
                    if (isinstance(field, ScalarField) and len(unique_indices) == 1) or \
                       (not isinstance(field, ScalarField) and len(unique_indices) == expected_cols):
                        data = df.iloc[:, unique_indices].values
                        if isinstance(field, ScalarField):
                            data = data.ravel()
                        field.data = data
                        used_indices.update(unique_indices)

            # Автоопределение original_cloud_index по бинарной НЕ константной колонке (и 0, и 1)
            if auto_oci and 'original_cloud_index' in self._fields:
                oci_field = self._fields['original_cloud_index']
                need_override = getattr(oci_field, 'size', 0) == 0
                if not need_override:
                    try:
                        uniq = np.unique(np.nan_to_num(np.asarray(oci_field.data)))
                        # если текущее поле константное, пробуем переопределить
                        need_override = uniq.size == 1
                    except Exception:
                        need_override = True
                if need_override:
                    chosen_idx = None
                    for idx in range(num_cols):
                        if idx in used_indices:
                            continue
                        series = pd.to_numeric(df.iloc[:, idx], errors='coerce')
                        values = series.to_numpy()
                        if values.size == 0:
                            continue
                        values = np.nan_to_num(values, nan=0.0)
                        unique_vals = np.unique(values)
                        # Требуем строго два значения {0, 1}
                        if unique_vals.size == 2 and np.isin(unique_vals, [0.0, 1.0]).all():
                            chosen_idx = idx
                            break
                    if chosen_idx is not None:
                        series = pd.to_numeric(df.iloc[:, chosen_idx], errors='coerce').fillna(0)
                        oci_field.data = series.astype(np.int32).to_numpy().ravel()

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

    def index_cut(self, idx_labels: np.ndarray) -> 'PCD':
        """Subset all fields by indexes or boolean mask.

        Accepts:
        - integer index array
        - boolean mask (1D)
        - tuple from np.where (e.g., (array([...]),))
        """
        # Unwrap np.where output like (array([...]),)
        if isinstance(idx_labels, tuple):
            if len(idx_labels) != 1:
                raise ValueError("index_cut expects a 1D selector; got multi-axis tuple")
            idx_labels = idx_labels[0]

        idx_labels = np.asarray(idx_labels)

        # Determine target length for fallback allocations
        if idx_labels.dtype == bool:
            # Validate mask length if possible
            if idx_labels.ndim != 1:
                raise ValueError("Boolean mask for index_cut must be 1D")
            if self.points.shape[0] != idx_labels.shape[0]:
                raise ValueError(
                    f"Boolean mask length ({idx_labels.shape[0]}) must match the number of points ({self.points.shape[0]})"
                )
            target_len = int(idx_labels.sum())
        else:
            target_len = int(len(idx_labels))

        for name, field in self._fields.items():
            data = field.data
            # Safely try to index; if it fails (empty/mismatch), allocate empty of proper shape
            try:
                field.data = data[idx_labels]
            except Exception:
                # Build an empty array with the right length and shape
                if getattr(field, 'num_columns', None) is not None and getattr(data, 'ndim', 1) > 1:
                    shape = (target_len, field.num_columns)
                elif getattr(data, 'ndim', 1) > 1:
                    shape = (target_len, data.shape[1])
                else:
                    shape = (target_len,)
                # Preserve dtype similar to previous default
                dtype = field.default_value.dtype
                field.data = np.empty(shape, dtype=dtype)
        return self

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

    def show(self, color_field: str = 'intensity', labels=None) -> None:
        """ show PCD object """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        if labels is not None:
            labels = np.asarray(labels)
            unique_labels, inverse_indices = np.unique(labels, return_inverse=True)

            num_unique_labels = len(unique_labels)
            # Генерируем случайные цвета для каждого уникального кластера
            colors_for_labels = np.random.rand(num_unique_labels, 3)

            # Обрабатываем точки шума (метка -1), окрашивая их в серый цвет
            noise_label_index = np.where(unique_labels == -1)[0]
            if len(noise_label_index) > 0:
                colors_for_labels[noise_label_index[0]] = [0.5, 0.5, 0.5]

            # Сопоставляем цвета обратно исходным точкам, используя инверсные индексы
            colors = colors_for_labels[inverse_indices]
            pcd.colors = o3d.utility.Vector3dVector(colors)
        elif color_field == 'rgb' and self.rgb.size > 0:
            colors = np.asarray(self.rgb)
            colors = colors / 255.0  # normalize RGB values
            pcd.colors = o3d.utility.Vector3dVector(colors)
        elif color_field in self._fields and self._fields[color_field].size > 0:
            field_values = np.asarray(self._fields[color_field].data)
            range_val = field_values.max() - field_values.min()
            if range_val > 1e-9:
                field_values = (field_values - field_values.min()) / range_val
            else:
                field_values = np.zeros_like(field_values)

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
