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


def _create_voxel_grid_fast(points: np.ndarray, grid_cell_size: float):
    """
    Создает воксельную сетку для быстрого поиска соседей, используя
    векторизованные операции NumPy. Это значительно быстрее, чем подход
    с использованием Python-словарей и циклов.
    """
    num_points = points.shape[0]
    if num_points == 0:
        return None, None, None, None

    # 1. Определяем границы и размеры сетки (это и так было быстро)
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)

    min_bound -= grid_cell_size * 2  # Добавляем отступы для надежности
    max_bound += grid_cell_size * 2

    grid_dims = np.ceil((max_bound - min_bound) /
                        grid_cell_size).astype(np.int64)
    num_cells = np.prod(grid_dims)

    # 2. Векторизованно вычисляем хэш ячейки для КАЖДОЙ точки
    #    Это заменяет медленный Python-цикл.
    point_to_cell_idx = np.floor(
        (points - min_bound) / grid_cell_size).astype(np.int64)

    # Формула для получения уникального ID (хэша) для каждой 3D-ячейки
    cell_hashes = (point_to_cell_idx[:, 0] * grid_dims[1] * grid_dims[2] +
                   point_to_cell_idx[:, 1] * grid_dims[2] +
                   point_to_cell_idx[:, 2])

    # 3. Сортируем точки по хэшу их ячеек.
    #    `np.argsort` возвращает индексы, которые бы отсортировали массив.
    #    Это самая мощная часть оптимизации.
    sorted_indices = np.argsort(cell_hashes)

    # Применяем сортировку к хэшам и к исходным индексам точек
    sorted_hashes = cell_hashes[sorted_indices]
    # `point_indices_sorted` - это и есть наш главный массив, где индексы точек
    # сгруппированы по ячейкам. Он заменяет `np.concatenate(list(grid_dict.values()))`.
    point_indices_sorted = np.arange(
        num_points)[sorted_indices].astype(np.int32)

    # 4. Находим, где начинаются группы точек для каждой ячейки.
    #    `np.unique` с `return_index=True` очень эффективно находит
    #    первое вхождение каждого уникального хэша в отсортированном массиве.
    unique_hashes, first_indices = np.unique(sorted_hashes, return_index=True)

    # 5. Создаем таблицу для быстрого доступа `cell_starts_ends`.
    #    Она будет содержать начало и конец среза в `point_indices_sorted` для каждой ячейки.
    cell_starts_ends = np.zeros((num_cells, 2), dtype=np.int32)

    # Для всех ячеек, в которых есть точки...
    # ...записываем, где начинается их блок.
    cell_starts_ends[unique_hashes, 0] = first_indices.astype(np.int32)

    # Конец блока для одной ячейки - это начало блока для следующей.
    # Поэтому мы можем "сдвинуть" массив `first_indices` и добавить в конец
    # общее число точек.
    end_indices = np.append(first_indices[1:], num_points).astype(np.int32)
    cell_starts_ends[unique_hashes, 1] = end_indices

    return point_indices_sorted, cell_starts_ends, min_bound.astype(np.float32), grid_dims.astype(np.int32)


def _create_voxel_grid(points, grid_cell_size):
    """
    Создает воксельную сетку для быстрого поиска соседей в Numba.
    """
    if points.shape[0] == 0:
        return None, None, None, None

    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)

    # Расширяем границы, чтобы избежать проблем на краях
    min_bound -= grid_cell_size
    max_bound += grid_cell_size

    grid_dims = np.ceil((max_bound - min_bound) /
                        grid_cell_size).astype(np.int32)
    num_cells = np.prod(grid_dims)

    # Словарь для хранения точек в ячейках: {cell_idx: [point_idx1, point_idx2, ...]}
    # В Numba удобнее использовать списки списков или специальные структуры,
    # но для предобработки на CPU dict - это просто и понятно.
    grid_dict = {i: [] for i in range(num_cells)}

    point_to_cell_idx = np.floor(
        (points - min_bound) / grid_cell_size).astype(np.int32)
    for i, grid_coords in enumerate(point_to_cell_idx):
        cell_hash = grid_coords[0] * grid_dims[1] * grid_dims[2] + \
            grid_coords[1] * grid_dims[2] + \
            grid_coords[2]
        if 0 <= cell_hash < num_cells:
            grid_dict[cell_hash].append(i)

    # Преобразуем dict в массивы, которые Numba сможет использовать
    # `cell_starts_ends` хранит [начало, конец] среза в `point_indices_sorted` для каждой ячейки
    cell_starts_ends = np.zeros((num_cells, 2), dtype=np.int32)
    point_indices_sorted = np.concatenate(
        list(grid_dict.values())).astype(np.int32)

    current_pos = 0
    for i in range(num_cells):
        num_pts_in_cell = len(grid_dict[i])
        cell_starts_ends[i, 0] = current_pos
        cell_starts_ends[i, 1] = current_pos + num_pts_in_cell
        current_pos += num_pts_in_cell

    return point_indices_sorted, cell_starts_ends, min_bound, grid_dims


@njit(parallel=True, fastmath=True)
def _illuminance_kernel_numba(
    points,
    normals,
    num_rays,
    max_ray_distance,
    ao_neighbor_radius,
    num_steps,
    point_indices_sorted,
    cell_starts_ends,
    min_bound,
    grid_dims,
    grid_cell_size
):
    """
    Numba-ядро для расчета AO с использованием воксельной сетки.
    """
    num_points = points.shape[0]
    illuminance = np.zeros(num_points, dtype=np.float32)
    ao_neighbor_radius_sq = ao_neighbor_radius * \
        ao_neighbor_radius  # Сравниваем квадраты расстояний

    for i in prange(num_points):
        point_p = points[i]
        normal_p = normals[i]

        if np.linalg.norm(normal_p) < 1e-6:
            # Нейтральное значение для точек без нормали
            illuminance[i] = 0.5
            continue

        occluded_count = 0
        for r_idx in range(num_rays):
            # Генерация случайного луча в полусфере нормали
            ray_dir = np.random.randn(3).astype(np.float32)
            ray_dir /= np.linalg.norm(ray_dir)
            if np.dot(ray_dir, normal_p) < 0:
                ray_dir = -ray_dir

            is_occluded_this_ray = False
            for step in range(1, num_steps + 1):
                dist_along_ray = (step / num_steps) * max_ray_distance
                test_point = point_p + ray_dir * dist_along_ray

                # --- Логика поиска в воксельной сетке ---
                grid_coords = np.floor(
                    (test_point - min_bound) / grid_cell_size).astype(np.int32)

                # Итерируемся по текущей ячейке и 26 ее соседям (куб 3x3x3)
                for dz in range(-1, 2):
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            check_coords = grid_coords + \
                                np.array([dx, dy, dz])

                            # Проверяем, что координаты ячейки в пределах сетки
                            if (check_coords[0] >= 0 and check_coords[0] < grid_dims[0] and
                                check_coords[1] >= 0 and check_coords[1] < grid_dims[1] and
                                    check_coords[2] >= 0 and check_coords[2] < grid_dims[2]):

                                # Хэш ячейки для доступа к данным
                                cell_hash = check_coords[0] * grid_dims[1] * grid_dims[2] + \
                                    check_coords[1] * grid_dims[2] + \
                                    check_coords[2]

                                start, end = cell_starts_ends[cell_hash]
                                for pt_idx_in_sorted_array in range(start, end):
                                    # Индекс точки в исходном массиве `points`
                                    j = point_indices_sorted[pt_idx_in_sorted_array]

                                    # Пропускаем проверку с самой собой
                                    if i == j:
                                        continue

                                    # Проверяем расстояние
                                    vec_to_neighbor = points[j] - \
                                        test_point
                                    dist_sq = vec_to_neighbor[0]**2 + \
                                        vec_to_neighbor[1]**2 + \
                                        vec_to_neighbor[2]**2

                                    if dist_sq < ao_neighbor_radius_sq:
                                        is_occluded_this_ray = True
                                        break  # -> выход из цикла по точкам в ячейке

                                if is_occluded_this_ray:
                                    break  # -> выход из цикла по dx
                            if is_occluded_this_ray:
                                break  # -> выход из цикла по dy
                        if is_occluded_this_ray:
                            break  # -> выход из цикла по dz

                if is_occluded_this_ray:
                    occluded_count += 1
                    break  # -> выход из цикла по шагам луча (step)

        illuminance[i] = 1.0 - (occluded_count / num_rays)

    return illuminance


class PCD:
    def __init__(self,
                 points=np.empty((0, 3)),
                 intensity=np.empty(0),
                 rgb=np.empty((0, 3)),
                 original_cloud_index=np.empty(0),
                 gps_time=np.empty(0),
                 illuminance=np.empty(0),
                 normals=np.empty((0, 3))):
        self._points = points
        self.intensity = intensity
        self._rgb = rgb
        self.original_cloud_index = original_cloud_index
        self.gps_time = gps_time
        self.illuminance = illuminance
        self._normals = normals

    @property
    def df(self) -> pd.DataFrame:
        """ merge all fields in DataFrame """
        data = {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'intensity': self.intensity,
            'r': self.r,
            'g': self.g,
            'b': self.b,
            'original_cloud_index': self.original_cloud_index,
            'gps_time': self.gps_time,
            'illuminance': self.illuminance
        }
        return pd.DataFrame(data)

    def save(self, file_path: str, verbose: bool = False) -> None:
        def save_pcd(self, file_path, verbose=False):
            """ save .pcd """
            if verbose:
                print(f"Saving file {file_path} ...")
                start = time()
            dt = np.zeros((len(self.points), 8), dtype=np.float32)
            dt[:, :3] = self.points
            if self.rgb.size > 0:
                rgb = np.uint8(self.rgb)
                dt[:, 3] = pypcd.encode_rgb_for_pcl(rgb)
            dt[:, 4] = self.gps_time if self.gps_time.size > 0 else None
            dt[:, 5] = self.original_cloud_index if self.original_cloud_index.size > 0 else None
            dt[:, 6] = self.intensity if self.intensity.size > 0 else None
            dt[:, 7] = self.illuminance if self.illuminance.size > 0 else None
            md = {'version': .7,
                  'fields': ['x', 'y', 'z', 'rgb', 'GpsTime', 'Original_cloud_index', 'Intensity', 'Illuminance'],
                  'count': [1, 1, 1, 1, 1, 1, 1, 1],
                  'width': len(dt),
                  'height': 1,
                  'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                  'points': len(dt),
                  'type': ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                  'size': [4, 4, 4, 4, 4, 4, 4, 4],
                  'data': 'binary'}
            pc_data = dt.view(np.dtype([('x', np.float32),
                                        ('y', np.float32),
                                        ('z', np.float32),
                                        ('rgb', np.float32),
                                        ('GpsTime', np.float32),
                                        ('Original_cloud_index', np.float32),
                                        ('Intensity', np.float32),
                                        ('Illuminance', np.float32)])).squeeze()

            new_cloud = pypcd.PointCloud(md, pc_data)
            new_cloud.save_pcd(file_path, 'binary')
            if verbose:
                end = time()-start
                print(f"Time saving: {end:.3f} s")

        def save_las(self, file_path, verbose=False):
            """" save .las """
            if verbose:
                print(f"Saving file {file_path} ...")
                start = time()
            header = laspy.LasHeader(point_format=3, version="1.4")
            header.point_count = len(self.points)
            las = laspy.LasData(header)
            las.add_extra_dim(laspy.ExtraBytesParams(
                name="illuminance", type=np.float32))
            self.points = np.asarray(self.points, dtype=np.float32)
            las.x = self.points[:, 0]
            las.y = self.points[:, 1]
            las.z = self.points[:, 2]
            if self.rgb.size > 0:
                rgb = self.rgb.astype(np.uint16)
                las.red = rgb[:, 0] * 256
                las.green = rgb[:, 1] * 256
                las.blue = rgb[:, 2] * 256
            if self.intensity.size > 0:
                las.intensity = self.intensity
            if self.illuminance.size > 0:
                las.illuminance = self.illuminance
            if self.gps_time.size > 0:
                las.gps_time = self.gps_time
            if self.original_cloud_index.size > 0:
                las.point_source_id = self.original_cloud_index
            las.write(file_path)
            if verbose:
                end = time()-start
                print(f"Time saving: {end:.3f} s")

        def save_laz(self, file_path, verbose=False):
            """" save .laz """
            if verbose:
                print(f"Saving file {file_path} ...")
                start = time()
            header = laspy.LasHeader(point_format=3, version="1.4")
            header.point_count = len(self.points)
            las = laspy.LasData(header)
            las.add_extra_dim(laspy.ExtraBytesParams(
                name="illuminance", type=np.float32))

            self.points = np.asarray(self.points, dtype=np.float32)
            las.x = self.points[:, 0]
            las.y = self.points[:, 1]
            las.z = self.points[:, 2]
            if self.rgb.size > 0:
                rgb = self.rgb.astype(np.uint16)
                las.red = rgb[:, 0] * 256
                las.green = rgb[:, 1] * 256
                las.blue = rgb[:, 2] * 256
            if self.intensity.size > 0:
                las.intensity = self.intensity
            if self.illuminance.size > 0:
                las.illuminance = self.illuminance
            if self.gps_time.size > 0:
                las.gps_time = self.gps_time
            if self.original_cloud_index.size > 0:
                las.point_source_id = self.original_cloud_index
            las.write(file_path)
            if verbose:
                end = time()-start
                print(f"Time saving: {end:.3f} s")

        def save_csv(self, file_path, verbose=False):
            """" save .csv """
            if verbose:
                print(f"Saving file {file_path} ...")
                start = time()
            data = {}
            if self.points.size > 0:
                points = np.asarray(self.points)
                data["x"] = points[:, 0]
                data["y"] = points[:, 1]
                data["z"] = points[:, 2]
            if self.intensity.size > 0:
                data["Intensity"] = self.intensity
            if self.illuminance.size > 0:
                data["Illuminance"] = self.illuminance
            if self.gps_time.size > 0:
                data["GpsTime"] = self.gps_time
            if self.original_cloud_index.size > 0:
                data["Original_cloud_index"] = self.original_cloud_index
            if self.rgb.size > 0:
                rgb = np.asarray(self.rgb)
                data["red"] = rgb[:, 0]
                data["green"] = rgb[:, 1]
                data["blue"] = rgb[:, 2]
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            if verbose:
                end = time()-start
                print(f"Time saving: {end:.3f} s")

        def save_txt(self, file_path, verbose=False):
            """ save .txt """
            if verbose:
                print(f"Saving file {file_path} ...")
                start = time()
            # Determine the columns to write based on available data
            columns_to_write = []
            if self.points.size > 0:
                columns_to_write.extend(['X', 'Y', 'Z'])
            if self.intensity.size > 0:
                columns_to_write.append('Intensity')
            if self.rgb.size > 0:
                columns_to_write.extend(['R', 'G', 'B'])
            if self.original_cloud_index.size > 0:
                columns_to_write.append('Original_cloud_index')
            if self.gps_time.size > 0:
                columns_to_write.append('GpsTime')
            if self.illuminance.size > 0:
                columns_to_write.append('Illuminance_(PCV)')

            # Write the file
            with open(file_path, 'w') as file:
                # Write the header line
                header_line = '//' + ' '.join(columns_to_write)
                file.write(header_line + '\n')

                # Write the data lines
                num_points = len(self.points) if self.points.size > 0 else 0
                for i in range(num_points):
                    values = []
                    if self.points.size > 0:
                        values.extend(self.points[i])
                    if self.intensity.size > 0:
                        values.append(self.intensity[i])
                    if self.rgb.size > 0:
                        values.extend(self.rgb[i])
                    if self.original_cloud_index.size > 0:
                        values.append(self.original_cloud_index[i])
                    if self.gps_time.size > 0:
                        values.append(self.gps_time[i])
                    if self.illuminance.size > 0:
                        values.append(self.illuminance[i])
                    line = ' '.join(map(str, values))
                    file.write(line + '\n')

            if verbose:
                end = time()-start
                print(f"Time saving: {end:.3f} s")

        def save_h5(self, file_path, verbose=False):
            """Save data to an HDF5 file."""
            if verbose:
                start = time()
                print(f"Saving data to {file_path} ...")

            with h5py.File(file_path, 'w') as h5f:
                if self.points.size > 0:
                    h5f.create_dataset('points', data=self.points)
                if self.intensity.size > 0:
                    h5f.create_dataset('Intensity', data=self.intensity)
                if self.rgb.size > 0:
                    h5f.create_dataset('rgb', data=self.rgb)
                if self.original_cloud_index.size > 0:
                    h5f.create_dataset('Original_cloud_index',
                                       data=self.original_cloud_index)
                if self.gps_time.size > 0:
                    h5f.create_dataset('GpsTime', data=self.gps_time)
                if self.illuminance.size > 0:
                    h5f.create_dataset('Illuminance',
                                       data=self.illuminance)

            if verbose:
                end = time() - start
                print(f"Time saving data: {end:.3f} s")

        if file_path.endswith('.pcd'):
            save_pcd(self, file_path, verbose=verbose)
        elif file_path.endswith('.las'):
            save_las(self, file_path, verbose=verbose)
        elif file_path.endswith('.laz'):
            save_laz(self, file_path, verbose=verbose)
        elif file_path.endswith('.csv'):
            save_csv(self, file_path, verbose=verbose)
        elif file_path.endswith('.txt'):
            save_txt(self, file_path, verbose=verbose)
        elif file_path.endswith('.h5'):
            save_h5(self, file_path, verbose=verbose)
        else:
            print("invalid format")

    @classmethod
    def read(cls, file_path: str, verbose: bool = False) -> 'PCD':
        instance = cls()
        instance.open(file_path, verbose=verbose)
        return instance

    def open(self, file_path: str, verbose: bool = False) -> None:
        def open_pcd(self, file_path, verbose=False):
            """ open .pcd """
            if verbose:
                start = time()
                print(f"Opening file {file_path} ...")
            cloud = pypcd.PointCloud.from_path(file_path)
            data = cloud.pc_data.view(np.float32).reshape(
                cloud.pc_data.shape + (-1,))
            ix = cloud.get_metadata()["fields"].index('x')
            self.points = data[:, ix:ix + 3]
            try:
                ii = cloud.get_metadata()["fields"].index('Intensity')
                self.intensity = np.nan_to_num(np.asarray(data[:, ii]))
            except ValueError:
                self.intensity = np.empty(0)
            try:
                il = cloud.get_metadata()["fields"].index('Illuminance')
                self.illuminance = np.nan_to_num(np.asarray(data[:, il]))
            except ValueError:
                self.illuminance = np.empty(0)
            try:
                ir = cloud.get_metadata()["fields"].index('rgb')
                rgb = pypcd.decode_rgb_from_pcl(data[:, ir])
                self.rgb = np.nan_to_num(rgb)
            except ValueError:
                self.rgb = np.empty((0, 3))
            try:
                ig = cloud.get_metadata()["fields"].index('GpsTime')
                self.gps_time = np.nan_to_num(np.asarray(data[:, ig]))
            except ValueError:
                self.gps_time = np.empty(0)
            try:
                iid = cloud.get_metadata()["fields"].index(
                    'Original_cloud_index')
                self.original_cloud_index = np.nan_to_num(
                    np.asarray(data[:, iid]))
            except ValueError:
                self.original_cloud_index = np.empty(0)
            if verbose:
                end = time()-start
                print(f"Time stacking data: {end:.3f} s")

        def open_h5(self, file_path, verbose=False):
            """ open .h5 """
            if verbose:
                start = time()
                print(f"Opening file {file_path} ...")
            h5f = h5py.File(file_path, 'r')
            try:
                self.points = np.asarray(h5f.get('points'))
            except:
                self.points = np.empty((0, 3))
            try:
                self.intensity = np.asarray(h5f.get('Intensity'))
            except:
                self.intensity = np.empty(0)
            try:
                self.illuminance = np.asarray(h5f.get('Illuminance'))
            except:
                self.illuminance = np.empty(0)
            try:
                self.rgb = np.asarray(h5f.get('rgb'))
            except:
                self.rgb = np.empty((0, 3))
            try:
                self.gps_time = np.asarray(h5f.get('GpsTime'))
            except:
                self.gps_time = np.empty(0)
            try:
                self.original_cloud_index = np.asarray(
                    h5f.get('Original_cloud_index'))
            except:
                self.original_cloud_index = np.empty(0)
            h5f.close()
            if verbose:
                end = time()-start
                print(f"Time stacking data: {end:.3f} s")

        def open_las(self, file_path, verbose=False):
            """ open .las """
            if verbose:
                start = time()
                print(f"Opening file {file_path} ...")
            las = laspy.read(file_path)
            points = np.vstack(
                [las.points.x, las.points.y, las.points.z]).transpose()
            self.points = points
            try:
                self.intensity = las.intensity
            except:
                self.intensity = np.empty(0)  # np.full(points.shape[0], 0)
            try:
                self.illuminance = las.illuminance
            except:
                self.illuminance = np.empty(0)
            try:
                rgb = np.vstack(
                    [las.points.red, las.points.green, las.points.blue]).transpose()
                self.rgb = (rgb // 256).astype(np.uint8)
            except:
                # np.zeros((points.shape[0], 3), dtype=np.int32)
                self.rgb = np.empty((0, 3))
            try:
                self.original_cloud_index = las.point_source_id
            except:
                self.original_cloud_index = np.empty(0)
            try:
                self.gps_time = las.gps_time
            except:
                self.gps_time = np.empty(0)
            if verbose:
                end = time()-start
                print(f"Time stacking data: {end:.3f} s")

        def open_laz(self, file_path, verbose=False):
            """ open .laz """
            if verbose:
                start = time()
                print(f"Opening file {file_path} ...")
            with laspy.open(file_path) as fh:
                las = fh.read()
                points = np.vstack(
                    [las.points.x, las.points.y, las.points.z]).transpose()
                self.points = points
                try:
                    self.intensity = np.nan_to_num(
                        np.asarray(las.intensity, dtype=np.int32))
                except:
                    self.intensity = np.empty(0)  # np.full(points.shape[0], 0)
                try:
                    self.illuminance = np.nan_to_num(
                        np.asarray(las.illuminance, dtype=np.int32))
                except:
                    self.illuminance = np.empty(0)
                try:
                    rgb = np.vstack(
                        [las.points.red, las.points.green, las.points.blue]).transpose()
                    self.rgb = (rgb // 256).astype(np.uint8)
                except:
                    # np.zeros((points.shape[0], 3), dtype=np.int32)
                    self.rgb = np.empty((0, 3))
                try:
                    self.original_cloud_index = np.nan_to_num(np.asarray(
                        las.point_source_id, dtype=np.float16))
                except:
                    self.original_cloud_index = np.empty(0)
                try:
                    self.gps_time = np.nan_to_num(
                        np.asarray(las.gps_time, dtype=np.float16))
                except AttributeError:
                    self.gps_time = np.empty(0)
            if verbose:
                end = time()-start
                print(f"Time stacking data: {end:.3f} s")

        def open_csv(self, file_path, verbose=False):
            """ open .csv """
            if verbose:
                start = time()
                print(f"Opening file {file_path} ...")
            df = pd.read_csv(file_path)
            self.points = df[['x', 'y', 'z']
                             ].values if 'x' in df.columns else np.empty((0, 3))
            self.intensity = df['Intensity'].values if 'Intensity' in df.columns else np.empty(
                0)
            self.gps_time = df['GpsTime'].values if 'GpsTime' in df.columns else np.empty(
                0)
            self.original_cloud_index = df['Original_cloud_index'].values if 'Original_cloud_index' in df.columns else np.empty(
                0)
            self.rgb = df[['red', 'green', 'blue']
                          ].values if 'red' in df.columns else np.empty((0, 3))
            self.illuminance = df['Illuminance'].values if 'Illuminance' in df.columns else np.empty(
                0)
            if verbose:
                end = time()-start
                print(f"Time stacking data: {end:.3f} s")

        def open_txt(self, file_path, verbose=False):
            """ open .txt """

            if verbose:
                start = time()
                print(f"Opening file {file_path} ...")
            # Read the file
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Read the header line
            header = [
                col.strip('//') for col in lines[0].strip().split() if col.startswith('//')]
            if lines[0].startswith('//'):
                header = [col.strip('//') for col in lines[0].split()]
            else:
                if verbose:
                    print("Header is empty. Using default column names.")
                header = ['X', 'Y', 'Z', 'Intensity',
                          'R', 'G', 'B', 'Original_cloud_index', 'Gps_Time', 'Illuminance_(PCV)']
            # Initialize dictionaries to store data
            data = {col: [] for col in header}

            # Read the data lines
            for line in lines[1:]:
                values = line.strip().split()
                for col, value in zip(header, values):
                    data[col].append(float(value))

            # Initialize dictionaries to store data
            data = {col: [] for col in header if not col.startswith('//')}

            # Read the data lines
            for line in lines[1:]:
                values = line.strip().split()
                for col, value in zip(header, values):
                    if col.startswith('//'):
                        continue
                    data[col].append(float(value))

            # Convert lists to numpy arrays for easier manipulation
            for col in data:
                data[col] = np.array(data[col])

            # Assign data to attributes
            if 'X' in data and 'Y' in data and 'Z' in data:
                self.points = np.vstack(
                    (data['X'], data['Y'], data['Z'])).T
            if 'Intensity' in data:
                self.intensity = data['Intensity']
            if 'R' in data and 'G' in data and 'B' in data:
                self.rgb = np.vstack((data['R'], data['G'], data['B'])).T
            if 'Original_cloud_index' in data:
                self.original_cloud_index = data['Original_cloud_index']
            if 'Gps_Time' in data:
                self.gps_time = data['Gps_Time']
            if 'Illuminance_(PCV)' in data:
                self.illuminance = data['Illuminance_(PCV)']
            if verbose:
                end = time()-start
                print(f"Time stacking data: {end:.3f} s")

        if file_path.endswith(".h5"):
            open_h5(self, file_path, verbose=verbose)
        elif file_path.endswith('.pcd'):
            open_pcd(self, file_path, verbose=verbose)
        elif file_path.endswith('.las'):
            open_las(self, file_path, verbose=verbose)
        elif file_path.endswith('.laz'):
            open_laz(self, file_path, verbose=verbose)
        elif file_path.endswith('.csv'):
            open_csv(self, file_path, verbose=verbose)
        elif file_path.endswith('.txt'):
            open_txt(self, file_path, verbose=verbose)
        else:
            print("invalid format")
        self.check_and_pad_fields()

    def check_and_pad_fields(self):
        """ check if all fields have the same length, and pad with zeros if not """

        len_points = len(self.points) if self.points is not None else 0
        len_intensity = len(
            self.intensity) if self.intensity is not None else 0
        len_rgb = len(self.rgb) if self.rgb is not None else 0
        len_original_cloud_index = len(
            self.original_cloud_index) if self.original_cloud_index is not None else 0
        len_gps_time = len(self.gps_time) if self.gps_time is not None else 0
        len_illuminance = len(
            self.illuminance) if self.illuminance is not None else 0

        max_length = max(len_points, len_intensity, len_rgb,
                         len_original_cloud_index, len_gps_time, len_illuminance)

        if len_points < max_length:
            padding = np.zeros((max_length - len_points, 3))
            self.points = np.vstack((self.points, padding))

        if len_intensity < max_length:
            padding = np.zeros(max_length - len_intensity)
            if self.intensity is not None:
                self.intensity = np.hstack((self.intensity, padding))
            else:
                self.intensity = padding

        if len_rgb < max_length:
            padding = np.zeros((max_length - len_rgb, 3))
            if self.rgb is not None:
                self.rgb = np.vstack((self.rgb, padding))
            else:
                self.rgb = padding

        if len_original_cloud_index < max_length:
            padding = np.zeros(max_length - len_original_cloud_index)
            if self.original_cloud_index is not None:
                self.original_cloud_index = np.hstack(
                    (self.original_cloud_index, padding))
            else:
                self.original_cloud_index = padding

        if len_gps_time < max_length:
            padding = np.zeros(max_length - len_gps_time)
            if self.gps_time is not None:
                self.gps_time = np.hstack((self.gps_time, padding))
            else:
                self.gps_time = padding

        if len_illuminance < max_length:
            padding = np.zeros(max_length - len_illuminance)
            if self.illuminance is not None:
                self.illuminance = np.hstack((self.illuminance, padding))
            else:
                self.illuminance = padding

    def clone(self) -> 'PCD':
        """ clone PCD object """
        return copy.deepcopy(self)

    def sample_fps(self, num_sample: int, verbose: bool = False) -> None:
        """ sampling 'num_sample' points from 'PCD' class via farthest point sampling algorithm """
        start = time()
        if verbose:
            end = time() - start
            print(f"Time sampling (fps): {end:.3f} s")
        np_points = np.asarray([self.points])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        points_torch = torch.Tensor(np_points).to(device)
        centroids = fps.farthest_point_sample(points_torch, num_sample)
        pt_sampled = points_torch[0][centroids[0]]
        centroids = centroids.cpu().data.numpy()
        self.intensity = self.intensity[centroids[0]]
        self.rgb = self.rgb[centroids[0]]
        self.original_cloud_index = self.original_cloud_index[centroids[0]]
        self.gps_time = self.gps_time[centroids[0]]
        self.illuminance = self.illuminance[centroids[0]]
        self.points = pt_sampled.cpu().detach().numpy()
        if hasattr(self, 'normals'):
            self.normals = self.normals[centroids[0]]

    def index_cut(self, idx_labels: np.ndarray) -> None:
        """ cut points and intensity using indexes """
        # TODO: fix normals
        self.points = self.points[idx_labels]
        try:
            self.intensity = self.intensity[idx_labels]
        except:
            self.intensity = np.empty(0)
        try:
            self.original_cloud_index = self.original_cloud_index[idx_labels]
        except:
            self.original_cloud_index = np.empty(0)
        try:
            self.gps_time = self.gps_time[idx_labels]
        except:
            self.gps_time = np.empty(0)
        try:
            self.rgb = self.rgb[idx_labels]
        except:
            self.rgb = np.empty((0, 3))
        try:
            self.illuminance = self.illuminance[idx_labels]
        except:
            self.illuminance = np.empty(0)
        if hasattr(self, 'normals'):
            try:
                self.normals = self.normals[idx_labels]
            except:
                self.normals = np.empty((0, 3))

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

    def calculate_illuminance(self,
                              num_rays: int = 32,
                              max_ray_distance: float = 0.5,
                              ao_neighbor_radius: float = 0.02,
                              normal_est_radius: float = None,
                              normal_est_max_nn: int = 30,
                              force_normal_recalculation: bool = False,
                              verbose: bool = False) -> None:
        """
        Calculates ambient occlusion for each point and stores it in the `illuminance` attribute.

        This method simulates light rays originating from each point and checks for occlusions
        by nearby points. The illuminance value is proportional to the number of unoccluded rays.
        A lower value means more occlusion and less light.

        Args:
            num_rays (int): Number of rays to cast from each point.
            max_ray_distance (float): Maximum distance to check for occluders.
            ao_neighbor_radius (float): Radius to search for neighbors around points on the ray.
            normal_est_radius (float): The radius to search for neighbors for normal estimation.
                                       If None, it's set to `max_ray_distance / 2`.
            normal_est_max_nn (int): The maximum number of neighbours to search for normal estimation.
            force_normal_recalculation (bool): If True, normals will be re-estimated even if they exist.
            verbose (bool): If True, prints progress information.
        """
        num_points = len(self.points)
        if num_points == 0:
            logger.debug("No points to calculate illuminance.")
            return

        if normal_est_radius is None:
            normal_est_radius = max_ray_distance / 2

        # 1. Ensure normals are available.
        if self._normals is None or self._normals.shape[0] != num_points or force_normal_recalculation:
            logger.debug(
                "Normals not available or recalculation forced. Estimating normals...")
            self.estimate_normals(radius=normal_est_radius,
                                  max_nn=normal_est_max_nn)

        # 2. Build a KDTree for efficient neighbor searches.
        logger.debug("Building KDTree for AO calculation...")
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(self.points)
        kdtree = o3d.geometry.KDTreeFlann(pcd_o3d)

        if self.illuminance is None or self.illuminance.shape[0] != num_points:
            self.illuminance = np.zeros(num_points, dtype=np.float32)

        logger.debug(
            f"Calculating Ambient Occlusion for {num_points} points...")
        logger.debug(
            f"  Parameters: num_rays={num_rays}, max_ray_dist={max_ray_distance}, ao_neighbor_radius={ao_neighbor_radius}")

        # 3. Iterate over each point to calculate its illuminance.
        for i in tqdm(range(num_points), desc="Calculating Ambient Occlusion"):

            point_p = self.points[i]
            normal_p = self._normals[i]

            if np.linalg.norm(normal_p) < 1e-6:
                # Default for points with no valid normal
                self.illuminance[i] = 0.5
                continue

            rays = self._generate_hemisphere_rays(normal_p, num_rays)
            occluded_count = 0

            # 4. For each ray, check for occlusion.
            for ray_dir in rays:
                is_occluded_this_ray = False
                # We sample points along the ray and check if any geometry is nearby.
                num_steps = 10
                for step in range(1, num_steps + 1):
                    dist_along_ray = (step / num_steps) * max_ray_distance
                    test_point_on_ray = point_p + ray_dir * dist_along_ray

                    # Search for neighbors around the sample point on the ray.
                    [k, idx, _] = kdtree.search_radius_vector_3d(
                        test_point_on_ray, ao_neighbor_radius)

                    if k > 1 or (k == 1 and idx[0] != i):
                        is_occluded_this_ray = True
                        break

                if is_occluded_this_ray:
                    occluded_count += 1

            # 5. Illuminance is the ratio of unoccluded rays.
            self.illuminance[i] = 1.0 - (occluded_count / num_rays)\


    def calculate_illuminance_fast(self,
                                   num_rays: int = 32,
                                   max_ray_distance: float = 0.5,
                                   ao_neighbor_radius: float = 0.02,
                                   normal_est_radius: float = None,
                                   normal_est_max_nn: int = 30,
                                   force_normal_recalculation: bool = False,
                                   verbose: bool = False) -> None:
        """
        Быстрый и полностью параллельный расчет Ambient Occlusion с использованием numba
        и пространственной сетки (Voxel Grid).
        """
        num_points = len(self.points)
        if num_points == 0:
            logger.debug("Нет точек для расчета освещенности.")
            return

        if normal_est_radius is None:
            normal_est_radius = max_ray_distance / 2

        # Проверка и расчет нормалей
        if not hasattr(self, '_normals') or self._normals is None or self._normals.shape[0] != num_points or force_normal_recalculation:
            logger.debug(
                "Нормали отсутствуют или требуется пересчет. Оцениваем нормали...")
            self.estimate_normals(radius=normal_est_radius,
                                  max_nn=normal_est_max_nn)

        points = self.points.astype(np.float32)
        normals = self._normals.astype(np.float32)

        # 1. Создаем воксельную сетку для быстрого поиска
        # Размер ячейки сетки лучше всего выбирать равным радиусу поиска
        grid_cell_size = ao_neighbor_radius
        logger.debug(
            f"Создание воксельной сетки с размером ячейки {grid_cell_size:.4f}...")
        point_indices_sorted, cell_starts_ends, min_bound, grid_dims = _create_voxel_grid_fast(
            points, grid_cell_size)

        # 2. Запускаем единый, полностью скомпилированный Numba-kernel
        logger.debug("Запуск Numba-ядра для расчета AO...")

        # Количество шагов вдоль луча. 10-20 обычно достаточно.
        num_steps = 10

        # tqdm можно обернуть вокруг вызова ядра, если хочется видеть общий прогресс,
        # но это не покажет прогресс внутри параллельного цикла.
        # Для отладки можно убрать `parallel=True` и обернуть `prange` в `tqdm`.

        illuminance = _illuminance_kernel_numba(
            points, normals, num_rays, max_ray_distance, ao_neighbor_radius, num_steps,
            point_indices_sorted, cell_starts_ends, min_bound, grid_dims, grid_cell_size
        )

        logger.debug("Расчет AO завершен.")
        self.illuminance = illuminance

    def estimate_normals(self, radius: float = 0.1, max_nn: int = 30) -> None:
        """ estimate normals """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        self._normals = np.asarray(pcd.normals)

    def get_normals(self, radius: float = 0.1, max_nn: int = 30) -> np.ndarray:
        """ get normals """
        if self._normals is None:
            logger.debug("Estimating normals")
            self.estimate_normals(radius=radius, max_nn=max_nn)
        return self._normals

    @property
    def normals(self) -> np.ndarray:
        return self.get_normals()

    def unique(self) -> None:
        """ leaves only unique point values """
        self.points, unique_indices = np.unique(
            self.points, axis=0, return_index=True)
        self.intensity = np.take(self.intensity, unique_indices)
        self.rgb = np.take(self.rgb, unique_indices, axis=0)
        self.original_cloud_index = np.take(
            self.original_cloud_index, unique_indices)
        self.gps_time = np.take(self.gps_time, unique_indices)
        self.illuminance = np.take(self.illuminance, unique_indices)

    def append(self, other: 'PCD') -> None:
        """ append PCD object """
        if not isinstance(other, PCD):
            raise TypeError("Argument must be an instance of PCD")
        self.points = np.concatenate((self.points, other.points), axis=0)
        self.intensity = np.concatenate(
            (self.intensity, other.intensity), axis=0)
        self.rgb = np.concatenate((self.rgb, other.rgb), axis=0)
        self.original_cloud_index = np.concatenate(
            (self.original_cloud_index, other.original_cloud_index), axis=0)
        self.gps_time = np.concatenate((self.gps_time, other.gps_time), axis=0)
        self.illuminance = np.concatenate(
            (self.illuminance, other.illuminance), axis=0)
        if hasattr(other, 'normals'):
            self.normals = np.concatenate(
                (self.normals, other.normals), axis=0)

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
        """ normalize fields """
        def normalize(array: np.ndarray) -> np.ndarray:
            if array.size > 0:
                return (array - array.min()) / (array.max() - array.min())
            return array
        self.points = normalize(self.points)
        self.intensity = normalize(self.intensity)
        self.rgb = normalize(self.rgb)
        self.original_cloud_index = normalize(self.original_cloud_index)
        self.gps_time = normalize(self.gps_time)
        self.illuminance = normalize(self.illuminance)
        self.nan_to_zero()

    def shift_to_origin(self) -> None:
        """ shift points to origin """
        self.points = self.points - self.points.mean(axis=0)

    def shift_to_zero(self) -> None:
        """ shift points to zero """
        self.points = self.points - self.points.min(axis=0)

    def nan_to_zero(self) -> None:
        """ replace NaN to 0 """
        self.points = np.nan_to_num(self.points)
        self.intensity = np.nan_to_num(self.intensity)
        self.rgb = np.nan_to_num(self.rgb)
        self.original_cloud_index = np.nan_to_num(self.original_cloud_index)
        self.gps_time = np.nan_to_num(self.gps_time)
        self.illuminance = np.nan_to_num(self.illuminance)

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

    @normals.setter
    def normals(self, value):
        self._normals = value

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        self._points = value

    @property
    def x(self):
        return self._points[:, 0]

    @x.setter
    def x(self, value):
        self._points[:, 0] = value

    @property
    def y(self):
        return self._points[:, 1]

    @y.setter
    def y(self, value):
        self._points[:, 1] = value

    @property
    def z(self):
        return self._points[:, 2]

    @z.setter
    def z(self, value):
        self._points[:, 2] = value

    @property
    def rgb(self):
        return self._rgb

    @rgb.setter
    def rgb(self, value):
        self._rgb = value

    @property
    def r(self):
        return self._rgb[:, 0]

    @r.setter
    def r(self, value):
        self._rgb[:, 0] = value

    @property
    def g(self):
        return self._rgb[:, 1]

    @g.setter
    def g(self, value):
        self._rgb[:, 1] = value

    @property
    def b(self):
        return self._rgb[:, 2]

    @b.setter
    def b(self, value):
        self._rgb[:, 2] = value
