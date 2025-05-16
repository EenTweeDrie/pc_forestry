import numpy as np
from .PCD import PCD
from collections import defaultdict
from tqdm import tqdm
import open3d as o3d
import pandas as pd
import copy

from loguru import logger


class VOXEL(PCD):
    distance = None
    _mean_normals = None

    def __init__(self, index):
        super().__init__()
        self.index = index

    @property
    def num_points(self):
        return self.points.shape[0]

    @property
    def mean_intensity(self):
        return np.mean(self.intensity) if self.intensity.size > 0 else None

    @property
    def mean_rgb(self):
        return np.mean(self.rgb, axis=0) if self.rgb.size > 0 else None

    @property
    def mean_illuminance(self):
        return np.mean(self.illuminance) if self.illuminance.size > 0 else None

    @property
    def mean_gps_time(self):
        return np.mean(self.gps_time) if self.gps_time.size > 0 else None

    def estimate_mean_normals(self):
        self._mean_normals = np.mean(
            self.normals, axis=0) if self.normals.size > 0 else None

    def get_mean_normals(self):
        if self._mean_normals is None:
            self.estimate_mean_normals()
        return self._mean_normals

    @property
    def mean_normals(self):
        return self.get_mean_normals()

    @property
    def label(self):
        return np.bincount(self.original_cloud_index.astype(np.int64)).argmax()\


    def add_point(self,
                  point,
                  intensity=None,
                  rgb=None,
                  original_cloud_index=None,
                  gps_time=None,
                  illuminance=None,
                  normals=None):
        self.points = np.append(self.points, [point], axis=0)
        if intensity is not None:
            self.intensity = np.append(self.intensity, intensity)
        if rgb is not None:
            self.rgb = np.append(self.rgb, [rgb], axis=0)
        if original_cloud_index is not None:
            self.original_cloud_index = np.append(
                self.original_cloud_index, original_cloud_index)
        if gps_time is not None:
            self.gps_time = np.append(self.gps_time, gps_time)
        if illuminance is not None:
            self.illuminance = np.append(self.illuminance, illuminance)
        if normals is not None:
            self.normals = np.append(self.normals, [normals], axis=0)

    def calculate_center(self, voxel_size):
        return (np.array(self.index) + 0.5) * voxel_size

    def calculate_center_of_points(self):
        return np.mean(self.points, axis=0)

    def closest_point_to_center(self):
        center = self.calculate_center()
        if center is None:
            return None
        distances = np.linalg.norm(self.points - center, axis=1)
        return self.points[np.argmin(distances)]

    def __str__(self):
        return f"Voxel(grid_index={self.index}, points={self.num_points})"

    def __repr__(self):
        return self.__str__()


class VOXELGRID:
    def __init__(self, PC: PCD, voxel_size: float):
        self.PC = PC
        self.voxel_size = voxel_size
        self.voxels = None

    @classmethod
    def create(cls, PC: PCD, voxel_size: float, verbose: bool = False):
        voxel_indices = np.floor(PC.points / voxel_size).astype(np.int32)
        voxel_dict = defaultdict(VOXEL)

        if verbose:
            iterator = tqdm(enumerate(zip(PC.points, voxel_indices)), total=len(
                PC.points), desc="Creating voxel grid")
        else:
            iterator = enumerate(zip(PC.points, voxel_indices))

        for i, (point, index) in iterator:
            index_tuple = tuple(index)
            if index_tuple not in voxel_dict:
                voxel_dict[index_tuple] = VOXEL(index_tuple)
            voxel_dict[index_tuple].add_point(
                point,
                PC.intensity[i] if PC.intensity.size > 0 else None,
                PC.rgb[i] if PC.rgb.size > 0 else None,
                PC.original_cloud_index[i] if PC.original_cloud_index.size > 0 else None,
                PC.gps_time[i] if PC.gps_time.size > 0 else None,
                PC.illuminance[i] if PC.illuminance.size > 0 else None,
                PC.normals[i] if PC.normals.size > 0 else None
            )

        instance = cls(PC, voxel_size)
        instance.voxels = list(voxel_dict.values())

        return instance

    def __getitem__(self, index: int) -> VOXEL:
        return self.voxels[index]

    def __len__(self) -> int:
        return len(self.voxels)

    def get_voxel_by_grid_index(self, index: tuple) -> VOXEL:
        for voxel in self.voxels:
            if voxel.index == index:
                return voxel
        return None

    def clone(self):
        return copy.deepcopy(self)

    @property
    def df(self) -> pd.DataFrame:
        """ merge all fields of voxels in DataFrame """
        mean_normals = np.array([voxel.mean_normals for voxel in self.voxels])
        mean_rgb = np.array([voxel.mean_rgb for voxel in self.voxels])
        index = np.array([voxel.index for voxel in self.voxels])
        data = {
            'x': [index[i][0] for i in range(len(index))],
            'y': [index[i][1] for i in range(len(index))],
            'z': [index[i][2] for i in range(len(index))],
            'num_points': [voxel.num_points for voxel in self.voxels],
            'mean_intensity': [voxel.mean_intensity for voxel in self.voxels],
            'mean_r': [mean_rgb[i][0] for i in range(len(mean_rgb))],
            'mean_g': [mean_rgb[i][1] for i in range(len(mean_rgb))],
            'mean_b': [mean_rgb[i][2] for i in range(len(mean_rgb))],
            'mean_illuminance': [voxel.mean_illuminance for voxel in self.voxels],
            'mean_gps_time': [voxel.mean_gps_time for voxel in self.voxels],
            'distance': [voxel.distance for voxel in self.voxels],
            'mean_normals_x': [mean_normals[i][0] for i in range(len(mean_normals))],
            'mean_normals_y': [mean_normals[i][1] for i in range(len(mean_normals))],
            'mean_normals_z': [mean_normals[i][2] for i in range(len(mean_normals))],
            'label': [voxel.label for voxel in self.voxels]
        }
        return pd.DataFrame(data)

    @property
    def normalized_df(self):
        df = self.df
        df['num_points'] = (df['num_points'] - df['num_points'].min()) / \
            (df['num_points'].max() - df['num_points'].min())
        df['mean_intensity'] = (df['mean_intensity'] - df['mean_intensity'].min()) / (
            df['mean_intensity'].max() - df['mean_intensity'].min())
        df['mean_r'] = (df['mean_r'] - df['mean_r'].min()) / \
            (df['mean_r'].max() - df['mean_r'].min())
        df['mean_g'] = (df['mean_g'] - df['mean_g'].min()) / \
            (df['mean_g'].max() - df['mean_g'].min())
        df['mean_b'] = (df['mean_b'] - df['mean_b'].min()) / \
            (df['mean_b'].max() - df['mean_b'].min())
        df['mean_illuminance'] = (df['mean_illuminance'] - df['mean_illuminance'].min()) / (
            df['mean_illuminance'].max() - df['mean_illuminance'].min())
        df['mean_gps_time'] = (df['mean_gps_time'] - df['mean_gps_time'].min()) / \
            (df['mean_gps_time'].max() - df['mean_gps_time'].min())
        df = df.apply(lambda x: x.fillna(0) if x.dtype == "float64" else x)
        return df

    def show(self, color_field: str = 'intensity') -> None:
        pcd = o3d.geometry.PointCloud()
        voxel_centers = np.array([voxel.calculate_center(
            self.voxel_size) for voxel in self.voxels if voxel.calculate_center(self.voxel_size) is not None])
        pcd.points = o3d.utility.Vector3dVector(voxel_centers)

        if voxel_centers.size > 0:
            field_values = np.array([getattr(voxel, color_field).mean() for voxel in self.voxels if hasattr(
                voxel, color_field) and getattr(voxel, color_field).size > 0])
            if field_values.size > 0:
                field_values = (field_values - field_values.min()) / \
                    (field_values.max() - field_values.min())
                colors = np.zeros((field_values.shape[0], 3))
                colors[:, 0] = field_values  # r
                colors[:, 1] = field_values  # g
                colors[:, 2] = field_values  # b
                pcd.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.get_render_option().background_color = [0.25, 0.25, 0.25]
        vis.add_geometry(pcd)
        vis.run()

    def get_voxels_by_layer(self, layer: int, dimension: int = 2) -> list:
        """Возвращает все воксели с указанного слоя"""
        return [voxel for voxel in self.voxels if voxel.index[dimension] == layer]

    def calculate_distances_to_coordinate(self, coordinate) -> dict:
        """Считает расстояния до координаты дерева для всех вокселей"""
        coordinate = [coordinate[0], coordinate[1], 0]
        distances = {}
        voxels = self.get_voxels_by_layer(0)
        for voxel in voxels:
            center = voxel.calculate_center(self.voxel_size)
            if center is not None:
                distance = np.linalg.norm(
                    np.array(center) - np.array(coordinate))
                distances[voxel.index] = distance
        return distances

    def calculate_distances_to_previous_layer(self, coordinate) -> dict:
        """Считает расстояния до ближайшего вокселя в нижнем слое с label = 0 (ствол)"""
        index = np.array([voxel.index for voxel in self.voxels])
        max_layer = max([index[i][2] for i in range(len(index))])

        for layer in tqdm(range(max_layer+1), desc="Calculating distances to previous layer"):
            current_layer_voxels = self.get_voxels_by_layer(layer)
            if layer == 0:
                tree_center = [coordinate[0], coordinate[1], 0]
                for voxel in current_layer_voxels:
                    center = voxel.calculate_center(self.voxel_size)
                    if center is not None:
                        distance = np.linalg.norm(
                            np.array(center) - np.array(tree_center))
                        self.get_voxel_by_grid_index(
                            voxel.index).distance = distance / self.voxel_size
            else:
                # find labeled voxels in previous layers
                labeled_voxels = []
                i = layer - 1
                while not labeled_voxels:
                    if i < 0:
                        logger.error(f"No labeled voxels found in layer {i}")
                        break
                    previous_layer_voxels = self.get_voxels_by_layer(i)
                    labeled_voxels = [
                        voxel for voxel in previous_layer_voxels if voxel.label == 0]
                    i -= 1

                for voxel in current_layer_voxels:
                    center = voxel.calculate_center(self.voxel_size)
                    if center is not None and labeled_voxels:
                        min_distance = float('inf')
                        for labeled_voxel in labeled_voxels:
                            labeled_center = labeled_voxel.calculate_center(
                                self.voxel_size)
                            if labeled_center is not None:
                                distance = np.linalg.norm(
                                    np.array(center) - np.array(labeled_center))
                                if distance < min_distance:
                                    min_distance = distance
                        self.get_voxel_by_grid_index(
                            voxel.index).distance = min_distance / self.voxel_size
