from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger

# Forward declaration to avoid circular import
if 'VOXELGRID' not in globals():
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from .VOXEL import VOXELGRID


class FeatureCalculator(ABC):
    """Abstract base class for feature calculators."""

    @abstractmethod
    def calculate(self, voxel_grid: 'VOXELGRID', **kwargs) -> np.ndarray:
        """
        Calculates the feature for a given VOXELGRID.
        Returns a numpy array of feature values, one for each voxel.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the feature."""
        pass


class DistanceToCoordinate(FeatureCalculator):
    @property
    def name(self) -> str:
        return "distance_to_coord"

    def calculate(self, voxel_grid: 'VOXELGRID', **kwargs) -> np.ndarray:
        coordinate = kwargs.get("coordinate")
        if coordinate is None:
            raise ValueError("coordinate is required for DistanceToCoordinate feature.")

        distances = []
        coord_3d = np.array([coordinate[0], coordinate[1], 0])

        for voxel in voxel_grid.voxels:
            center = voxel.calculate_center(voxel_grid.voxel_size)
            if center is not None:
                distance = np.linalg.norm(np.array(center) - coord_3d)
                distances.append(distance)
            else:
                distances.append(np.nan)
        return np.array(distances)


class DistanceToPreviousLayer(FeatureCalculator):
    @property
    def name(self) -> str:
        return "distance"

    def calculate(self, voxel_grid: 'VOXELGRID', **kwargs) -> np.ndarray:
        coordinate = kwargs.get("coordinate")
        if coordinate is None:
            raise ValueError("coordinate is required for DistanceToPreviousLayer feature.")

        index = np.array([voxel.index for voxel in voxel_grid.voxels])
        if len(index) == 0:
            return np.array([])

        max_layer = max([idx[2] for idx in index])

        all_distances = {}

        layers = [[] for _ in range(max_layer + 1)]
        voxel_map = {voxel.index: voxel for voxel in voxel_grid.voxels}
        for voxel in voxel_grid.voxels:
            layers[voxel.index[2]].append(voxel)

        for layer_num in range(max_layer + 1):
            current_layer_voxels = layers[layer_num]

            if layer_num == 0:
                coord_3d = np.array([coordinate[0], coordinate[1], 0])
                for voxel in current_layer_voxels:
                    center = voxel.calculate_center(voxel_grid.voxel_size)
                    if center is not None:
                        distance = np.linalg.norm(np.array(center) - coord_3d)
                        all_distances[voxel.index] = distance
                    else:
                        all_distances[voxel.index] = np.nan
            else:
                labeled_voxels = []
                i = layer_num - 1
                while not labeled_voxels and i >= 0:
                    previous_layer_voxels = layers[i]
                    labeled_voxels = [v for v in previous_layer_voxels if voxel_map[v.index].label == 0]
                    i -= 1

                if not labeled_voxels:
                    logger.warning(f"No labeled voxels found in layers below {layer_num}, using coordinate distance.")

                for voxel in current_layer_voxels:
                    center = voxel.calculate_center(voxel_grid.voxel_size)
                    if center is not None:
                        if labeled_voxels:
                            min_distance = float('inf')
                            for labeled_voxel in labeled_voxels:
                                labeled_center = labeled_voxel.calculate_center(voxel_grid.voxel_size)
                                if labeled_center is not None:
                                    distance = np.linalg.norm(np.array(center) - np.array(labeled_center))
                                    if distance < min_distance:
                                        min_distance = distance
                            all_distances[voxel.index] = min_distance
                        else:  # Fallback for layers with no labeled voxels below
                            coord_3d = np.array([coordinate[0], coordinate[1], 0])
                            distance = np.linalg.norm(np.array(center) - coord_3d)
                            all_distances[voxel.index] = distance

                    else:
                        all_distances[voxel.index] = np.nan

        final_distances = [all_distances[v.index] for v in voxel_grid.voxels]
        return np.array(final_distances)


class FeatureManager:
    def __init__(self, voxel_grid: 'VOXELGRID'):
        self.voxel_grid = voxel_grid
        self.feature_calculators = []
        self.df = None

    def add_feature(self, feature: FeatureCalculator):
        self.feature_calculators.append(feature)

    def extract_features(self, **kwargs) -> pd.DataFrame:
        base_df = self.voxel_grid.df

        feature_dfs = []
        for feature in tqdm(self.feature_calculators, desc="Extracting features"):
            values = feature.calculate(self.voxel_grid, **kwargs)
            feature_df = pd.DataFrame({feature.name: values})
            feature_dfs.append(feature_df)

        self.df = pd.concat([base_df] + feature_dfs, axis=1)
        return self.df

    def get_normalized_df(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("Run extract_features first to generate the DataFrame.")

        df = self.df.copy()

        for col in ['num_points', 'mean_intensity', 'mean_r', 'mean_g', 'mean_b', 'mean_illuminance', 'mean_gps_time']:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val - min_val > 0:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[col] = 0  # Or 0.5, depending on desired behavior for single-value columns

        for col in ['distance', 'distance_to_coord']:
            if col in df.columns:
                df[col] = (df[col] / self.voxel_grid.voxel_size)

        df = df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'fc' else x)
        return df
