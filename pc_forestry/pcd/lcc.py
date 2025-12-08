import numpy as np
from scipy.ndimage import label
import scipy
import open3d as o3d
import matplotlib.pyplot as plt
import cc3d  # connected-components
from PIL import Image
import PIL


class LCC:
    def __init__(self, voxel_size=0.1, connectivity=26):
        self.voxel_size = voxel_size
        self.connectivity = connectivity
        self.labels_ = None
        self.voxel_grid_ = None
        self.voxel_labels_map_ = None
        self._voxel_offset_ = None  # смещение для нормализованных индексов

    def fit(self, points):
        self._label_connected_components_with_voxels(points)
        self._assign_labels_from_voxels_to_original_points(points)
        return self

    def _voxelize_points(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, self.voxel_size)  # create voxel grid from point cloud
        return voxel_grid

    def _voxel_grid_to_numpy(self, voxel_grid):
        voxels = voxel_grid.get_voxels()
        voxel_indices = np.array([v.grid_index for v in voxels])
        return voxel_indices

    def _encode_voxels_to_cc3d_format(self, voxel_indices):
        # Нормализуем индексы так, чтобы минимум по каждой оси стал 0
        mins = np.min(voxel_indices, axis=0)
        norm_indices = (voxel_indices - mins).astype(np.int64)

        # Размеры компактного объема
        x_max = int(np.max(norm_indices[:, 0]))
        y_max = int(np.max(norm_indices[:, 1]))
        z_max = int(np.max(norm_indices[:, 2]))

        shape = (x_max + 1, y_max + 1, z_max + 1)

        # Ограничение на размер аллокации (например, до ~500 млн ячеек)
        max_cells = 500_000_000
        total_cells = int(shape[0]) * int(shape[1]) * int(shape[2])
        if total_cells > max_cells:
            raise MemoryError(
                f"Voxel grid too large: shape={shape}, cells={total_cells} > {max_cells}. "
                f"Увеличьте voxel_size или заранее ограничьте область."
            )

        # Используем bool для экономии памяти
        labels = np.zeros(shape, dtype=np.bool_)

        for x, y, z in norm_indices:
            labels[x, y, z] = True

        # Сохраняем смещение для последующей декодировки
        self._voxel_offset_ = mins.astype(np.int64)

        return labels

    def _decode_cc3d_to_voxels_with_class_labels(self, cc3d_input, voxel_indices):
        class_labels = np.zeros(voxel_indices.shape[0], dtype=np.int32)

        if self._voxel_offset_ is None:
            offset = np.array([0, 0, 0], dtype=np.int64)
        else:
            offset = self._voxel_offset_

        for i, xyz in enumerate(voxel_indices):
            xi, yi, zi = (xyz - offset).astype(np.int64)
            class_labels[i] = cc3d_input[xi, yi, zi]
        return class_labels

    def _label_connected_components_with_voxels(self, pcd_points):
        self.voxel_grid_ = self._voxelize_points(pcd_points)
        voxel_indices = self._voxel_grid_to_numpy(self.voxel_grid_)

        cc3d_labels = self._encode_voxels_to_cc3d_format(voxel_indices)
        labels_out, N_components = cc3d.connected_components(cc3d_labels, connectivity=self.connectivity, return_N=True)

        class_labels = self._decode_cc3d_to_voxels_with_class_labels(labels_out, voxel_indices)
        self.voxel_labels_map_ = {tuple(voxel_indices[i]): class_labels[i] for i in range(len(voxel_indices))}

    def _assign_labels_from_voxels_to_original_points(self, original_points):
        voxel_size = self.voxel_grid_.voxel_size
        origin = self.voxel_grid_.origin

        labels = np.zeros(len(original_points), dtype=int)
        for i, point in enumerate(original_points):
            voxel_index = tuple(((point - origin) / voxel_size).astype(int))
            labels[i] = self.voxel_labels_map_.get(voxel_index, 0)

        self.labels_ = labels
