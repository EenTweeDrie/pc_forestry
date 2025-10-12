import numpy as np
from scipy.ndimage import label
import scipy
import open3d as o3d
from pc_forestry.pcd.PCD import PCD
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
        x_max = np.max(voxel_indices[:, 0])
        y_max = np.max(voxel_indices[:, 1])
        z_max = np.max(voxel_indices[:, 2])

        labels = np.zeros((x_max+1, y_max+1, z_max+1))

        for x, y, z in voxel_indices:
            labels[x, y, z] = 1

        return labels

    def _decode_cc3d_to_voxels_with_class_labels(self, cc3d_input, voxel_indices):
        class_labels = np.zeros(voxel_indices.shape[0], dtype=np.int32)

        for i, xyz in enumerate(voxel_indices):
            class_labels[i] = cc3d_input[xyz[0], xyz[1], xyz[2]]
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
