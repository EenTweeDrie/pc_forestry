import os
from tqdm import tqdm
import math
import statistics
import circle_fit as cf
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
from ..path_manager import PathManager
from ..pcd.PCD import PCD
from .mesh_adapter import MeshAdapter
from .utils import shp_create
from .VOR_TES import VOR_TES
import hdbscan
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN


class CoordinatesPipeline:
    def __init__(self, base_path: str, file_name: str) -> None:
        self.params: Dict[str, Any] = {}
        self.base_path = base_path
        self.path_manager = PathManager().set_base_dir(base_path)
        self.file_name = file_name
        self.mesh_adapter: Optional[MeshAdapter] = None

    def set_params(self, params: Dict[str, Any]) -> "CoordinatesPipeline":
        self.params = dict(params)
        return self

    def update_params(self, params: Dict[str, Any]) -> "CoordinatesPipeline":
        self.params.update(params)
        return self

    def set_mesh(self, mesh_name: str = 'Mesh.stl') -> "CoordinatesPipeline":
        """
        Устанавливает mesh адаптер для адаптивного определения высоты срезов.

        Args:
            mesh_name: Имя файла mesh'а (по умолчанию 'mesh.stl')
        """
        mesh_path = self.path_manager.get_mesh_file_path(mesh_name)
        try:
            self.mesh_adapter = MeshAdapter(mesh_path)
            print(f"Загружен mesh из файла: {mesh_path}")

            stats = self.mesh_adapter.get_mesh_statistics()
            if stats:
                print(f"Статистика mesh'а:")
                print(f"  - Вершин: {stats['num_vertices']}")
                print(f"  - Треугольников: {stats['num_triangles']}")
                print(f"  - Высота: {stats['z_min']:.2f} - {stats['z_max']:.2f} м")
                print(f"  - Размеры: {stats['x_extent']:.2f} x {stats['y_extent']:.2f} м")

        except Exception as e:
            print(f"Предупреждение: Не удалось загрузить mesh файл {mesh_path}: {e}")
            print("Будет использоваться фиксированная высота среза")
            self.mesh_adapter = None

        return self

    def cut_mesh_data(self, force: bool = True) -> "CoordinatesPipeline":
        if not force and os.path.exists(self.path_manager.get_cut_area_file_path(self.file_name)):
            return self

        pc_area = PCD.read(self.path_manager.get_area_file_path(self.file_name))
        shift_vector = self.params.get('shift_vector', pc_area.calculate_auto_shift_vector())
        pc_area.shift_with_vector(shift_vector=[shift_vector[0], shift_vector[1], 0])

        if self.mesh_adapter is not None:
            height_from = self.params.get('mesh_height_from')
            height_to = self.params.get('mesh_height_to')
            grid_resolution = self.params.get('mesh_grid_resolution')

            height_mask = self.mesh_adapter.create_relative_height_slice(
                pc_area.points,
                height_from=height_from,
                height_to=height_to,
                grid_resolution=grid_resolution
            )
            idx_labels = np.where(height_mask)[0]
            pc_area.index_cut(idx_labels)

        else:
            raise Exception("Mesh adapter not found")

        pc_area.shift_with_vector(shift_vector=[-shift_vector[0], -shift_vector[1], 0])
        pc_area.save(self.path_manager.get_cut_area_file_path(self.file_name))
        return self

    def cut_slice_data(self) -> "CoordinatesPipeline":
        pc_area = PCD.read(self.path_manager.get_area_file_path(self.file_name))
        shift_vector = self.params.get('shift_vector', pc_area.calculate_auto_shift_vector())
        pc_area.shift_with_vector(shift_vector=shift_vector)

        idx_labels = np.where((pc_area.points[:, 2] > self.params['low_height']) & (pc_area.points[:, 2] <= self.params['high_height']))
        pc_area.index_cut(idx_labels)

        pc_area.save(self.path_manager.get_cut_area_file_path(self.file_name))
        return self

    def make_cells(self, force: bool = True) -> "CoordinatesPipeline":
        if not force and os.path.exists(self.path_manager.get_cells_data_dir(self.params['intensity_cut'])):
            return self

        pc_area = PCD.read(self.path_manager.get_cut_area_file_path(self.file_name))
        idx_labels = np.where(pc_area.intensity >= self.params['intensity_cut'])
        pc_area.index_cut(idx_labels)

        shp_poly = shp_create(pc_area.points)

        vortes = VOR_TES(pc=pc_area, algo=self.params['algo'], n_clusters=self.params['n_clusters'])
        vortes.select_borders(self.path_manager.get_cells_borders_dir(self.params['intensity_cut']), shp_poly, verbose=False)
        vortes.select_clusters(
            path_folder_from=self.path_manager.get_cells_borders_dir(self.params['intensity_cut']),
            path_folder_to=self.path_manager.get_cells_data_dir(self.params['intensity_cut'])
        )
        return self

    def make_stumps(self, force: bool = True) -> "CoordinatesPipeline":
        if not force and os.path.exists(self.path_manager.get_stumps_dir(self.params['intensity_cut'])):
            return self

        path_file_cells = self.path_manager.get_cells_data_dir(self.params['intensity_cut'])
        file_paths = self.path_manager.get_file_paths(path_file_cells)
        tfni = 0
        TN = []
        TCX = []
        TCY = []
        TD = []

        for filename in tqdm(file_paths):
            if filename.endswith('.pcd'):
                pc_cells = PCD.read(filename)

                P = pd.DataFrame(pc_cells.points, columns=['X', 'Y', 'Z'])
                X = np.asarray(P)
                clustering = hdbscan.HDBSCAN(min_samples=50, gen_min_span_tree=True).fit(X)
                labels_stumps = clustering.labels_

                for i in tqdm(np.unique(labels_stumps)):
                    if i > -1:
                        pc_stump = pc_cells.clone()
                        idx_label = np.where(labels_stumps == i)
                        pc_stump.index_cut(idx_label)

                        height = pc_stump.points.max(axis=0)[2]-pc_stump.points.min(axis=0)[2]
                        if height >= self.params['height_limit_1']:

                            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
                            inliers = lof.fit_predict(pc_stump.points) > 0
                            pc_stump.index_cut(inliers)

                            P = pd.DataFrame(pc_stump.points[:, 0:2], columns=['X', 'Y'])
                            X = np.asarray(P)
                            if pc_stump.points.shape[0] < 85000:
                                clustering = DBSCAN(eps=self.params['eps_XY'], min_samples=50).fit(X)  # 0.35 perm = 3.5
                                labels_XY = clustering.labels_
                            else:
                                labels_XY = np.zeros(pc_stump.points.shape[0])

                            for j in np.unique(labels_XY):
                                if j > -1:
                                    pc_stump_clear = pc_stump.clone()
                                    idx_label = np.where(labels_XY == j)
                                    pc_stump_clear.index_cut(idx_label)

                                    height = pc_stump_clear.points.max(axis=0)[2]-pc_stump_clear.points.min(axis=0)[2]
                                    if height >= self.params['height_limit_2']:

                                        P = pd.DataFrame(pc_stump_clear.points[:, 2], columns=['Z'])
                                        X = np.asarray(P)
                                        if pc_stump_clear.points.shape[0] < 50000:
                                            clustering = DBSCAN(eps=self.params['eps_Z'], min_samples=50).fit(X)  # 0.35 perm = 3.5
                                            labels_Z = clustering.labels_
                                        else:
                                            labels_Z = np.zeros(pc_stump_clear.points.shape[0])

                                        max_shape = 0
                                        i_max_shape = -1
                                        for k in np.unique(labels_Z):
                                            if k >= -1:
                                                pc_stump_verifiable = pc_stump_clear.clone()
                                                idx_label = np.where(labels_Z == k)
                                                pc_stump_verifiable.index_cut(idx_label)
                                                if pc_stump_verifiable.points.shape[0] > max_shape:
                                                    max_shape = pc_stump_verifiable.points.shape[0]
                                                    i_max_shape = k

                                        if i_max_shape != -1:
                                            pc_stump_suitable = pc_stump_clear.clone()
                                            idx_label = np.where(labels_Z == i_max_shape)
                                            pc_stump_suitable.index_cut(idx_label)

                                            r_list = []
                                            xy_list = []
                                            save_center = [0, 0, 0]

                                            x_min, y_min, z_min = pc_stump_suitable.points.min(axis=0)
                                            x_max, y_max, z_max = pc_stump_suitable.points.max(axis=0)
                                            if z_max-z_min > 1:

                                                num_layers = 4
                                                layer = (z_max-z_min)/num_layers

                                                for l in range(num_layers):
                                                    pc_stump_suitable_layer = pc_stump_suitable.clone()
                                                    idx_layer = np.where(
                                                        (pc_stump_suitable_layer.points[:, 2] >= l*layer+z_min) & (pc_stump_suitable_layer.points[:, 2] < (l+1)*layer+z_min))
                                                    pc_stump_suitable_layer.index_cut(idx_layer)

                                                    try:
                                                        xc, yc, r, _ = cf.hyper_fit(pc_stump_suitable_layer.points)
                                                    except:
                                                        xc, yc, r, _ = 0, 0, 0, 0
                                                    r_list.append(r)
                                                    xy_list.append([xc, yc])

                                                xy_list = np.asarray(xy_list)

                                                r_median = statistics.median(r_list)
                                                x_median = statistics.median(xy_list[:, 0])
                                                y_median = statistics.median(xy_list[:, 1])
                                                check_x = np.median(pc_stump_suitable.points[:, 0])
                                                check_y = np.median(pc_stump_suitable.points[:, 1])

                                                x_min, y_min, z_min = pc_stump_suitable.points.min(axis=0)
                                                x_max, y_max, z_max = pc_stump_suitable.points.max(axis=0)
                                                check_r_median = ((x_max - x_min) + (y_max - y_min))/4
                                                if (r_median > 0.65) or (r_median > 2.1*check_r_median) or (r_median == 0.0):
                                                    r_median = check_r_median

                                                dist = math.sqrt((xy_list[0][0] - check_x)**2 + (xy_list[0][1] - check_y)**2)
                                                if dist > 0.25:
                                                    dist = math.sqrt((x_median - check_x)**2 + (y_median - check_y)**2)
                                                    if dist > 0.25:
                                                        save_center = [check_x, check_y, 1]
                                                    else:
                                                        save_center = [x_median, y_median, 1]
                                                else:
                                                    save_center = [xy_list[0][0], xy_list[0][1], 1]

                                                tfni += 1
                                                filename_stumps_out = 'int' + \
                                                    str(self.params['intensity_cut']) + '_' + str(tfni).rjust(4, '0') + '.pcd'
                                                fname_stumps_out = os.path.join(self.path_manager.get_stumps_dir(
                                                    self.params['intensity_cut']), filename_stumps_out)
                                                pc_stump_suitable.save(fname_stumps_out)

                                                TN.append(filename_stumps_out)
                                                TCX.append(save_center[0])
                                                TCY.append(save_center[1])
                                                TD.append(r_median*2)
        TN = np.asarray(TN)
        TCX = np.asarray(TCX)
        TCY = np.asarray(TCY)
        TD = np.asarray(TD)

        bd = pd.DataFrame({"Name_stump"+'_int' + str(self.params['intensity_cut']): TN,
                          "X": TCX, "Y": TCY, "Diameter"+'_int' + str(self.params['intensity_cut']): TD})
        bd.to_csv(self.path_manager.get_stumps_dir(self.params['intensity_cut']) +
                  f'/stumps_{self.params["intensity_cut"]}.csv', index=False, sep=';')

        file = open(os.path.join(self.path_manager.get_stumps_dir(self.params['intensity_cut']), "coordinates_paths.txt"), "a")
        file.write("\n"+self.path_manager.get_stumps_dir(self.params['intensity_cut']) + f'/stumps_{self.params["intensity_cut"]}.csv')
        file.close()

        return self
