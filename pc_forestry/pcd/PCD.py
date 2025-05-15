import torch
import numpy as np
import pprint
from time import time
from .PCD_UTILS import PCD_UTILS
import open3d as o3d
import laspy
import pyvista
from pypcd import pypcd
import h5py
import pandas as pd
import copy


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
                las.red = self.rgb[:, 0] * 256
                las.green = self.rgb[:, 1] * 256
                las.blue = self.rgb[:, 2] * 256
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
                las.red = self.rgb[:, 0] * 256
                las.green = self.rgb[:, 1] * 256
                las.blue = self.rgb[:, 2] * 256
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
        centroids = PCD_UTILS.farthest_point_sample(points_torch, num_sample)
        pt_sampled = points_torch[0][centroids[0]]
        centroids = centroids.cpu().data.numpy()
        self.intensity = self.intensity[centroids[0]]
        self.rgb = self.rgb[centroids[0]]
        self.original_cloud_index = self.original_cloud_index[centroids[0]]
        self.gps_time = self.gps_time[centroids[0]]
        self.illuminance = self.illuminance[centroids[0]]
        self.points = pt_sampled.cpu().detach().numpy()

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
            print("Estimating normals")
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

    def concatenate(self, data: np.ndarray) -> None:
        ''' DEPRECATED, use self.append() instead'''
        dt = np.c_[self.points, self.intensity,
                   self.rgb, self.original_cloud_index, self.gps_time]
        dt = np.concatenate((dt, data), axis=0)
        dt = np.array(dt, dtype=np.float32)
        self.points = dt[:, 0:3]
        self.intensity = dt[:, 3]
        self.rgb = dt[:, 4:7]
        self.original_cloud_index = dt[:, 7]
        self.gps_time = dt[:, 8]
        self.illuminance = dt[:, 9]

    def append(self, other: 'PCD') -> None:
        """ append PCD object """
        if not isinstance(other, PCD):
            raise TypeError("Argument must be an instance of PCD")
        self.points = np.concatenate((self.points, other.points), axis=0)
        self.intensity = np.concatenate(
            (self.intensity, other.intensity), axis=0)
        self.rgb = np.concatenate((self.rgb, other.rgb), axis=0)
        self.original_cloud_index = np.concatenate(
            (self.original_cloud_index, other.index), axis=0)
        self.gps_time = np.concatenate((self.gps_time, other.gps_time), axis=0)
        self.illuminance = np.concatenate(
            (self.illuminance, other.illuminance), axis=0)

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

    def visual_gif(self, path_gif: str, zoom: float = 0.4, point_size: float = 4.0) -> None:
        """ visualize PCD object as gif """
        cloud = pyvista.PointSet(self.points)
        scalars = np.linalg.norm(cloud.points - cloud.center, axis=1)
        pl = pyvista.Plotter(off_screen=True)
        pl.add_mesh(
            cloud,
            color='#fff7c2',
            scalars=scalars,
            opacity=1,
            point_size=point_size,
            show_scalar_bar=False,
        )
        pl.background_color = 'k'
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
