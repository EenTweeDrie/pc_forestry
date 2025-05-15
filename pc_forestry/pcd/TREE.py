from .PCD import PCD
import numpy as np
import torch
import numpy as np
from predict.models.pointnet2_cls_ssg import get_model
import predict.utils.pointcloud_utils as pcu
import pandas as pd
import circle_fit as cf
import statistics
from .PCD_UTILS import PCD_UTILS
from ..utils.fps import farthest_point_sample
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

import hdbscan
from sklearn.neighbors import LocalOutlierFactor
import logging
import open3d as o3d

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def angle_between_vectors(vector1, vector2):
    v1 = [vector1[1][0] - vector1[0][0],
          vector1[1][1] - vector1[0][1],
          vector1[1][2] - vector1[0][2]]

    v2 = [vector2[1][0] - vector2[0][0],
          vector2[1][1] - vector2[0][1],
          vector2[1][2] - vector2[0][2]]

    # Вычисляем скалярное произведение
    dot_product = np.dot(v1, v2)

    # Вычисляем длины векторов
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # Вычисляем косинус угла
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)

    # Вычисляем угол в радианах, а затем переводим в градусы
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def predict_cluster(cluster, device):
    """ predict the cluster """
    model_name = 'int0000_7000-512-rlish-s4762'
    model_path = 'predictmdl/checkpoints/' + model_name + '/models/model.t7'
    species_names = ['Trunk', 'Not_Trunk']

    points = torch.Tensor([cluster]).to(device)
    centroids = farthest_point_sample(points, 512)
    pc_sampled = points[0][centroids[0]].cpu().detach().numpy()
    X_test = pcu.tree_normalize(np.array([pc_sampled]))

    int2name = {i: name for i, name in enumerate(species_names)}
    NUM_CLASSES = len(int2name)

    model = get_model(NUM_CLASSES, normal_channel=False).to(device)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    data = torch.tensor(X_test, device=device)
    data = data.permute(0, 2, 1)

    with torch.no_grad():
        logits, _ = model(data)
        # Переводим log_softmax в обычные вероятности
        probabilities = torch.exp(logits)

    prob_class_0 = probabilities[:, 0]
    return prob_class_0.item()


class TREE(PCD):
    diameter_LS = None
    diameter_HLS = None

    def __init__(self,
                 points: np.ndarray = None,
                 intensity: np.ndarray = None,
                 gps_time: np.ndarray = None,
                 original_cloud_index: np.ndarray = None,
                 rgb: np.ndarray = None,
                 illuminance: np.ndarray = None,
                 name: str = None,
                 coordinate: np.ndarray = None,
                 ):
        super().__init__(points=points,
                         intensity=intensity,
                         gps_time=gps_time,
                         original_cloud_index=original_cloud_index,
                         rgb=rgb,
                         illuminance=illuminance)
        self.name = name
        self.coordinate = coordinate
        self.trunk_slice: PCD = None
        self.custom_coordinate = None

    @classmethod
    def init_from_pcd(cls, pc: PCD) -> None:
        """ initialize tree from PCD object """
        instance = cls(
            points=pc.points,
            intensity=pc.intensity,
            gps_time=pc.gps_time,
            original_cloud_index=pc.original_cloud_index,
            rgb=pc.rgb,
            illuminance=pc.illuminance,
        )
        return instance

    @classmethod
    def read(cls, file_path: str, verbose: bool = False) -> 'PCD':
        instance = cls()
        instance.open(file_path, verbose=verbose)
        instance.name = file_path.split('/')[-1].split('.')[0]
        return instance

    def shift_to_coordinate(self) -> None:
        """ shift points to coordinate """
        self.points = self.points - self.coordinate

    def find_trunk_cluster(self, height_threshold: float = 3.0, intensity_cut: float = 5000) -> None:
        """ find the trunk cluster """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            # Step 1: Filter points within the lower height_threshold meters of the cloud
            z_min = np.min(self.points[:, 2])
            z_max = z_min + height_threshold
            idx_labels = np.where(
                (self.points[:, 2] >= z_min) & (self.points[:, 2] <= z_max))
            self.trunk_slice = self.clone_like_pcd()
            self.trunk_slice.index_cut(idx_labels)

            idx_labels = np.where(self.trunk_slice.intensity >= intensity_cut)
            self.trunk_slice.index_cut(idx_labels)
            lower_points = self.trunk_slice.points

            if lower_points.size == 0:
                logger.error(
                    "No points found in the lower 2-3 meters of the cloud.")
                return None

            # Step 2: Apply HDBSCAN clustering
            clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
            cluster_labels = clusterer.fit_predict(lower_points)

            if len(set(cluster_labels)) <= 1:
                logger.error("HDBSCAN failed to find distinct clusters.")
                return None

            # Step 3: Predict the cluster
            probabilities = []
            clusters_indices = []

            for i in list(set(cluster_labels)):
                if i == -1:
                    continue
                cluster = lower_points[cluster_labels == i]
                if cluster.shape[0] > 100:
                    probabilities.append(predict_cluster(cluster, device))
                    clusters_indices.append(i)

            # Step 4: Sort the clusters by probability
            pdf = pd.DataFrame(
                {'probability': probabilities, 'cluster_index': clusters_indices})
            pdf = pdf.sort_values(by='probability', ascending=False)
            pdf = pdf.reset_index(drop=True)

            # Step 5: Get the best cluster
            best_index = None
            for i in range(len(pdf)):
                choosen_index = pdf.iloc[i]['cluster_index']
                if (pdf.iloc[i]['probability'] > 0):
                    choosen_cluster = lower_points[cluster_labels ==
                                                   choosen_index]
                    if max(choosen_cluster[:, 2]) - min(choosen_cluster[:, 2]) > height_threshold/2:
                        if min(choosen_cluster[:, 2]) - min(lower_points[:, 2]) < 0.25:
                            best_index = choosen_index
                            break

            # Step 6: Cut the trunk slice
            if best_index is not None:
                idx_labels = np.where(cluster_labels == best_index)
                self.trunk_slice.index_cut(idx_labels)
            else:
                logger.warning("Probability is too low")

            # Step 7: Apply Statistical Outlier Removal
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            inliers = lof.fit_predict(self.trunk_slice.points) > 0
            self.trunk_slice.index_cut(inliers)

            if self.trunk_slice.points.size == 0:
                logger.error("All points were removed as outliers.")
                return None

        except Exception as e:
            logger.exception(
                "An error occurred while finding the trunk cluster: %s", e)

    def clone_like_pcd(self):
        return PCD(points=self.points,
                   intensity=self.intensity,
                   gps_time=self.gps_time,
                   original_cloud_index=self.original_cloud_index,
                   rgb=self.rgb,
                   illuminance=self.illuminance)

    def estimate_diameter(self,
                          num_layers: int = 10,
                          koef: float = 1.05,
                          low: float = 1.3,
                          high: float = 1.4
                          ) -> None:
        """ estimate the diameter of the tree """
        if self.trunk_slice is None:
            self.find_trunk_cluster()
        r_points = self.trunk_slice.points

        x_min, y_min, z_min = r_points.min(axis=0)
        x_max, y_max, z_max = r_points.max(axis=0)

        layer = (z_max-z_min)/num_layers
        rh_list = []
        r_list = []

        for i in range(num_layers):
            idx_labels = np.where(
                (r_points[:, 2] >= i*layer+z_min) & (r_points[:, 2] < (i+1)*layer+z_min))
            points_layer_i = r_points[idx_labels]

            try:
                xc, yc, r, _ = cf.standardLSQ(points_layer_i)
                xc, yc, rh, _ = cf.hyperLSQ(points_layer_i)
            except:
                xc, yc, r, _ = 0, 0, 0, 0
                xc, yc, rh, _ = 0, 0, 0, 0
            rh_list.append(rh)
            r_list.append(r)

        if len(r_list) == 0:
            for i in range(num_layers):
                idx_labels = np.where(
                    (self.trunk_slice.points[:, 2] >= i*layer+z_min) &
                    (self.trunk_slice.points[:, 2] < (i+1)*layer+z_min))
                points_layer_i = self.trunk_slice.points[idx_labels]
            try:
                xc, yc, r, _ = cf.standardLSQ(points_layer_i)
            except:
                xc, yc, r, _ = 0, 0, 0, 0
            r_list.append(r)

        if len(rh_list) == 0:
            for i in range(num_layers):
                idx_labels = np.where(
                    (self.trunk_slice.points[:, 2] >= i*layer+z_min) &
                    (self.trunk_slice.points[:, 2] < (i+1)*layer+z_min))
                points_layer_i = self.trunk_slice.points[idx_labels]
            try:
                xc, yc, rh, _ = cf.hyperLSQ(points_layer_i)
            except:
                xc, yc, rh, _ = 0, 0, 0, 0
            rh_list.append(rh)

        r_median = statistics.median(r_list)
        rh_median = statistics.median(rh_list)

        r_median = min(r_list)
        rh_median = min(rh_list)

        idx_labels = np.where(
            (r_points[:, 2] >= min(self.trunk_slice.points[:, 2])+low) &
            (r_points[:, 2] < min(self.trunk_slice.points[:, 2])+high))
        points_layer_i = r_points[idx_labels]

        try:
            xc, yc, r, _ = cf.standardLSQ(points_layer_i)
            xc, yc, rh, _ = cf.hyperLSQ(points_layer_i)
        except:
            xc, yc, r, _ = 0, 0, 0, 0
            xc, yc, rh, _ = 0, 0, 0, 0
        r13 = r
        rh13 = rh

        if (r13 > koef*r_median) and (r13 < (koef + 0.1)*r_median):
            r_median = r13
        if (rh13 > koef*rh_median) and (rh13 < (koef + 0.1)*rh_median):
            rh_median = rh13

        x_min, y_min, z_min = self.trunk_slice.points.min(axis=0)
        x_max, y_max, z_max = self.trunk_slice.points.max(axis=0)
        check_r_median = ((x_max - x_min) + (y_max - y_min))/4
        if (r_median > 0.65) or (r_median > 2.1*check_r_median) or (r_median == 0.0):
            logger.info(f'Fallback1')
            r_median = check_r_median
        if (rh_median > 0.65) or (rh_median > 2.1*check_r_median) or (rh_median == 0.0):
            logger.info(f'Fallback2')
            rh_median = check_r_median

        breast_diameter_tree = 100 * float(PCD_UTILS.toFixed(r_median*2, 4))
        breast_diameter_tree_hyper = 100 * \
            float(PCD_UTILS.toFixed(rh_median*2, 4))
        breast_diameter_tree = float(f"{breast_diameter_tree:.2f}")
        breast_diameter_tree_hyper = float(f"{breast_diameter_tree_hyper:.2f}")

        self.diameter_LS = breast_diameter_tree
        self.diameter_HLS = breast_diameter_tree_hyper

    def estimate_coordinate(self, error_threshold: float = 0.2, low_height: float = 0, high_height: float = 0.6):
        """ estimate the coordinate of the tree """
        # If there is no trunk slice, find it
        if self.trunk_slice is None:
            self.find_trunk_cluster()

        z_min = min(self.trunk_slice.points[:, 2])

        # Find the center of the circle at a height of 0.3 meters
        idx_labels_0_3 = np.where(
            (self.trunk_slice.points[:, 2] >= z_min + low_height) &
            (self.trunk_slice.points[:, 2] < high_height + z_min + low_height)
        )
        points_layer_0_3 = self.trunk_slice.points[idx_labels_0_3]
        xc_circle, yc_circle, _, _ = cf.standardLSQ(points_layer_0_3)

        # Find the center of mass of points at a height of up to high_height=0.75 meters
        idx_labels_0_75 = np.where(
            self.trunk_slice.points[:, 2] < high_height + z_min)
        points_layer_0_75 = self.trunk_slice.points[idx_labels_0_75]
        xc_mass, yc_mass = np.mean(points_layer_0_75[:, 0]), np.mean(
            points_layer_0_75[:, 1])

        # Select the coordinate depending on the distance between the centers
        distance = np.sqrt((xc_circle - xc_mass) ** 2 +
                           (yc_circle - yc_mass) ** 2)
        if distance > error_threshold:
            # logger.info(f'Choose the center of mass')
            coordinate = [xc_mass, yc_mass,
                          (high_height-low_height)/2+low_height+z_min]
        else:
            # logger.info(f'Choose the center of the circle')
            coordinate = [xc_circle, yc_circle,
                          (high_height-low_height)/2+low_height+z_min]

        if low_height == 0:
            # logger.info(f'Default coordinate: {coordinate}')
            self.coordinate = coordinate
        else:
            # logger.info(f'Custom coordinate: {coordinate}')
            self.custom_coordinate = coordinate

    def get_angle(self):
        """ calculate the angle of the tree """
        if self.coordinate is None:
            self.estimate_coordinate()
        if self.custom_coordinate is None:
            self.estimate_coordinate(low_height=1, high_height=1.6)
        vector1 = [self.coordinate, self.custom_coordinate]
        vector2 = [self.coordinate,
                   [self.coordinate[0], self.coordinate[1], self.custom_coordinate[2]]]
        return angle_between_vectors(vector1, vector2)

    def get_tan_angle(self):
        """ calculate the tangent of the angle of the tree """
        return np.tan(np.radians(self.get_angle()))

    def get_cos_angle(self):
        """ calculate the cosine of the angle of the tree """
        return np.cos(np.radians(self.get_angle()))

    def show_with_parameters(self):
        pcd_to_show = []

        if self.custom_coordinate is None:
            self.estimate_coordinate(low_height=1, high_height=1.6)

        if self.diameter_LS is None:
            self.estimate_diameter()

        if self.trunk_slice:
            z_min = min(self.trunk_slice.points[:, 2])
            trunk_slice_pcd = o3d.geometry.PointCloud()
            trunk_slice_pcd.points = o3d.utility.Vector3dVector(
                self.trunk_slice.points)
            pcd_to_show.append(trunk_slice_pcd)

            if self.diameter_LS and self.coordinate:
                # Display the diameter as a circle at a height of 1.3 meters
                circle_center = self.custom_coordinate
                circle_radius = self.diameter_LS / 2 / 100  # convert to meters
                circle_points = []
                for angle in np.linspace(0, 2 * np.pi, 100):
                    x = circle_center[0] + circle_radius * np.cos(angle)
                    y = circle_center[1] + circle_radius * np.sin(angle)
                    circle_points.append([x, y, circle_center[2]])
                circle_points = np.array(circle_points)
                # Create a point cloud for the circle
                circle_pcd = o3d.geometry.PointCloud()
                circle_pcd.points = o3d.utility.Vector3dVector(circle_points)
                circle_pcd.paint_uniform_color([1, 0, 0])  # red color
                pcd_to_show.append(circle_pcd)

            if self.coordinate:
                # Display the coordinate as a thick point at a height of 0 meters
                coordinate_point = [self.coordinate[0],
                                    self.coordinate[1], z_min]
                coordinate_pcd = o3d.geometry.PointCloud()
                coordinate_pcd.points = o3d.utility.Vector3dVector(
                    [coordinate_point])
                coordinate_pcd.paint_uniform_color([0, 1, 0])  # green color
                pcd_to_show.append(coordinate_pcd)

            o3d.visualization.draw_geometries(pcd_to_show)
        else:
            return ValueError("No trunk slice found")
