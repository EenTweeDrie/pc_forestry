from .PCD import PCD
import numpy as np
import torch
import numpy as np
from ..predict.models.pointnet2_cls_ssg import get_model
from ..predict.utils import pointcloud_utils as pcu
import pandas as pd
import circle_fit as cf
import statistics
from ..utils.fps import farthest_point_sample
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

import hdbscan
from sklearn.neighbors import LocalOutlierFactor
from loguru import logger
import open3d as o3d
import os


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
    model_path = os.path.join(os.path.dirname(__file__),
                              '..', 'predict', 'checkpoints', model_name, 'models', 'model.t7')
    species_names = ['Trunk', 'Not_Trunk']

    points = torch.Tensor([cluster]).to(device)
    centroids = farthest_point_sample(points, 512)
    pc_sampled = points[0][centroids[0]].cpu().detach().numpy()
    X_test = pcu.tree_normalize(np.array([pc_sampled]))

    int2name = {i: name for i, name in enumerate(species_names)}
    NUM_CLASSES = len(int2name)

    model = get_model(NUM_CLASSES, normal_channel=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
        self.trunk: PCD = None

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
    def read(cls, file_path: str) -> 'PCD':
        instance = cls()
        instance.open(file_path)
        instance.name = file_path.split('/')[-1].split('.')[0]
        return instance

    def shift_to_coordinate(self) -> None:
        """
        Сдвигает облако точек к началу координат.
        Общий вектор сдвига от исходного состояния сохраняется в `self.shift`.
        """
        if self.coordinate is None:
            self.estimate_coordinate()
        shift_this_call = self.coordinate.copy()
        if hasattr(self, 'shift'):
            self.shift += shift_this_call
        else:
            self.shift = shift_this_call
        self.points = self.points - shift_this_call
        self.coordinate = np.array([0, 0, 0])

    def find_trunk_ml(self, model_path: str, config: dict = None) -> None:
        from ..ml.ml_pipeline import MLPipeline

        # Дефолтная конфигурация
        default_config = {
            'voxel_size': 0.3,
            'type_df': 'original',
            'fast_mode': True,
            'proba_threshold': 0.35,
            'feature_params': {}  # Универсальные параметры для любых признаков
        }

        # Обновляем дефолтную конфигурацию пользовательской
        if config:
            default_config.update(config)

        mlp = (
            MLPipeline('tmp_ml_pipeline')
            .set_model_type('catboost')
            .set_datasets_config(default_config)
            .set_model(model_path)
        )

        vg = mlp.fit(self)
        trunk_voxels = [voxel for voxel in vg.voxels if voxel.label == 0]
        self.trunk = vg.get_pcd_by_voxels(trunk_voxels)
        # Вернуть координаты ствола в исходную систему, если в инференсе был сдвиг
        shift_vec = getattr(vg, 'shift_vector', None)
        if shift_vec is not None:
            try:
                # В инференсе применялся points = points - shift; здесь вернем назад (прибавим shift)
                self.trunk.shift_with_vector(-shift_vec)
            except Exception:
                pass

    def find_trunk_cluster(self, height_threshold: float = 3.0, intensity_cut: float = 5000) -> None:
        """ find the trunk cluster """
        device = self.device

        try:
            # Step 1: Filter points within the lower height_threshold meters of the cloud
            z_min = np.min(self.points[:, 2])
            z_max = z_min + height_threshold
            idx_labels = ((self.points[:, 2] >= z_min) & (self.points[:, 2] <= z_max))
            self.trunk_slice = self.clone_like_pcd()
            print(idx_labels.shape)
            self.trunk_slice.index_cut(idx_labels)

            idx_labels = (self.trunk_slice.intensity >= intensity_cut)
            self.trunk_slice.index_cut(idx_labels)
            lower_points = self.trunk_slice.points

            if lower_points.size == 0:
                logger.error(
                    "No points found in the lower 2-3 meters of the cloud.")
                return None

            # Step 2: Apply HDBSCAN clustering
            clusterer = hdbscan.HDBSCAN(min_cluster_size=10,
                                        core_dist_n_jobs=1)
            cluster_labels = clusterer.fit_predict(lower_points)

            if len(set(cluster_labels)) <= 1:
                logger.error("HDBSCAN failed to find distinct clusters.")
                return None

            # Step 3: Predict the cluster
            probabilities = []
            clusters_indices = []
            # debug
            cluser_points = []

            for i in list(set(cluster_labels)):
                if i == -1:
                    continue
                cluster = lower_points[cluster_labels == i]
                if cluster.shape[0] >= 512:
                    probabilities.append(predict_cluster(cluster, device))
                    clusters_indices.append(i)
                    # debug
                    cluser_points.append(cluster.shape[0])

            # Step 4: Sort the clusters by probability
            pdf = pd.DataFrame(
                {'probability': probabilities, 'cluster_index': clusters_indices, 'points': cluser_points})
            pdf = pdf.sort_values(by='probability', ascending=False)
            pdf = pdf.reset_index(drop=True)
            # logger.debug(pdf)

            # Step 5: Get the best cluster
            best_index = None
            for i in range(len(pdf)):
                choosen_index = pdf.iloc[i]['cluster_index']
                if (pdf.iloc[i]['probability'] > 0):
                    choosen_cluster = lower_points[cluster_labels == choosen_index]
                    if max(choosen_cluster[:, 2]) - min(choosen_cluster[:, 2]) > height_threshold/2:
                        if min(choosen_cluster[:, 2]) - min(lower_points[:, 2]) < 0.25:
                            best_index = choosen_index
                            logger.debug(
                                f'Best cluster is {best_index} with {choosen_cluster.shape[0]} points')
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
                "An error occurred while finding the trunk cluster: {}", e)

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
                          low: float = 1.25,
                          high: float = 1.35
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
                xc, yc, r, _ = cf.standardLSQ(points_layer_i[:, :2])
                xc, yc, rh, _ = cf.hyperLSQ(points_layer_i[:, :2])
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
                xc, yc, r, _ = cf.standardLSQ(points_layer_i[:, :2])
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
                xc, yc, rh, _ = cf.hyperLSQ(points_layer_i[:, :2])
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
            xc, yc, r, _ = cf.standardLSQ(points_layer_i[:, :2])
            xc, yc, rh, _ = cf.hyperLSQ(points_layer_i[:, :2])
        except:
            xc, yc, r, _ = 0, 0, 0, 0
            xc, yc, rh, _ = 0, 0, 0, 0
        r13 = r
        rh13 = rh

        # Рассчитываем радиус на высоте 0.9м
        idx_labels_09 = np.where(
            (r_points[:, 2] >= min(self.trunk_slice.points[:, 2]) + 0.85) &
            (r_points[:, 2] < min(self.trunk_slice.points[:, 2]) + 0.95))
        points_layer_09 = r_points[idx_labels_09]

        try:
            _, _, r09, _ = cf.standardLSQ(points_layer_09[:, :2])
            _, _, rh09, _ = cf.hyperLSQ(points_layer_09[:, :2])
        except:
            r09, rh09 = 0, 0

        # Если диаметр на 1.3м больше чем на 0.9м, и расчет на 0.9м успешен,
        # используем диаметр на 0.9м с понижающим коэффициентом.
        if r13 > r09 and r09 > 0:
            logger.debug(f'r13: {r13} > r09: {r09}, using r09 * 0.95: {r09 * 0.95}')
            r13 = r09 * 0.95

        if rh13 > rh09 and rh09 > 0:
            logger.debug(f'rh13: {rh13} > rh09: {rh09}, using rh09 * 0.95: {rh09 * 0.95}')
            rh13 = rh09 * 0.95

        if (r13 > koef*r_median) and (r13 < (koef + 0.1)*r_median):
            r_median = r13
        if (rh13 > koef*rh_median) and (rh13 < (koef + 0.1)*rh_median):
            rh_median = rh13

        x_min, y_min, z_min = self.trunk_slice.points.min(axis=0)
        x_max, y_max, z_max = self.trunk_slice.points.max(axis=0)
        check_r_median = ((x_max - x_min) + (y_max - y_min))/4
        if (r_median > 0.65) or (r_median > 2.1*check_r_median) or (r_median == 0.0):
            logger.debug('Fallback1')
            r_median = check_r_median
        if (rh_median > 0.65) or (rh_median > 2.1*check_r_median) or (rh_median == 0.0):
            logger.debug('Fallback2')
            rh_median = check_r_median

        breast_diameter_tree = float(f"{100 * r_median*2:.2f}")
        breast_diameter_tree_hyper = float(f"{100 * rh_median*2:.2f}")

        self.diameter_LS = breast_diameter_tree
        self.diameter_HLS = breast_diameter_tree_hyper

    def estimate_height(self):
        """ estimate the height of the tree """
        z_min = min(self.points[:, 2])
        z_max = max(self.points[:, 2])
        self.height = z_max - z_min

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
        xc_circle, yc_circle, _, _ = cf.standardLSQ(points_layer_0_3[:, :2])

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
            logger.debug(f'Choose the center of mass')
            coordinate = [xc_mass, yc_mass,
                          (high_height-low_height)/2+low_height+z_min]
        else:
            logger.debug(f'Choose the center of the circle')
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
            self.estimate_coordinate(low_height=1.3, high_height=1.4)

        if self.coordinate is None:
            self.estimate_coordinate()

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
                print(self.custom_coordinate, self.custom_coordinate[2])
                circle_radius = self.diameter_LS / 2 / 100  # convert to meters
                circle_points = []
                for angle in np.linspace(0, 2 * np.pi, 100):
                    x = circle_center[0] + circle_radius * np.cos(angle)
                    y = circle_center[1] + circle_radius * np.sin(angle)
                    circle_points.append([x-0.15, y-0.15, 1.3])
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

    def estimate_multi_trunk_diameters(self,
                                       cut_height: float = 1.2,
                                       slice_height: float = 0.1,
                                       min_cluster_size: int = 50,
                                       min_points_per_trunk: int = 50,
                                       lof_contamination: float = 0.1,
                                       dbscan_eps: float = 0.03,
                                       estimation_method: str = 'lsq',
                                       lsq_fallback_threshold: float = 1.05,
                                       ransac_residual_threshold: float = 0.05):
        """
        Оценивает диаметры для многоствольных деревьев и сохраняет в pandas.DataFrame.

        1. Обрезает ствол на высоте `cut_height`.
        2. Кластеризует верхнюю часть для определения отдельных стволов.
        3. Для каждого ствола берет срез `slice_height`, очищает его и оценивает диаметр.

        Args:
            cut_height (float): Высота для обрезки ствола от его основания.
            slice_height (float): Толщина среза для измерения диаметра.
            min_cluster_size (int): Минимальное количество точек для формирования кластера (ствола).
            min_points_per_trunk (int): Минимальное количество точек в кластере, чтобы считать его стволом.
            lof_contamination (float): Параметр contamination для LocalOutlierFactor при очистке среза.
            dbscan_eps (float): Параметр eps для DBSCAN при очистке среза.
            estimation_method (str): Метод оценки диаметра ('lsq' или 'ransac').
                'lsq': Метод наименьших квадратов с fallback-логикой (быстрее, чувствителен к выбросам).
                'ransac': RANSAC (устойчив к выбросам, может быть медленнее).
            lsq_fallback_threshold (float): Порог для fallback-логики в LSQ (e.g., 1.05 means diameter can't be > 5% larger than spread).
            ransac_residual_threshold (float): Максимальное расстояние от точки до окружности для RANSAC (в метрах).

        Returns:
            pd.DataFrame or None: DataFrame с данными о каждом стволе (координаты центра, диаметр)
                                 или None, если стволы не найдены.
        """
        if self.trunk is None:
            logger.warning("self.trunk is None. Запускаю find_trunk_ml() для поиска ствола.")
            self.find_trunk_ml()
            if self.trunk is None:
                logger.error("find_trunk_ml() не смог найти ствол. Операция прервана.")
                return None

        trunk_points = self.trunk.points

        # Предварительная очистка всего ствола от выбросов
        lof = LocalOutlierFactor(n_neighbors=10, contamination=0.1)
        inliers_mask = lof.fit_predict(trunk_points) > 0
        trunk_points = trunk_points[inliers_mask]

        z_min = np.min(trunk_points[:, 2])

        cut_z = z_min + cut_height
        upper_trunk_points_mask = trunk_points[:, 2] >= cut_z
        upper_trunk_points = trunk_points[upper_trunk_points_mask]

        if upper_trunk_points.shape[0] < min_cluster_size:
            logger.warning(f"Недостаточно точек ({upper_trunk_points.shape[0]}) выше {cut_height}м для кластеризации.")
            self.multi_trunk_diameters_df = None
            return None

        # Кластеризация верхней части ствола по XY координатам для разделения на отдельные стволы
        xy_points = upper_trunk_points[:, :2]
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, core_dist_n_jobs=1)
        labels = clusterer.fit_predict(xy_points)

        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Удаляем шум

        if not unique_labels:
            logger.warning("HDBSCAN не нашел ни одного кластера (ствола).")
            self.multi_trunk_diameters_df = None
            return None

        logger.info(f"Найдено {len(unique_labels)} потенциальных стволов.")

        trunks_data = []
        for label in sorted(list(unique_labels)):
            cluster_mask = (labels == label)
            cluster_points = upper_trunk_points[cluster_mask]

            if cluster_points.shape[0] < min_points_per_trunk:
                logger.debug(
                    f"Ствол {label}: пропущен, т.к. содержит слишком мало точек ({cluster_points.shape[0]}), требуется минимум {min_points_per_trunk}.")
                continue

            # Берем тонкий срез у основания каждого кластера для измерения диаметра
            slice_mask = (cluster_points[:, 2] >= cut_z) & (cluster_points[:, 2] < cut_z + slice_height)
            slice_points = cluster_points[slice_mask]

            # Применяем фильтры для очистки среза
            if slice_points.shape[0] > 20:  # Применяем очистку, если достаточно точек
                lof = LocalOutlierFactor(n_neighbors=10, contamination=lof_contamination)
                inliers_mask = lof.fit_predict(slice_points) > 0
                slice_points = slice_points[inliers_mask]

                if slice_points.shape[0] > 10:
                    db = DBSCAN(eps=dbscan_eps, min_samples=5).fit(slice_points[:, :2])
                    db_labels = db.labels_
                    unique_db_labels, counts = np.unique(db_labels[db_labels != -1], return_counts=True)
                    if counts.size > 0:
                        largest_cluster_label = unique_db_labels[np.argmax(counts)]
                        slice_points = slice_points[db_labels == largest_cluster_label]

            if slice_points.shape[0] < 10:  # Нужно достаточно точек для надежной оценки
                logger.warning(f"Ствол {label}: Недостаточно точек в срезе ({slice_points.shape[0]}) после очистки. Пропускаем.")
                continue

            try:
                slice_points_xy = slice_points[:, :2]
                xc, yc, diameter_m = None, None, None
                current_estimation_method = estimation_method

                if current_estimation_method == 'ransac':
                    try:
                        from skimage.measure import CircleModel, ransac
                        if slice_points_xy.shape[0] < 3:
                            raise ValueError("Недостаточно точек для RANSAC (минимум 3).")

                        model_robust, _ = ransac(slice_points_xy, CircleModel, min_samples=3,
                                                 residual_threshold=ransac_residual_threshold,
                                                 max_trials=100)
                        if model_robust is None:
                            raise RuntimeError("RANSAC не смог найти модель окружности.")

                        xc, yc, r = model_robust.params
                        diameter_m = r * 2
                        logger.debug(f"Ствол {label}: Диаметр RANSAC = {diameter_m*100:.2f} см.")
                    except Exception as e:
                        logger.warning(f"Ствол {label}: RANSAC не удался ({e}). Переключаюсь на LSQ.")
                        current_estimation_method = 'lsq'  # Fallback to LSQ

                if current_estimation_method == 'lsq':
                    if slice_points_xy.shape[0] < 4:
                        logger.warning(f"Ствол {label}: Недостаточно точек ({slice_points_xy.shape[0]}) для LSQ. Пропускаем.")
                        continue

                    xc_l, yc_l, r_l, _ = cf.standardLSQ(slice_points_xy)
                    diameter_lsq = r_l * 2

                    spread_x = np.ptp(slice_points_xy[:, 0])
                    spread_y = np.ptp(slice_points_xy[:, 1])
                    max_spread = max(spread_x, spread_y)

                    if diameter_lsq > lsq_fallback_threshold * max_spread:
                        logger.debug(
                            f"Ствол {label}: Диаметр LSQ ({diameter_lsq*100:.2f} см) > {lsq_fallback_threshold} * разброса ({max_spread*100:.2f} см). "
                            f"Используется fallback-диаметр, равный разбросу."
                        )
                        diameter_m = max_spread
                    else:
                        diameter_m = diameter_lsq
                    xc, yc = xc_l, yc_l

                if diameter_m is not None:
                    diameter_cm = float(f"{diameter_m * 100:.2f}")
                    trunks_data.append({
                        'xc': xc,
                        'yc': yc,
                        'z': cut_z,
                        'diameter_cm': diameter_cm
                    })
                    logger.debug(f"Ствол {label}: диаметр = {diameter_cm:.2f} см, центр = ({xc:.2f}, {yc:.2f}, {cut_z:.2f})")
                else:
                    logger.warning(f"Ствол {label}: Не удалось оценить диаметр методом '{estimation_method}'. Пропускаем.")

            except Exception as e:
                logger.error(f"Ствол {label}: Не удалось аппроксимировать окружность: {e}. Пропускаем.")

        if trunks_data:
            self.multi_trunk_diameters_df = pd.DataFrame(trunks_data)
        else:
            self.multi_trunk_diameters_df = None

        return self.multi_trunk_diameters_df
