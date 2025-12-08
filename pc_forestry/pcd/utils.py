from loguru import logger
import numpy as np
import pandas as pd
import hdbscan
from sklearn.neighbors import LocalOutlierFactor
from ..pcd.PCD import PCD
import torch
import os
from ..predict.utils import pointcloud_utils as pcu
from ..predict.models.pointnet2_cls_ssg import get_model
from ..utils.fps import farthest_point_sample
import circle_fit as cf
import warnings


def predict_cluster(cluster, device):
    """ predict the cluster """
    model_name = 'int0000_7000-512-rlish-s4762'
    model_path = os.path.join(os.path.dirname(__file__),
                              '..', 'predict', 'checkpoints', model_name, 'models', 'model.t7')
    species_names = ['Trunk', 'Not_Trunk']

    cluster_np = np.asarray(cluster, dtype=np.float32)
    points = torch.from_numpy(cluster_np).unsqueeze(0).to(device)
    centroids = farthest_point_sample(points, 512)
    pc_sampled = points[0][centroids[0]].cpu().detach().numpy()
    X_test = pcu.tree_normalize(np.array([pc_sampled]))

    int2name = {i: name for i, name in enumerate(species_names)}
    NUM_CLASSES = len(int2name)

    model = get_model(NUM_CLASSES, normal_channel=False).to(device)
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.eval()
    data = torch.tensor(X_test, device=device)
    data = data.permute(0, 2, 1)

    with torch.no_grad():
        logits, _ = model(data)
        # Переводим log_softmax в обычные вероятности
        probabilities = torch.exp(logits)

    prob_class_0 = probabilities[:, 0]
    return prob_class_0.item()


def get_trunk_slice(pc: PCD, height_threshold: float = 3.0, intensity_cut: float = 5000) -> PCD:
    """ find the trunk cluster """
    device = pc.device

    try:
        # Step 1: Filter points within the lower height_threshold meters of the cloud
        z_min = np.min(pc.points[:, 2])
        z_max = z_min + height_threshold
        idx_labels = ((pc.points[:, 2] >= z_min) & (pc.points[:, 2] <= z_max))
        trunk_slice = pc.clone()
        trunk_slice.index_cut(idx_labels)

        idx_labels = (trunk_slice.intensity >= intensity_cut)
        trunk_slice.index_cut(idx_labels)
        lower_points = trunk_slice.points

        if lower_points.size == 0:
            logger.error(
                "No points found in the lower 2-3 meters of the cloud.")
            return pc

        # Step 2: Apply HDBSCAN clustering
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.*")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=10,
                                        core_dist_n_jobs=1)
            cluster_labels = clusterer.fit_predict(lower_points)

        if len(set(cluster_labels)) <= 1:
            logger.error("HDBSCAN failed to find distinct clusters.")
            return pc

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
                        # logger.debug(
                        #     f'Best cluster is {best_index} with {choosen_cluster.shape[0]} points')
                        break

        # Step 6: Cut the trunk slice
        if best_index is not None:
            idx_labels = np.where(cluster_labels == best_index)
            trunk_slice.index_cut(idx_labels)
        # else:
        #     logger.warning("Probability is too low")

        # Step 7: Apply Statistical Outlier Removal
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.*")
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            inliers = lof.fit_predict(trunk_slice.points) > 0
        trunk_slice.index_cut(inliers)

        if trunk_slice.points.size == 0:
            logger.error("All points were removed as outliers.")
            return pc

        return trunk_slice

    except Exception as e:
        logger.exception(
            "An error occurred while finding the trunk cluster: {}", e)

    return pc


def get_tree_coordinate(trunk_slice: PCD, error_threshold: float = 0.2, low_height: float = 0, high_height: float = 0.6):
    """ estimate the coordinate of the tree """
    # If there is no trunk slice, find it
    z_min = min(trunk_slice.points[:, 2])

    # Find the center of the circle at a height of 0.3 meters
    idx_labels_0_3 = np.where(
        (trunk_slice.points[:, 2] >= z_min + low_height) &
        (trunk_slice.points[:, 2] < high_height + z_min + low_height)
    )
    points_layer_0_3 = trunk_slice.points[idx_labels_0_3]
    xc_circle, yc_circle, _, _ = cf.standardLSQ(points_layer_0_3[:, :2])

    # Find the center of mass of points at a height of up to high_height=0.75 meters
    idx_labels_0_75 = np.where(
        trunk_slice.points[:, 2] < high_height + z_min)
    points_layer_0_75 = trunk_slice.points[idx_labels_0_75]
    xc_mass, yc_mass = np.mean(points_layer_0_75[:, 0]), np.mean(
        points_layer_0_75[:, 1])

    # Select the coordinate depending on the distance between the centers
    distance = np.sqrt((xc_circle - xc_mass) ** 2 +
                       (yc_circle - yc_mass) ** 2)
    if distance > error_threshold:
        # logger.debug(f'Choose the center of mass')
        coordinate = [xc_mass, yc_mass,
                      (high_height-low_height)/2+low_height+z_min]
    else:
        # logger.debug(f'Choose the center of the circle')
        coordinate = [xc_circle, yc_circle,
                      (high_height-low_height)/2+low_height+z_min]

    return coordinate
