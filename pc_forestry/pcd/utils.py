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
from typing import List, Literal, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .VOXELGRIDFEATURES import VOXELGRIDFEATURES


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


def visual_gif_voxel_sizes(
    pc: PCD,
    path_gif: str,
    voxel_sizes: list[float],
    *,
    color_field: str = "rgb",
    component: int | None = None,
    zoom: float = 0.4,
    point_size: float = 6.0,
    orbit_turns: float = 2.0,
    rotate: bool = True,
    background_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    show_text: bool = True,
    backend: str = "pyvista",
    window_size: tuple[int, int] = (400, 400),
    fps: int | None = None,
    frame_duration: float | None = None,
    loop: int = 0,
    shift_to_center: bool = True,
    viewup: tuple[float, float, float] = (0.0, 0.0, 1.0),
    roll_deg: float = 0.0,
    elevation_deg: float = 0.0,
    azimuth_offset_deg: float = 0.0,
    **feature_kwargs,
) -> None:
    """
    GIF с вращением объекта и изменением voxel_size по кадрам.

    На каждом кадре:
    - строится новый `VOXELGRIDFEATURES` для соответствующего voxel_size
    - визуализируются центры вокселей
    - окраска берётся по `color_field`:
        - 'rgb' -> средний RGB по точкам внутри каждого вокселя
        - иначе -> фича из реестра (или атрибут вокселя), с нормализацией в [0,1] и палитрой blue->green->yellow->red

    :param pc: исходный PCD
    :param path_gif: путь для сохранения gif
    :param voxel_sizes: список voxel_size (каждый элемент = один кадр)
    :param orbit_turns: сколько оборотов камеры сделать за весь список кадров (если rotate=True)
    :param rotate: если False — камера фиксируется и НЕ вращается (для отладки/стабильности)
    :param backend:
        - "pyvista" (по умолчанию): пишет через `Plotter.open_gif()` + `write_frame()`
        - "imageio": рендерит кадры через `Plotter.screenshot()` и собирает GIF через `imageio`
          (полезно, если на вашей системе `open_gif()` даёт корректный только первый кадр)
    :param window_size: размер кадра (важно для backend="imageio")
    :param fps: FPS для backend="imageio" (альтернатива frame_duration)
    :param frame_duration: длительность кадра в секундах для backend="imageio" (если fps не задан)
    :param loop: количество повторов GIF. 0 = бесконечно (рекомендовано для “циклической” гифки).
    :param shift_to_center: если True, сдвигает точки так, чтобы центр объекта был около (0,0,0).
        Это часто исправляет “некорректное отображение” на больших координатах (UTM и т.п.).
    :param viewup: вектор “вверх” камеры (управляет наклоном/креном в составе camera_position).
        Часто достаточно (0,0,1), но можно задавать, например, (0,1,0).
    :param roll_deg: крен камеры в градусах (vtkCamera.Roll) после установки camera_position.
    :param elevation_deg: наклон камеры “вверх/вниз” в градусах (vtkCamera.Elevation) после установки camera_position.
    :param azimuth_offset_deg: сдвиг начального азимута орбиты (в градусах), полезно для выбора “стартового ракурса”.
    :param feature_kwargs: пробрасываются в вычисление фич (если `color_field` — фича)
    """
    import pyvista
    from .VOXELGRIDFEATURES import VOXELGRIDFEATURES

    voxel_sizes = [float(v) for v in (voxel_sizes or []) if float(v) > 0]
    if len(voxel_sizes) == 0:
        raise ValueError("voxel_sizes должен быть непустым списком положительных значений")
    backend = str(backend).lower().strip()
    if backend not in {"pyvista", "imageio"}:
        raise ValueError("backend должен быть 'pyvista' или 'imageio'")
    rotate = bool(rotate)
    try:
        loop = int(loop)
    except Exception:
        raise ValueError("loop должен быть int (0 = бесконечно)")
    if loop < 0:
        raise ValueError("loop должен быть >= 0 (0 = бесконечно)")
    shift_to_center = bool(shift_to_center)
    viewup = tuple(float(x) for x in viewup)
    if len(viewup) != 3:
        raise ValueError("viewup должен быть кортежем из 3 чисел")
    roll_deg = float(roll_deg)
    elevation_deg = float(elevation_deg)
    azimuth_offset_deg = float(azimuth_offset_deg)
    if fps is not None and fps <= 0:
        raise ValueError("fps должен быть положительным")
    if frame_duration is not None and frame_duration <= 0:
        raise ValueError("frame_duration должен быть положительным")

    # Центр и радиус орбиты — по центрам вокселей (стабильно между кадрами и ближе к видимой геометрии)
    vg_bounds = VOXELGRIDFEATURES.from_pcd(pc, voxel_sizes[0], verbose=False)
    pts_vis = np.asarray(vg_bounds.centers, dtype=np.float64)
    if pts_vis.size > 0:
        center = pts_vis.mean(axis=0)
        diag = pts_vis.max(axis=0) - pts_vis.min(axis=0)
        length = float(np.linalg.norm(diag))
    else:
        center = np.zeros((3,), dtype=np.float64)
        length = 1.0
    radius = max(length * 1.75, 1.0)
    z_lift = max(length * 0.75, 1.0)
    shift = center.copy() if shift_to_center else np.zeros((3,), dtype=np.float64)

    def colormap_bgyr(values_01: np.ndarray) -> np.ndarray:
        """blue -> green -> yellow -> red, values_01 в [0,1]."""
        v = np.asarray(values_01, dtype=np.float64)
        colors = np.zeros((v.shape[0], 3), dtype=np.float64)
        for i, x in enumerate(v):
            x = float(x)
            if x <= 0.33:
                t = x / 0.33
                colors[i] = [0.0, t, 1.0 - t]
            elif x <= 0.66:
                t = (x - 0.33) / (0.66 - 0.33)
                colors[i] = [t, 1.0, 0.0]
            else:
                t = (x - 0.66) / (1.0 - 0.66)
                colors[i] = [1.0, 1.0 - t, 0.0]
        return colors

    pl = pyvista.Plotter(off_screen=True, window_size=window_size)
    pl.background_color = background_color

    # Инициализация сцены первым кадром ДО open_gif (так стабильнее на разных бэкендах)
    n = len(voxel_sizes)
    first_vs = voxel_sizes[0]
    vg0 = VOXELGRIDFEATURES.from_pcd(pc, first_vs, verbose=False)
    cloud0 = pyvista.PointSet(vg0.centers - shift)
    if color_field == "rgb":
        colors0 = vg0._get_voxel_rgb()
    else:
        vals0 = vg0._get_voxel_scalar(color_field, component=component, **feature_kwargs)
        vals0 = vg0._normalize_01(vals0)
        colors0 = colormap_bgyr(vals0) if vals0.size > 0 else np.zeros((0, 3), dtype=np.float64)

    mesh_actor = pl.add_mesh(
        cloud0,
        scalars=colors0,
        rgb=True,
        opacity=1,
        point_size=point_size,
        show_scalar_bar=False,
    )
    text_actor = None
    if show_text:
        text_actor = pl.add_text(
            f"voxel_size={first_vs:g}",
            position="upper_left",
            font_size=12,
            color="black",
        )

    angle0 = np.deg2rad(azimuth_offset_deg)
    focus0 = (0.0, 0.0, 0.0) if shift_to_center else tuple(center.tolist())
    cam_pos0 = (focus0[0] + radius * np.cos(angle0), focus0[1] + radius * np.sin(angle0), focus0[2] + z_lift)
    pl.camera_position = (cam_pos0, focus0, viewup)
    try:
        if elevation_deg:
            pl.camera.Elevation(elevation_deg)
        if roll_deg:
            pl.camera.Roll(roll_deg)
        pl.camera.OrthogonalizeViewUp()
    except Exception:
        pass
    # ВАЖНО: vtkCamera.Zoom() мультипликативен. Не вызываем его на каждом кадре,
    # иначе будет “накапливающийся” зум. Вместо этого фиксируем view_angle один раз.
    try:
        base_view_angle = float(pl.camera.view_angle)
        if zoom and float(zoom) != 1.0:
            pl.camera.view_angle = base_view_angle / float(zoom)
    except Exception:
        pass
    pl.show(auto_close=False)

    if backend == "pyvista":
        # В разных версиях PyVista `open_gif` может/не может принимать `loop`
        try:
            pl.open_gif(path_gif, loop=loop)
        except TypeError:
            pl.open_gif(path_gif)
        pl.render()
        pl.write_frame()
    else:
        # backend == "imageio": собираем кадры через screenshot()
        try:
            import imageio.v2 as imageio  # type: ignore
        except Exception:
            import imageio  # type: ignore
        frames: list[np.ndarray] = []
        pl.render()
        frames.append(pl.screenshot(return_img=True))

    # Остальные кадры: удаляем предыдущие акторы и добавляем новые
    for i in range(1, n):
        vs = voxel_sizes[i]
        vg = VOXELGRIDFEATURES.from_pcd(pc, vs, verbose=False)
        cloud = pyvista.PointSet(vg.centers - shift)

        if color_field == "rgb":
            colors = vg._get_voxel_rgb()
        else:
            vals = vg._get_voxel_scalar(color_field, component=component, **feature_kwargs)
            vals = vg._normalize_01(vals)
            colors = colormap_bgyr(vals) if vals.size > 0 else np.zeros((0, 3), dtype=np.float64)

        try:
            pl.remove_actor(mesh_actor)
        except Exception:
            pass
        if text_actor is not None:
            try:
                pl.remove_actor(text_actor)
            except Exception:
                pass
            text_actor = None

        mesh_actor = pl.add_mesh(
            cloud,
            scalars=colors,
            rgb=True,
            opacity=1,
            point_size=point_size,
            show_scalar_bar=False,
        )
        if show_text:
            text_actor = pl.add_text(
                f"voxel_size={vs:g}",
                position="upper_left",
                font_size=12,
                color="black",
            )

        if rotate:
            frac = 0.0 if n <= 1 else (i / (n - 1))
            angle = np.deg2rad(azimuth_offset_deg) + 2.0 * np.pi * float(orbit_turns) * frac
            cam_pos = (focus0[0] + radius * np.cos(angle), focus0[1] + radius * np.sin(angle), focus0[2] + z_lift)
            pl.camera_position = (cam_pos, focus0, viewup)
            try:
                if elevation_deg:
                    pl.camera.Elevation(elevation_deg)
                if roll_deg:
                    pl.camera.Roll(roll_deg)
                pl.camera.OrthogonalizeViewUp()
            except Exception:
                pass

        pl.render()
        if backend == "pyvista":
            pl.write_frame()
        else:
            frames.append(pl.screenshot(return_img=True))

    if backend == "pyvista":
        pl.close()
        return

    # backend == "imageio": финальная сборка GIF
    pl.close()
    if fps is not None:
        duration = 1.0 / float(fps)
    else:
        duration = float(frame_duration) if frame_duration is not None else 0.12
    imageio.mimsave(path_gif, frames, duration=duration)


def visual_gif_inference_layers(
    pc: PCD,
    model,
    threshold: float,
    *,
    voxel_size: float,
    feature_names: List[str],
    feature_names_dist: List[str],
    path_gif: str,
    mode: Literal["cumulative", "current_layer"] = "cumulative",
    color_mode: Literal["proba", "mask", "label"] = "proba",
    zoom: float = 0.4,
    point_size: float = 6.0,
    rotate: bool = False,
    orbit_turns: float = 1.0,
    background_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    show_text: bool = True,
    window_size: Tuple[int, int] = (400, 400),
    fps: int | None = 8,
    frame_duration: float | None = None,
    loop: int = 0,
    pos_col: int = 1,
    shift_to_center: bool = True,
    show_unprocessed: bool = True,
    opacity: float = 1.0,
    viewup: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    roll_deg: float = 0.0,
    elevation_deg: float = 0.0,
    azimuth_offset_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, "VOXELGRIDFEATURES", np.ndarray]:
    """
    GIF по слоям Z “как идёт инференс”.

    Каждый кадр = завершение обработки очередного слоя (z) в цикле инференса.

    Рендер/запись сделаны максимально устойчиво на Windows: кадры берутся через
    `pyvista.Plotter.screenshot()` и собираются в gif через `imageio`.

    :returns: (final_predictions_voxels, real_target_voxels, grid, original_cloud_index)
    """
    import pyvista
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception:
        import imageio  # type: ignore

    from .VOXELGRIDFEATURES import VOXELGRIDFEATURES

    try:
        opacity = float(opacity)
    except Exception:
        raise ValueError("opacity должен быть числом в диапазоне [0, 1]")
    if not (0.0 <= opacity <= 1.0):
        raise ValueError("opacity должен быть в диапазоне [0, 1]")
    viewup = tuple(float(x) for x in viewup)
    if len(viewup) != 3:
        raise ValueError("viewup должен быть кортежем из 3 чисел")
    roll_deg = float(roll_deg)
    elevation_deg = float(elevation_deg)
    azimuth_offset_deg = float(azimuth_offset_deg)

    # ---- 1) Подготовка grid и статических фич (как в inference) ----
    dynamic_feats = ["distance_to_prev_layer", "distance_to_prev_layer_XY"]
    all_feats = list(feature_names) + list(feature_names_dist)
    static_feats = sorted(set(all_feats) - set(dynamic_feats))

    pc_real = pc.clone()
    grid = VOXELGRIDFEATURES.from_pcd(pc, voxel_size=voxel_size, verbose=False)
    real_target = grid.get_labels()

    slice_pc = get_trunk_slice(pc)
    coord = get_tree_coordinate(slice_pc)
    coord_kwargs = dict(coordinates=coord)

    if static_feats:
        grid.compute_features(static_feats, apply_to_voxels=False, coordinates=coord)

    final_predictions = np.zeros(len(grid), dtype=np.float32)
    prev_pred_mask = np.zeros(len(grid), dtype=bool)
    processed = np.zeros(len(grid), dtype=bool)

    # ---- 2) Камера/сцена ----
    pts_vis = np.asarray(grid.centers, dtype=np.float64)
    if pts_vis.size > 0:
        center = pts_vis.mean(axis=0)
        diag = pts_vis.max(axis=0) - pts_vis.min(axis=0)
        length = float(np.linalg.norm(diag))
    else:
        center = np.zeros((3,), dtype=np.float64)
        length = 1.0
    radius = max(length * 1.75, 1.0)
    z_lift = max(length * 0.75, 1.0)
    shift_to_center = bool(shift_to_center)
    shift = center.copy() if shift_to_center else np.zeros((3,), dtype=np.float64)

    def _normalize_01(values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        if v.size == 0:
            return v
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        vmin = float(v.min())
        vmax = float(v.max())
        denom = (vmax - vmin)
        if denom > 1e-12:
            return (v - vmin) / denom
        return np.zeros_like(v, dtype=np.float64)

    def colormap_bgyr(values_01: np.ndarray) -> np.ndarray:
        v = np.asarray(values_01, dtype=np.float64)
        colors = np.zeros((v.shape[0], 3), dtype=np.float64)
        for i, x in enumerate(v):
            x = float(x)
            if x <= 0.33:
                t = x / 0.33
                colors[i] = [0.0, t, 1.0 - t]
            elif x <= 0.66:
                t = (x - 0.33) / (0.66 - 0.33)
                colors[i] = [t, 1.0, 0.0]
            else:
                t = (x - 0.66) / (1.0 - 0.66)
                colors[i] = [1.0, 1.0 - t, 0.0]
        return colors

    pl = pyvista.Plotter(off_screen=True, window_size=window_size)
    pl.background_color = background_color
    pl.show(auto_close=False)

    # Фиксируем камеру (если rotate=False) на первом кадре.
    focus = (0.0, 0.0, 0.0) if shift_to_center else tuple(center.tolist())
    cam_fixed = (
        (focus[0] + radius, focus[1], focus[2] + z_lift),
        focus,
        viewup,
    )
    pl.camera_position = cam_fixed
    try:
        if elevation_deg:
            pl.camera.Elevation(elevation_deg)
        if roll_deg:
            pl.camera.Roll(roll_deg)
        pl.camera.OrthogonalizeViewUp()
    except Exception:
        pass
    # Зум фиксируем через view_angle, чтобы он не “накапливался” на кадрах.
    try:
        base_view_angle = float(pl.camera.view_angle)
        if zoom and float(zoom) != 1.0:
            pl.camera.view_angle = base_view_angle / float(zoom)
    except Exception:
        pass

    frames: list[np.ndarray] = []
    cam_locked = False

    # ---- 3) Цикл по слоям (как в inference) + кадры ----
    zs = sorted(grid.layer_to_indices.keys())
    model_cols = getattr(model, "feature_names_", None)
    if model_cols is None:
        raise ValueError("model должен иметь атрибут feature_names_ (как у CatBoost/ sklearn-обёрток)")

    for step, z in enumerate(zs):
        idx_cur = grid.layer_to_indices[z]

        df_layer = grid.get_features_df_for_layer(
            all_feats,
            z,
            dynamic_features=dynamic_feats,
            prev_pred_mask=prev_pred_mask,
            **coord_kwargs,
        )
        if not df_layer.empty:
            X = df_layer.reindex(columns=model_cols, fill_value=0.0)
            y_proba = model.predict_proba(X.values)[:, pos_col]
            final_predictions[idx_cur] = y_proba
            prev_pred_mask[idx_cur] = (y_proba <= threshold)
            processed[idx_cur] = True
        else:
            processed[idx_cur] = True

        # --- подготовка цветов для кадра ---
        N = len(grid)
        # default: тёмно-серый, чтобы объект был виден даже до инференса
        colors = np.ones((N, 3), dtype=np.float64) * np.array([0.2, 0.2, 0.2], dtype=np.float64)

        if mode == "current_layer":
            visible = np.zeros((N,), dtype=bool)
            visible[idx_cur] = True
        else:
            visible = processed.copy()

        if show_unprocessed and mode == "cumulative":
            # показываем весь объект всегда, а “обработанное” подсвечиваем
            visible = np.ones((N,), dtype=bool)
            processed_visible = processed.copy()
        else:
            processed_visible = visible.copy()

        if color_mode == "proba":
            # Нормализуем только “обработанные” (иначе первые кадры все одинаковые)
            if processed_visible.any():
                vals_proc = final_predictions[processed_visible].astype(np.float64, copy=False)
                norm_proc = _normalize_01(vals_proc)
                cmap_proc = colormap_bgyr(norm_proc)
                colors[processed_visible] = cmap_proc
        elif color_mode == "mask":
            # prev_pred_mask True/False (видимые слои)
            colors[processed_visible & (~prev_pred_mask)] = np.array([1.0, 0.0, 0.0])  # red
            colors[processed_visible & (prev_pred_mask)] = np.array([0.0, 0.2, 1.0])   # blue-ish
        else:  # "label"
            # Реальная метка.
            # По умолчанию показываем как grayscale, НО label==0 делаем красным (вместо чёрного).
            rt = real_target.astype(np.int64, copy=False)
            lab = _normalize_01(rt.astype(np.float64, copy=False))
            gray = np.stack([lab, lab, lab], axis=1)
            colors[processed_visible] = gray[processed_visible]

            # label==0 -> red
            mask0 = processed_visible & (rt == 0)
            if mask0.any():
                colors[mask0] = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        # PolyData обычно стабильнее для point clouds, чем PointSet
        cloud = pyvista.PolyData(grid.centers - shift)

        pl.clear()
        pl.add_mesh(
            cloud,
            scalars=colors,
            rgb=True,
            opacity=opacity,
            point_size=point_size,
            show_scalar_bar=False,
            render_points_as_spheres=True,
        )
        if show_text:
            pl.add_text(
                f"step={step+1}/{len(zs)}  thr={threshold:g}  voxel_size={voxel_size:g}",
                position="upper_left",
                font_size=12,
                color="black",
            )

        # Камеру лучше “поймать” по первому кадру через bounds, затем фиксировать
        if not cam_locked:
            try:
                pl.reset_camera()
            except Exception:
                pass
            # После reset_camera снова фиксируем view_angle (reset_camera может менять дистанцию/clip,
            # но view_angle обычно сохраняется; всё равно приводим к желаемому значению).
            try:
                base_view_angle = float(pl.camera.view_angle)
                if zoom and float(zoom) != 1.0:
                    pl.camera.view_angle = base_view_angle / float(zoom)
            except Exception:
                pass
            cam_fixed = pl.camera_position
            cam_locked = True

        if rotate:
            frac = 0.0 if len(zs) <= 1 else (step / (len(zs) - 1))
            angle = np.deg2rad(azimuth_offset_deg) + 2.0 * np.pi * float(orbit_turns) * frac
            cam_pos = (focus[0] + radius * np.cos(angle), focus[1] + radius * np.sin(angle), focus[2] + z_lift)
            pl.camera_position = (cam_pos, focus, viewup)
            try:
                if elevation_deg:
                    pl.camera.Elevation(elevation_deg)
                if roll_deg:
                    pl.camera.Roll(roll_deg)
                pl.camera.OrthogonalizeViewUp()
            except Exception:
                pass
        else:
            pl.camera_position = cam_fixed

        pl.render()
        frames.append(pl.screenshot(return_img=True))

    pl.close()

    # ---- 4) Запись gif (цикличный loop=0 по умолчанию) ----
    if fps is not None and fps > 0:
        duration = 1.0 / float(fps)
    else:
        duration = float(frame_duration) if frame_duration is not None else 0.12
    imageio.mimsave(path_gif, frames, duration=duration, loop=int(loop))

    return final_predictions, real_target, grid, pc_real.original_cloud_index
