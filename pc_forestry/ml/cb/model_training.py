from pc_forestry.ml.cnn3d.infer import infer_for_features_df, apply_predictions_to_voxelgridfeatures
from pc_forestry.ml.cnn3d.train import train_from_df
from pc_forestry.ml.pointnet2.infer import infer_for_points_df
from pc_forestry.ml.pointnet2.train import train_from_df as train_pointnet2_from_df
import os
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
import inspect

import numpy as np
import pandas as pd
import shutil

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    jaccard_score as sk_jaccard_score,
)
try:
    from pytorch_tabnet.tab_model import TabNetClassifier  # pyright: ignore[reportMissingImports]
except ImportError:
    TabNetClassifier = None

from scipy.spatial import cKDTree


from time import perf_counter
from collections import defaultdict
from contextlib import contextmanager

from pc_forestry.pcd.PCD import PCD
from pc_forestry.pcd.VOXELGRIDFEATURES import VOXELGRIDFEATURES
from pc_forestry.pcd.utils import get_trunk_slice, get_tree_coordinate


POINTNET2_TARGET_DEPENDENT_FEATURES = {
    "label",
    "distance_to_prev_layer",
    "distance_to_prev_layer_XY",
    "distance",
    "distance_XY",
}


@contextmanager
def timed(stats, name):
    t0 = perf_counter()
    try:
        yield
    finally:
        stats[name] += perf_counter() - t0


def inference_cnn3d(
    pc,
    voxel_size,
    feature_names,
    feature_names_dist,
    checkpoint="checkpoints/cnn3d_model.pt"
):
    pc_real = pc.clone()

    grid = VOXELGRIDFEATURES.from_pcd(pc, voxel_size=voxel_size)
    grid_real = VOXELGRIDFEATURES.from_pcd(pc_real, voxel_size=voxel_size)
    real_target = grid_real.get_labels()

    slice = get_trunk_slice(pc)
    coord = get_tree_coordinate(slice)
    coord_kwargs = dict(coordinates=coord)

    grid.compute_features(feature_names + feature_names_dist, **coord_kwargs, apply_to_voxels=False)
    df = grid.get_features_df(feature_names + feature_names_dist)

    df_pred = infer_for_features_df(df, checkpoint_path=checkpoint, name="one_tree")
    apply_predictions_to_voxelgridfeatures(grid, df_pred)

    # берём вероятности (для бинарки обычно нужно proba_1)
    if "proba_1" in df_pred.columns:
        voxel_proba = df_pred["proba_1"].to_numpy(dtype=np.float32)
    else:
        proba_cols = [c for c in df_pred.columns if c.startswith("proba_")]
        voxel_proba = df_pred[proba_cols].max(axis=1).to_numpy(dtype=np.float32)     # float на каждую точку

    return grid, real_target, voxel_proba


def build_pointnet2_features_df(
    pc,
    *,
    voxel_size,
    feature_names,
    feature_names_dist,
):
    all_feats = feature_names + feature_names_dist
    bad_feats = sorted(set(all_feats) & POINTNET2_TARGET_DEPENDENT_FEATURES)
    if bad_feats:
        raise ValueError(
            "PointNet++ не поддерживает target-dependent признаки, так как это даёт data leak: "
            f"{bad_feats}. Убери их из feature_names/feature_names_dist."
        )

    pc_inference = pc.clone()
    pc_inference.original_cloud_index = np.zeros(len(pc_inference.points), dtype=np.float32)
    pc_real = pc.clone()

    slice_pc = get_trunk_slice(pc_inference)
    coord = get_tree_coordinate(slice_pc)
    grid = VOXELGRIDFEATURES.from_pcd(pc_inference, voxel_size=voxel_size, verbose=False)
    grid.compute_features(all_feats, coordinates=coord, apply_to_voxels=False)

    point_df = pc_real.df[["x", "y", "z", "original_cloud_index"]].copy()
    point_df["point_index"] = np.arange(len(point_df), dtype=np.int64)
    point_df["target"] = (point_df["original_cloud_index"].astype(int) != 0).astype(np.int32)
    point_df.drop(columns=["original_cloud_index"], inplace=True, errors="ignore")

    if all_feats:
        voxel_df = grid.get_features_df(all_feats)
        inv = np.asarray(grid._inverse, dtype=np.int64)
        voxel_df_points = voxel_df.iloc[inv].reset_index(drop=True)
        feature_cols_expanded = [c for c in voxel_df_points.columns if c not in {"x", "y", "z"}]
        for col in feature_cols_expanded:
            point_df[col] = voxel_df_points[col].to_numpy()

    return point_df, grid


def inference_pointnet2(
    pc,
    *,
    voxel_size,
    feature_names,
    feature_names_dist,
    checkpoint="checkpoints/pointnet2_model.pt",
    num_votes: int = 3,
):
    point_df, grid = build_pointnet2_features_df(
        pc,
        voxel_size=voxel_size,
        feature_names=feature_names,
        feature_names_dist=feature_names_dist,
    )
    df_pred = infer_for_points_df(
        point_df,
        checkpoint_path=checkpoint,
        name="one_tree",
        num_votes=num_votes,
    )

    if "proba_1" in df_pred.columns:
        point_proba = df_pred["proba_1"].to_numpy(dtype=np.float32)
    else:
        proba_cols = [c for c in df_pred.columns if c.startswith("proba_")]
        point_proba = df_pred[proba_cols].max(axis=1).to_numpy(dtype=np.float32)

    real_target = point_df["target"].to_numpy(dtype=np.int32)
    return df_pred, real_target, point_proba, grid


def inference(pc, model, threshold, *,
              voxel_size: float,
              feature_names: List[str],
              feature_names_dist: List[str],
              model_feature_names=None,
              model_type: str = "catboost",
              tabnet_fill_values: Optional[Dict[str, float]] = None):
    """
    Инференс с проверкой на target leak.

    ВАЖНО: Используем отдельные копии облака точек для построения сетки и получения ground truth,
    чтобы избежать target leak через original_cloud_index или другие артефакты.
    """
    stats = defaultdict(float)

    dynamic_feats = ['distance_to_prev_layer', 'distance_to_prev_layer_XY']
    all_feats = feature_names + feature_names_dist
    static_feats = sorted(set(all_feats) - set(dynamic_feats))

    with timed(stats, "clone_pc"):
        # Создаем чистую копию для инференса (без меток)
        pc_inference = pc.clone()
        # Обнуляем original_cloud_index чтобы исключить любую связь с ground truth
        pc_inference.original_cloud_index = np.zeros(len(pc_inference.intensity), dtype=np.float32)

        # Отдельная копия для получения ground truth
        pc_real = pc.clone()

    with timed(stats, "build_grid"):
        # Используем очищенную копию для построения сетки
        grid = VOXELGRIDFEATURES.from_pcd(pc_inference, voxel_size=voxel_size)

    with timed(stats, "get_labels"):
        # Ground truth получаем из оригинального облака
        grid_real = VOXELGRIDFEATURES.from_pcd(pc_real, voxel_size=voxel_size)
        real_target = grid_real.get_labels()

    with timed(stats, "get_trunk_slice"):
        slice_pc = get_trunk_slice(pc_inference)
    with timed(stats, "get_tree_coordinate"):
        coord = get_tree_coordinate(slice_pc)
    coord_kwargs = dict(coordinates=coord)

    if static_feats:
        with timed(stats, "compute_static_features"):
            grid.compute_features(static_feats, apply_to_voxels=False, coordinates=coord)

    with timed(stats, "alloc_arrays"):
        final_predictions = np.zeros(len(grid), dtype=np.float32)
        prev_pred_mask = np.zeros(len(grid), dtype=bool)
    pos_col = 1

    for z in sorted(grid.layer_to_indices.keys()):
        idx_cur = grid.layer_to_indices[z]

        with timed(stats, "layer_get_df"):
            df_layer = grid.get_features_df_for_layer(
                all_feats,
                z,
                dynamic_features=dynamic_feats,
                prev_pred_mask=prev_pred_mask,
                **coord_kwargs
            )
        if df_layer.empty:
            continue

        with timed(stats, "layer_reindex"):
            model_cols = model.feature_names_ if model_feature_names is None else model_feature_names
            X = df_layer.reindex(columns=model_cols, fill_value=0.0)

        with timed(stats, "layer_predict_proba"):
            if model_type == "tabnet":
                X_tabnet, _ = _sanitize_tabnet_frame(X, model_cols, fill_values=tabnet_fill_values)
                y_proba = model.predict_proba(X_tabnet.to_numpy(dtype=np.float32, copy=False))[:, pos_col]
            else:
                y_proba = model.predict_proba(X.values)[:, pos_col]

        with timed(stats, "layer_write_back"):
            final_predictions[idx_cur] = y_proba
            prev_pred_mask[idx_cur] = (y_proba <= threshold)

    return final_predictions, real_target, grid, pc_real.original_cloud_index, dict(stats)


# ----------------- Point IoU helper -----------------


def iou_points_radius(pred_xyz: np.ndarray,
                      gt_xyz: np.ndarray,
                      r: float,
                      *,
                      workers: int = -1) -> float:
    """
    IoU по точкам в 3D: пересечение = число уникальных GT-точек,
    к которым нашлась хотя бы одна pred-точка в радиусе r.
    """
    pred_xyz = np.asarray(pred_xyz, dtype=np.float64)
    gt_xyz = np.asarray(gt_xyz, dtype=np.float64)

    if len(pred_xyz) == 0 and len(gt_xyz) == 0:
        return 1.0
    if len(pred_xyz) == 0 or len(gt_xyz) == 0:
        return 0.0

    tree = cKDTree(gt_xyz)
    neigh_lists = tree.query_ball_point(pred_xyz, r, workers=workers)  # [web:7]

    matched_gt = set()
    for neigh in neigh_lists:
        if neigh:
            matched_gt.update(neigh)

    inter = len(matched_gt)
    union = len(pred_xyz) + len(gt_xyz) - inter
    return inter / union if union > 0 else 0.0


def _sanitize_tabnet_frame(
    X: pd.DataFrame,
    feature_cols: List[str],
    fill_values: Optional[Dict[str, float]] = None,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    X_local = X.reindex(columns=feature_cols, fill_value=np.nan).copy()
    X_local.replace([np.inf, -np.inf], np.nan, inplace=True)

    medians: Dict[str, float] = {}
    if fill_values:
        medians.update({str(k): float(v) for k, v in fill_values.items()})

    for col in feature_cols:
        series = pd.to_numeric(X_local[col], errors="coerce").astype(np.float32)
        median_val = medians.get(col)
        if median_val is None:
            arr = series.to_numpy(dtype=np.float32, copy=False)
            if np.isnan(arr).all():
                median_val = 0.0
            else:
                median_val = float(np.nanmedian(arr))
                if not np.isfinite(median_val):
                    median_val = 0.0
            medians[col] = median_val
        X_local[col] = series.fillna(median_val).astype(np.float32)

    return X_local, medians


# ----------------- Config -----------------

@dataclass
class ExperimentConfig:
    name: str
    feature_names: List[str]
    feature_names_dist: List[str]
    voxel_size: float = 0.2
    model_type: str = "catboost"
    catboost_params: Dict[str, Any] = field(default_factory=dict)
    cnn3d_params: Dict[str, Any] = field(default_factory=dict)
    tabnet_params: Dict[str, Any] = field(default_factory=dict)
    pointnet2_params: Dict[str, Any] = field(default_factory=dict)
    default_threshold: float = 0.3

    # Радиус для point-IoU (в единицах xyz)
    iou_radius: float = 0.05


class ModelTraining:
    """
    Стенд для:
    - подготовки фич из папки с .txt,
    - обучения CatBoost,
    - инференса по PC / по тестовому датасету,
    - сохранения всего в директорию эксперимента.

    ВАЖНО: чтобы посчитать IoU по точкам, сохраняется отдельный points_test.csv.
    """

    def __init__(self, base_dir: str, name: str):
        self.base_dir = base_dir
        self.exp_name = name
        self.exp_dir = os.path.join(base_dir, name)
        os.makedirs(self.exp_dir, exist_ok=True)

        self.config = ExperimentConfig(
            name=name,
            feature_names=[],
            feature_names_dist=[],
            catboost_params={},
        )

        self.model: Optional[Any] = None
        self.model_cols: Optional[List[str]] = None

    # ---------- builder-style методы ----------

    def with_feature_names(self, feature_names: List[str]):
        self.config.feature_names = feature_names
        return self

    def with_feature_names_dist(self, feature_names_dist: List[str]):
        self.config.feature_names_dist = feature_names_dist
        return self

    def with_hyperparams(self, params: Dict[str, Any]):
        if "voxel_size" in params:
            self.config.voxel_size = params["voxel_size"]
        if "model_type" in params:
            self.config.model_type = params["model_type"]
        if "model" in params:
            self.config.model_type = params["model"]
        if "catboost" in params:
            self.config.catboost_params = params["catboost"]
        if "cnn3d" in params:
            self.config.cnn3d_params = params["cnn3d"]
        if "tabnet" in params:
            self.config.tabnet_params = params["tabnet"]
        if "pointnet2" in params:
            self.config.pointnet2_params = params["pointnet2"]
        if "pointnet++" in params:
            self.config.pointnet2_params = params["pointnet++"]
        if "threshold" in params:
            self.config.default_threshold = params["threshold"]
        if "iou_radius" in params:
            self.config.iou_radius = params["iou_radius"]
        return self

    # ---------- служебные пути ----------

    @property
    def config_path(self):
        return os.path.join(self.exp_dir, "config.json")

    @property
    def model_path(self):
        if self._model_type() == "cnn3d":
            return os.path.join(self.exp_dir, "cnn3d_model.pt")
        if self._model_type() == "pointnet2":
            return os.path.join(self.exp_dir, "pointnet2_model.pt")
        if self._model_type() == "tabnet":
            return os.path.join(self.exp_dir, "tabnet_model.zip")
        return os.path.join(self.exp_dir, "model.cbm")

    @property
    def model_meta_path(self):
        return os.path.splitext(self.model_path)[0] + ".meta.json"

    @property
    def log_path(self):
        return os.path.join(self.exp_dir, "metrics.json")

    @property
    def features_train_path(self):
        return os.path.join(self.exp_dir, "features_train.csv")

    @property
    def features_test_path(self):
        return os.path.join(self.exp_dir, "features_test.csv")

    @property
    def points_test_path(self):
        # отдельный файл для point-level метрик (IoU в пространстве)
        return os.path.join(self.exp_dir, "points_test.csv")

    # ---------- сохранение / загрузка конфигурации ----------

    def _save_config(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, ensure_ascii=False, indent=2)

    def _model_type(self) -> str:
        model_type = str(getattr(self.config, "model_type", "catboost") or "catboost").strip().lower()
        aliases = {
            "catboost": "catboost",
            "cb": "catboost",
            "cnn3d": "cnn3d",
            "cnn": "cnn3d",
            "cnn3d_model": "cnn3d",
            "pointnet2": "pointnet2",
            "pointnet++": "pointnet2",
            "pointnetpp": "pointnet2",
            "tabnet": "tabnet",
        }
        if model_type not in aliases:
            raise ValueError(f"Неизвестный model_type: {self.config.model_type}")
        return aliases[model_type]

    @classmethod
    def from_existing(cls, exp_dir: str) -> "ModelTraining":
        with open(os.path.join(exp_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg = ExperimentConfig(**cfg_dict)

        base_dir = os.path.dirname(exp_dir)
        mt = cls(base_dir=base_dir, name=cfg.name)
        mt.config = cfg

        if mt._model_type() == "catboost":
            model = CatBoostClassifier()
            model.load_model(mt.model_path)
            mt.model = model

            fi = model.get_feature_importance(prettified=True)
            mt.model_cols = list(fi["Feature Id"])
        elif mt._model_type() == "tabnet":
            if TabNetClassifier is None:
                raise ImportError("Для загрузки TabNet нужен пакет pytorch-tabnet.")
            model = TabNetClassifier()
            model.load_model(mt.model_path)
            mt.model = model
            if os.path.exists(mt.model_meta_path):
                with open(mt.model_meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                mt.model_cols = meta.get("feature_cols")
        elif mt._model_type() == "pointnet2":
            if os.path.exists(mt.model_meta_path):
                with open(mt.model_meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                mt.model_cols = meta.get("feature_cols")
        elif os.path.exists(mt.model_meta_path):
            with open(mt.model_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            mt.model_cols = meta.get("feature_cols")

        return mt

    # ---------- подготовка датасетов ----------

    def _prepare_features_for_folder(self, folder: str, output_csv: str):
        feature_names = self.config.feature_names
        feature_names_dist = self.config.feature_names_dist
        voxel_size = self.config.voxel_size

        all_dfs = []
        prepared_dir = os.path.join(folder, "prepared")

        for file in os.listdir(prepared_dir):
            if not file.endswith(".txt"):
                continue

            pc = PCD.read(os.path.join(prepared_dir, file))
            slice_pc = get_trunk_slice(pc)
            coord = get_tree_coordinate(slice_pc)

            grid = VOXELGRIDFEATURES.from_pcd(pc, voxel_size=voxel_size, verbose=False)
            _ = grid.compute_features(feature_names + feature_names_dist, coordinates=coord)
            df = grid.get_features_df(feature_names + feature_names_dist)
            y = grid.get_labels()

            df["target"] = y
            df["name"] = file.replace(".txt", "")
            all_dfs.append(df)

        if not all_dfs:
            raise ValueError(f"Нет файлов для подготовки фич в {prepared_dir}")

        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(output_csv, index=False, sep=";")

    def _prepare_pointnet2_features_for_folder(self, folder: str, output_csv: str):
        feature_names = self.config.feature_names
        feature_names_dist = self.config.feature_names_dist
        voxel_size = self.config.voxel_size

        all_dfs = []
        prepared_dir = os.path.join(folder, "prepared")

        for file in os.listdir(prepared_dir):
            if not file.endswith(".txt"):
                continue

            pc = PCD.read(os.path.join(prepared_dir, file))
            df_points, _ = build_pointnet2_features_df(
                pc,
                voxel_size=voxel_size,
                feature_names=feature_names,
                feature_names_dist=feature_names_dist,
            )
            df_points["name"] = file.replace(".txt", "")
            all_dfs.append(df_points)

        if not all_dfs:
            raise ValueError(f"Нет файлов для подготовки point features в {prepared_dir}")

        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(output_csv, index=False, sep=";")

    def prepare_datasets(self,
                         train_folder: str,
                         test_folder: Optional[str] = None,
                         force: bool = False):
        self._save_config()

        if not os.path.exists(self.features_train_path) or force:
            if self._model_type() == "pointnet2":
                self._prepare_pointnet2_features_for_folder(train_folder, self.features_train_path)
            else:
                self._prepare_features_for_folder(train_folder, self.features_train_path)

        if test_folder is not None:
            if not os.path.exists(self.features_test_path) or force:
                if self._model_type() == "pointnet2":
                    self._prepare_pointnet2_features_for_folder(test_folder, self.features_test_path)
                else:
                    self._prepare_features_for_folder(test_folder, self.features_test_path)

        return self

    # ---------- обучение ----------

    def train(self):
        df = pd.read_csv(self.features_train_path, sep=";")
        if self._model_type() == "cnn3d":
            params = {
                "epochs": 30,
                "batch_size": 2,
                "lr": 2e-3,
            }
            params.update(self.config.cnn3d_params or {})
            allowed_params = set(inspect.signature(train_from_df).parameters)
            cnn3d_params = {k: v for k, v in params.items() if k in allowed_params}

            ckpt_path = train_from_df(
                df,
                out_path=self.model_path,
                **cnn3d_params,
            )

            self.model = None
            if os.path.exists(self.model_meta_path):
                with open(self.model_meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.model_cols = meta.get("feature_cols")

            metrics = {
                "model_type": "cnn3d",
                "checkpoint": ckpt_path,
            }
        elif self._model_type() == "pointnet2":
            params = {
                "epochs": 50,
                "batch_size": 8,
                "lr": 1e-3,
                "num_points": 1024,
                "samples_per_epoch": 128,
            }
            params.update(self.config.pointnet2_params or {})
            allowed_params = set(inspect.signature(train_pointnet2_from_df).parameters)
            pointnet2_params = {k: v for k, v in params.items() if k in allowed_params}

            ckpt_path = train_pointnet2_from_df(
                df,
                out_path=self.model_path,
                **pointnet2_params,
            )

            self.model = None
            if os.path.exists(self.model_meta_path):
                with open(self.model_meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.model_cols = meta.get("feature_cols")

            metrics = {
                "model_type": "pointnet2",
                "checkpoint": ckpt_path,
            }
        elif self._model_type() == "tabnet":
            if TabNetClassifier is None:
                raise ImportError("Для обучения TabNet нужен пакет pytorch-tabnet.")

            import torch

            y = df["target"]
            X = df.drop(columns=["target"])
            X = X.select_dtypes(include=[np.number])

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.1, random_state=42, stratify=y
            )

            feature_cols = list(X_train.columns)
            X_train_clean, tabnet_fill_values = _sanitize_tabnet_frame(X_train, feature_cols)
            X_val_clean, _ = _sanitize_tabnet_frame(X_val, feature_cols, fill_values=tabnet_fill_values)

            X_train_np = X_train_clean.to_numpy(dtype=np.float32, copy=False)
            y_train_np = y_train.values.astype(int)
            X_val_np = X_val_clean.to_numpy(dtype=np.float32, copy=False)
            y_val_np = y_val.values.astype(int)

            params = {
                "n_d": 64,
                "n_a": 64,
                "n_steps": 5,
                "gamma": 1.5,
                "n_independent": 2,
                "n_shared": 2,
                "lambda_sparse": 1e-4,
                "optimizer_fn": torch.optim.Adam,
                "optimizer_params": {"lr": 2e-2},
                "mask_type": "entmax",
                "scheduler_params": {"step_size": 50, "gamma": 0.9},
                "scheduler_fn": torch.optim.lr_scheduler.StepLR,
                "verbose": 1,
                "device_name": "cuda" if torch.cuda.is_available() else "cpu",
            }
            params.update(self.config.tabnet_params or {})

            fit_params = {
                "eval_set": [(X_val_np, y_val_np)],
                "eval_name": ["val"],
                "eval_metric": ["logloss", "auc"],
                "max_epochs": 100,
                "patience": 20,
                "batch_size": 2048,
                "virtual_batch_size": 256,
                "drop_last": False,
            }
            for key in ("eval_set", "eval_name", "eval_metric", "max_epochs", "patience", "batch_size", "virtual_batch_size", "drop_last"):
                if key in params:
                    fit_params[key] = params.pop(key)

            model = TabNetClassifier(**params)
            model.fit(X_train_np, y_train_np, **fit_params)

            self.model = model
            self.model_cols = feature_cols
            model.save_model(os.path.splitext(self.model_path)[0])

            with open(self.model_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "feature_cols": feature_cols,
                        "tabnet_fill_values": tabnet_fill_values,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            y_val_proba = model.predict_proba(X_val_np)[:, 1]
            val_auc = roc_auc_score(y_val, y_val_proba)
            metrics = {"model_type": "tabnet", "val_auc": float(val_auc)}
        else:
            y = df["target"]
            X = df.drop(columns=["target"])

            X = X.select_dtypes(include=[np.number])

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.1, random_state=42, stratify=y
            )

            params = {
                "iterations": 1000,
                "learning_rate": 0.05,
                "depth": 6,
                "random_seed": 42,
                "verbose": 100,
                "early_stopping_rounds": 50,
            }
            params.update(self.config.catboost_params or {})

            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val))

            self.model = model
            self.model_cols = model.feature_names_
            model.save_model(self.model_path)

            y_val_proba = model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, y_val_proba)

            metrics = {"model_type": "catboost", "val_auc": float(val_auc)}

        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        return self

    def inference(self, pc, threshold: Optional[float] = None):
        if threshold is None:
            threshold = self.config.default_threshold

        if self._model_type() == "cnn3d":
            grid, real_target, voxel_proba = inference_cnn3d(
                pc=pc,
                voxel_size=self.config.voxel_size,
                feature_names=self.config.feature_names,
                feature_names_dist=self.config.feature_names_dist,
                checkpoint=self.model_path,
            )
            original_cloud_index = getattr(pc, "original_cloud_index", None)
            return voxel_proba, real_target, grid, original_cloud_index, {}
        if self._model_type() == "pointnet2":
            df_pred, real_target, point_proba, _ = inference_pointnet2(
                pc=pc,
                voxel_size=self.config.voxel_size,
                feature_names=self.config.feature_names,
                feature_names_dist=self.config.feature_names_dist,
                checkpoint=self.model_path,
                num_votes=int(self.config.pointnet2_params.get("num_votes", 3)),
            )
            original_cloud_index = getattr(pc, "original_cloud_index", None)
            return point_proba, real_target, df_pred, original_cloud_index, {}

        tabnet_fill_values = None
        if self._model_type() == "tabnet":
            if self.model is None:
                if TabNetClassifier is None:
                    raise ImportError("Для инференса TabNet нужен пакет pytorch-tabnet.")
                model = TabNetClassifier()
                model.load_model(self.model_path)
                self.model = model
            if os.path.exists(self.model_meta_path):
                with open(self.model_meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if self.model_cols is None:
                    self.model_cols = meta.get("feature_cols")
                tabnet_fill_values = meta.get("tabnet_fill_values")
            if not self.model_cols:
                raise ValueError("Для TabNet не найдены feature_cols в метаданных эксперимента.")
        elif self.model is None:
            model = CatBoostClassifier()
            model.load_model(self.model_path)
            self.model = model

        return inference(
            pc=pc,
            model=self.model,
            threshold=threshold,
            voxel_size=self.config.voxel_size,
            feature_names=self.config.feature_names,
            feature_names_dist=self.config.feature_names_dist,
            model_feature_names=self.model_cols,
            model_type=self._model_type(),
            tabnet_fill_values=tabnet_fill_values,
        )

    # ---------- predict_on_test: voxels + points_test.csv ----------

    def predict_on_test(self, test_folder: str, threshold: Optional[float] = None) -> Optional[pd.DataFrame]:
        prepared_dir = os.path.join(test_folder, "prepared")
        if not os.path.exists(prepared_dir):
            return None

        thr = self.config.default_threshold if threshold is None else float(threshold)

        all_records = []
        all_point_records = []

        for file in os.listdir(prepared_dir):
            if not file.endswith(".txt"):
                continue

            pc = PCD.read(os.path.join(prepared_dir, file))
            file_name = file.replace(".txt", "")
            if self._model_type() == "pointnet2":
                preds, targets, _, _, _ = self.inference(pc, threshold=thr)
                n_points = len(preds)

                df_file = pd.DataFrame({
                    "name": [file_name] * n_points,
                    "voxel_index": np.arange(n_points, dtype=int),
                    "pred_proba": preds.astype(float),
                    "target": targets.astype(int),
                })
                all_records.append(df_file)

                point_pred_proba = preds.astype(np.float32)
                point_pred_bin = (point_pred_proba > thr).astype(np.int32)

                dfp = pc.df[["x", "y", "z", "original_cloud_index"]].copy()
                dfp["name"] = file_name
                dfp["pred"] = point_pred_bin
                dfp["gt"] = (dfp["original_cloud_index"].astype(int) != 0).astype(np.int32)
                all_point_records.append(dfp[["name", "x", "y", "z", "pred", "gt"]])

                from pc_forestry.pcd.fields import ScalarField

                class PredField(ScalarField):
                    @property
                    def name(self): return "pred"

                pc._fields["pred"] = PredField(point_pred_proba)

                class Label(ScalarField):
                    @property
                    def name(self) -> str: return "label"

                pc._fields["label"] = Label(point_pred_bin)
                continue

            preds, targets, grid, _, _ = self.inference(pc, threshold=thr)
            n_voxels = len(preds)

            # -------- voxel-level dataset (как было) --------
            df_file = pd.DataFrame({
                "name": [file_name] * n_voxels,
                "voxel_index": np.arange(n_voxels, dtype=int),
                "pred_proba": preds.astype(float),
                "target": targets.astype(int),
            })
            all_records.append(df_file)

            # -------- point-level dataset для IoU --------
            point_pred_proba = preds[grid._inverse].astype(np.float32)
            point_pred_bin = (point_pred_proba > thr).astype(np.int32)

            # pc.df должен содержать x,y,z и истинную метку/индикатор в original_cloud_index
            dfp = pc.df[["x", "y", "z", "original_cloud_index"]].copy()
            dfp["name"] = file_name
            dfp["pred"] = point_pred_bin

            # ВАЖНО: тут предполагается, что original_cloud_index — это GT в {0,1}
            dfp["gt"] = (dfp["original_cloud_index"].astype(int) != 0).astype(np.int32)

            all_point_records.append(dfp[["name", "x", "y", "z", "pred", "gt"]])

            # -------- визуализация (оставил как у тебя) --------
            from pc_forestry.pcd.fields import ScalarField

            class PredField(ScalarField):
                @property
                def name(self): return "pred"

            pc._fields["pred"] = PredField(point_pred_proba)
            # pc.show(color_field="pred")

            class Label(ScalarField):
                @property
                def name(self) -> str: return "label"

            pc._fields["label"] = Label(point_pred_bin)
            # pc.show(color_field="label")

        if not all_records:
            return None

        df_all = pd.concat(all_records, ignore_index=True)
        df_all.to_csv(self.features_test_path, index=False, sep=";")

        if all_point_records:
            df_points = pd.concat(all_point_records, ignore_index=True)
            df_points.to_csv(self.points_test_path, index=False, sep=";")

        return df_all

    # ---------- eval: + point IoU ----------

    def eval(self) -> Dict[str, float]:
        if not os.path.exists(self.features_test_path):
            raise ValueError("Нет test датасета для eval (features_test.csv)")

        df = pd.read_csv(self.features_test_path, sep=";")
        required_cols = {"name", "voxel_index", "pred_proba", "target"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"features_test.csv должен содержать колонки: {required_cols}")

        y_true = df["target"].values.astype(int)
        y_score = df["pred_proba"].values.astype(float)

        auc = roc_auc_score(y_true, y_score)

        qaucs = []
        for name, grp in df.groupby("name"):
            if len(np.unique(grp["target"])) < 2:
                continue
            qaucs.append(roc_auc_score(grp["target"].values, grp["pred_proba"].values))
        qauc_mean = float(np.mean(qaucs)) if qaucs else float("nan")

        avg_proba = float(np.mean(y_score))

        thr = self.config.default_threshold
        y_pred_bin = (y_score > thr).astype(int)

        acc = accuracy_score(y_true, y_pred_bin)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_bin, average="binary", zero_division=0
        )

        jacc = sk_jaccard_score(y_true, y_pred_bin, average="binary", zero_division=0)

        # -------- point IoU из points_test.csv --------
        point_iou = float("nan")
        if os.path.exists(self.points_test_path):
            dfp = pd.read_csv(self.points_test_path, sep=";")
            r = 0.001

            ious = []
            for name, grp in dfp.groupby("name"):
                pred_xyz = grp.loc[grp["pred"] == 1, ["x", "y", "z"]].to_numpy()
                gt_xyz = grp.loc[grp["gt"] == 1, ["x", "y", "z"]].to_numpy()
                ious.append(iou_points_radius(pred_xyz, gt_xyz, r))

            point_iou = float(np.mean(ious)) if ious else float("nan")

        metrics = {
            "AUC": float(auc),
            "qAUC": float(qauc_mean),
            "avg_proba": avg_proba,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "jaccard_score": float(jacc),
            "IoU_points": float(point_iou),
            "threshold": float(thr)
        }

        if os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8") as f:
                old_metrics = json.load(f)
        else:
            old_metrics = {}
        old_metrics.update(metrics)

        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(old_metrics, f, ensure_ascii=False, indent=2)

        return metrics

    def clone(self,
              name: str,
              overwrite: bool = False,
              copy_files: Optional[list[str]] = None) -> "ModelTraining":

        if copy_files is None:
            copy_files = [os.path.basename(self.model_path), "config.json"]
            if self._model_type() in {"cnn3d", "tabnet", "pointnet2"}:
                copy_files.append(os.path.basename(self.model_meta_path))

        new_exp_dir = os.path.join(self.base_dir, name)

        if os.path.exists(new_exp_dir):
            if not overwrite:
                raise FileExistsError(
                    f"Эксперимент '{name}' уже существует: {new_exp_dir}. "
                    f"Поставь overwrite=True если нужно перезаписать."
                )
            shutil.rmtree(new_exp_dir)

        os.makedirs(new_exp_dir, exist_ok=True)

        mt2 = ModelTraining(base_dir=self.base_dir, name=name)

        mt2.config = ExperimentConfig(
            name=name,
            feature_names=list(self.config.feature_names),
            feature_names_dist=list(self.config.feature_names_dist),
            voxel_size=float(self.config.voxel_size),
            model_type=str(self.config.model_type),
            catboost_params=dict(self.config.catboost_params),
            cnn3d_params=dict(self.config.cnn3d_params),
            tabnet_params=dict(self.config.tabnet_params),
            pointnet2_params=dict(self.config.pointnet2_params),
            default_threshold=float(self.config.default_threshold),
            iou_radius=float(self.config.iou_radius),
        )
        mt2._save_config()

        src_dir = self.exp_dir
        for fname in copy_files:
            src = os.path.join(src_dir, fname)
            dst = os.path.join(new_exp_dir, fname)

            if fname == "config.json":
                continue

            if os.path.exists(src):
                shutil.copy2(src, dst)

        mt2.model = self.model
        mt2.model_cols = self.model_cols

        return mt2
