from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


META_COLS_DEFAULT = ("x", "y", "z", "name", "target", "point_index")


@dataclass(frozen=True)
class LabelMapping:
    raw_to_contiguous: Dict[int, int]
    contiguous_to_raw: Dict[int, int]

    @staticmethod
    def from_raw_labels(raw_labels: Sequence[int]) -> "LabelMapping":
        uniq = sorted({int(x) for x in raw_labels})
        raw_to = {raw: i for i, raw in enumerate(uniq)}
        to_raw = {i: raw for raw, i in raw_to.items()}
        return LabelMapping(raw_to_contiguous=raw_to, contiguous_to_raw=to_raw)


def infer_feature_columns(
    df: pd.DataFrame,
    meta_cols: Sequence[str] = META_COLS_DEFAULT,
) -> List[str]:
    meta_set = set(meta_cols)
    return [c for c in df.columns if c not in meta_set and pd.api.types.is_numeric_dtype(df[c])]


def build_label_mapping_from_df(
    df: pd.DataFrame,
    label_col: str = "target",
    ignore_values: Sequence[int] = (2, -1),
) -> LabelMapping:
    labels = df[label_col].dropna().astype(int).values
    labels = [int(x) for x in labels if int(x) not in set(ignore_values)]
    if not labels:
        raise ValueError("В датасете не осталось валидных меток после ignore_values.")
    return LabelMapping.from_raw_labels(labels)


def split_names(names: Sequence[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(seed)
    names_local = list(names)
    rng.shuffle(names_local)
    n_val = max(1, int(round(len(names_local) * float(val_ratio))))
    val_names = names_local[:n_val]
    train_names = names_local[n_val:]
    if not train_names:
        train_names, val_names = val_names, train_names
    return train_names, val_names


def normalize_xyz(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    xyz = np.asarray(xyz, dtype=np.float32)
    centroid = xyz.mean(axis=0, keepdims=True) if len(xyz) else np.zeros((1, 3), dtype=np.float32)
    centered = xyz - centroid
    scale = float(np.sqrt((centered ** 2).sum(axis=1)).max()) if len(centered) else 1.0
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return centered / scale, centroid[0], scale


def sample_point_indices(
    num_rows: int,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if num_rows <= 0:
        raise ValueError("Нельзя сэмплировать из пустого облака.")
    if num_rows >= num_points:
        return rng.choice(num_rows, size=num_points, replace=False)

    base = np.arange(num_rows, dtype=np.int64)
    extra = rng.choice(num_rows, size=num_points - num_rows, replace=True)
    out = np.concatenate([base, extra], axis=0)
    rng.shuffle(out)
    return out


class PointCloudBlockDataset(Dataset):
    """
    Датасет для point-wise сегментации PointNet++.

    Каждый элемент — это семпл `num_points` точек из одного дерева:
    - x: (C, N), где первые 3 канала — нормализованные xyz
    - y: (N,)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        feature_cols: Optional[Sequence[str]] = None,
        name_col: str = "name",
        label_col: str = "target",
        point_index_col: str = "point_index",
        num_points: int = 1024,
        label_mapping: Optional[LabelMapping] = None,
        ignore_index: int = -1,
        ignore_label_values: Sequence[int] = (2, -1),
        samples_per_epoch: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.df = df.copy()
        self.name_col = name_col
        self.label_col = label_col
        self.point_index_col = point_index_col
        self.num_points = int(num_points)
        self.ignore_index = int(ignore_index)
        self.ignore_label_values = tuple(int(v) for v in ignore_label_values)
        self.seed = int(seed)

        self.df["x"] = self.df["x"].astype(np.float32)
        self.df["y"] = self.df["y"].astype(np.float32)
        self.df["z"] = self.df["z"].astype(np.float32)
        self.df[self.name_col] = self.df[self.name_col].astype(str)
        self.df[self.label_col] = self.df[self.label_col].astype(int)
        if self.point_index_col not in self.df.columns:
            self.df[self.point_index_col] = np.arange(len(self.df), dtype=np.int64)

        if feature_cols is None:
            feature_cols = infer_feature_columns(self.df)
        self.feature_cols = list(feature_cols)

        if label_mapping is None:
            label_mapping = build_label_mapping_from_df(
                self.df,
                label_col=self.label_col,
                ignore_values=self.ignore_label_values,
            )
        self.label_mapping = label_mapping

        self.names: List[str] = sorted(self.df[self.name_col].unique().tolist())
        if not self.names:
            raise ValueError("В датасете нет ни одного объекта.")

        self._clouds: Dict[str, Dict[str, np.ndarray]] = {}
        for name, grp in self.df.groupby(self.name_col, sort=False):
            xyz = grp[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=False)
            xyz_norm, centroid, scale = normalize_xyz(xyz)
            feats = grp[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
            labels_raw = grp[self.label_col].to_numpy(dtype=np.int64, copy=False)
            point_index = grp[self.point_index_col].to_numpy(dtype=np.int64, copy=False)

            labels = np.full(len(labels_raw), fill_value=self.ignore_index, dtype=np.int64)
            ignore_set = set(self.ignore_label_values)
            for i, raw in enumerate(labels_raw):
                raw_int = int(raw)
                if raw_int in ignore_set:
                    continue
                labels[i] = int(self.label_mapping.raw_to_contiguous.get(raw_int, self.ignore_index))

            self._clouds[str(name)] = {
                "xyz": xyz_norm.astype(np.float32, copy=False),
                "features": feats.astype(np.float32, copy=False),
                "labels": labels,
                "point_index": point_index,
                "centroid": np.asarray(centroid, dtype=np.float32),
                "scale": np.asarray(scale, dtype=np.float32),
            }

        self.samples_per_epoch = int(samples_per_epoch) if samples_per_epoch is not None else len(self.names)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        name = self.names[idx % len(self.names)]
        cloud = self._clouds[name]

        rng = np.random.default_rng(self.seed + int(idx))
        sel = sample_point_indices(len(cloud["xyz"]), self.num_points, rng)

        xyz = cloud["xyz"][sel]
        feats = cloud["features"][sel]
        labels = cloud["labels"][sel]
        point_index = cloud["point_index"][sel]

        x = np.concatenate([xyz, feats], axis=1).T.astype(np.float32, copy=False)
        y = labels.astype(np.int64, copy=False)

        meta = {
            "name": name,
            "point_index": point_index,
        }
        return torch.from_numpy(x), torch.from_numpy(y), meta


def pointnet2_collate(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, dict]],
) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
    xs, ys, metas = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0), list(metas)
