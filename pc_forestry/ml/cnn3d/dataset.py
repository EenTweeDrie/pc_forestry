from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


META_COLS_DEFAULT = ("x", "y", "z", "name", "target")


@dataclass(frozen=True)
class LabelMapping:
    """Маппинг исходных меток -> contiguous [0..K-1]."""

    raw_to_contiguous: Dict[int, int]
    contiguous_to_raw: Dict[int, int]

    @staticmethod
    def from_raw_labels(raw_labels: Sequence[int]) -> "LabelMapping":
        uniq = sorted({int(x) for x in raw_labels})
        raw_to = {raw: i for i, raw in enumerate(uniq)}
        to_raw = {i: raw for raw, i in raw_to.items()}
        return LabelMapping(raw_to_contiguous=raw_to, contiguous_to_raw=to_raw)


def _infer_feature_columns(df: pd.DataFrame, meta_cols: Sequence[str]) -> List[str]:
    meta_set = set(meta_cols)
    numeric_cols = [c for c in df.columns if c not in meta_set and pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols


def build_label_mapping_from_df(
    df: pd.DataFrame,
    label_col: str = "target",
    ignore_values: Sequence[int] = (2, -1),
) -> LabelMapping:
    labels = df[label_col].dropna().astype(int).values
    labels = [int(x) for x in labels if int(x) not in set(ignore_values)]
    if not labels:
        raise ValueError("В датасете не осталось валидных меток (всё попало в ignore_values).")
    return LabelMapping.from_raw_labels(labels)


class Features3DVolumeDataset(Dataset):
    """
    Датасет, который из таблицы вокселей собирает 3D объёмы по ключу `name`.

    Выход одного элемента:
    - x: FloatTensor (C, D, H, W)
    - y: LongTensor  (D, H, W) с ignore_index для пустых/паддинга
    - meta: dict с инфо (name, origin, shape, coords)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        feature_cols: Optional[Sequence[str]] = None,
        name_col: str = "name",
        x_col: str = "x",
        y_col: str = "y",
        z_col: str = "z",
        label_col: str = "target",
        label_mapping: Optional[LabelMapping] = None,
        ignore_index: int = -1,
        ignore_label_values: Sequence[int] = (2, -1),
    ) -> None:
        self.df = df.copy()
        self.name_col = name_col
        self.x_col = x_col
        self.y_col = y_col
        self.z_col = z_col
        self.label_col = label_col
        self.ignore_index = int(ignore_index)
        self.ignore_label_values = tuple(int(v) for v in ignore_label_values)

        # Нормализуем типы координат
        for c in (x_col, y_col, z_col):
            self.df[c] = self.df[c].astype(int)

        if feature_cols is None:
            feature_cols = _infer_feature_columns(self.df, meta_cols=(x_col, y_col, z_col, name_col, label_col))
        self.feature_cols = list(feature_cols)
        if not self.feature_cols:
            raise ValueError("Не удалось определить числовые feature_cols. Проверьте столбцы features.csv.")

        if label_mapping is None:
            label_mapping = build_label_mapping_from_df(self.df, label_col=label_col, ignore_values=ignore_label_values)
        self.label_mapping = label_mapping

        # Индексация по name
        if name_col not in self.df.columns:
            raise ValueError(f"В CSV отсутствует столбец '{name_col}'.")
        self.names: List[str] = sorted(self.df[name_col].astype(str).unique().tolist())

        # Готовим быстрый доступ: name -> view (индексы строк)
        self._groups: Dict[str, np.ndarray] = {}
        for nm, g in self.df.groupby(self.name_col, sort=False):
            self._groups[str(nm)] = g.index.values

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        rows = self.df.loc[self._groups[name]]

        xs = rows[self.x_col].to_numpy(dtype=np.int64, copy=False)
        ys = rows[self.y_col].to_numpy(dtype=np.int64, copy=False)
        zs = rows[self.z_col].to_numpy(dtype=np.int64, copy=False)

        # origin = min по каждой оси, чтобы получить компактный bounding box
        x0 = int(xs.min()) if xs.size else 0
        y0 = int(ys.min()) if ys.size else 0
        z0 = int(zs.min()) if zs.size else 0

        xi = (xs - x0).astype(np.int64, copy=False)
        yi = (ys - y0).astype(np.int64, copy=False)
        zi = (zs - z0).astype(np.int64, copy=False)

        W = int(xi.max() + 1) if xi.size else 1
        H = int(yi.max() + 1) if yi.size else 1
        D = int(zi.max() + 1) if zi.size else 1

        feats = rows[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
        C = int(feats.shape[1])

        vol = np.zeros((C, D, H, W), dtype=np.float32)
        # Заполняем признаки по координатам (C, D, H, W)
        # Если в одной ячейке несколько строк — усредняем (на всякий случай)
        # Реализация: аккумуляция sum + count
        acc = np.zeros((C, D, H, W), dtype=np.float32)
        cnt = np.zeros((1, D, H, W), dtype=np.float32)
        for i in range(len(rows)):
            zz, yy, xx = int(zi[i]), int(yi[i]), int(xi[i])
            acc[:, zz, yy, xx] += feats[i]
            cnt[0, zz, yy, xx] += 1.0
        mask = cnt > 0
        vol[:, mask[0]] = (acc[:, mask[0]] / cnt[0, mask[0]])

        # Метки
        y_raw = rows[self.label_col].to_numpy(dtype=np.int64, copy=False)
        y_grid = np.full((D, H, W), fill_value=self.ignore_index, dtype=np.int64)

        ignore_set = set(self.ignore_label_values)
        for i in range(len(rows)):
            raw = int(y_raw[i])
            if raw in ignore_set:
                continue
            zz, yy, xx = int(zi[i]), int(yi[i]), int(xi[i])
            y_grid[zz, yy, xx] = int(self.label_mapping.raw_to_contiguous.get(raw, self.ignore_index))

        x_t = torch.from_numpy(vol)  # (C, D, H, W)
        y_t = torch.from_numpy(y_grid)  # (D, H, W)

        meta = {
            "name": name,
            "origin": (x0, y0, z0),
            "shape": (D, H, W),
            "feature_cols": self.feature_cols,
            "contiguous_to_raw": self.label_mapping.contiguous_to_raw,
        }
        return x_t, y_t, meta


def pad_collate_3d(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, dict]],
    *,
    ignore_index: int = -1,
    multiple: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
    """
    Collate, который паддит объёмы до максимального D/H/W в батче,
    а также (по умолчанию) до ближайшего размера, кратного `multiple`.

    Это важно для 3D U-Net с несколькими pool/upsample: размеры D/H/W должны
    быть кратны \(2^{depth}\). В нашей модели depth=2 => multiple=4.

    - x: (B, C, D, H, W)
    - y: (B, D, H, W) с ignore_index в паддинге
    """
    xs, ys, metas = zip(*batch)
    C = int(xs[0].shape[0])
    maxD = max(int(x.shape[1]) for x in xs)
    maxH = max(int(x.shape[2]) for x in xs)
    maxW = max(int(x.shape[3]) for x in xs)

    multiple = max(1, int(multiple))
    if multiple > 1:
        def _ceil_to_mul(v: int) -> int:
            return int(((v + multiple - 1) // multiple) * multiple)

        maxD = _ceil_to_mul(maxD)
        maxH = _ceil_to_mul(maxH)
        maxW = _ceil_to_mul(maxW)

    xb = torch.zeros((len(xs), C, maxD, maxH, maxW), dtype=xs[0].dtype)
    yb = torch.full((len(xs), maxD, maxH, maxW), fill_value=int(ignore_index), dtype=ys[0].dtype)

    for i, (x, y) in enumerate(zip(xs, ys)):
        _, D, H, W = x.shape
        xb[i, :, :D, :H, :W] = x
        yb[i, :D, :H, :W] = y

    return xb, yb, list(metas)
