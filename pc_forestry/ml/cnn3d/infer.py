from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .model import UNet3DLight


def _infer_feature_columns_from_ckpt(ckpt: dict) -> List[str]:
    cols = ckpt.get("feature_cols")
    if not cols or not isinstance(cols, list):
        raise ValueError("В чекпойнте нет 'feature_cols'.")
    return [str(c) for c in cols]


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device | None = None) -> tuple[UNet3DLight, dict, torch.device]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    feature_cols = _infer_feature_columns_from_ckpt(ckpt)
    in_channels = int(ckpt["in_channels"])
    num_classes = int(ckpt["num_classes"])
    base_channels = int(ckpt.get("base_channels", 16))
    if in_channels != len(feature_cols):
        raise ValueError(f"Несоответствие каналов: ckpt.in_channels={in_channels}, feature_cols={len(feature_cols)}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3DLight(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt, device


def _apply_norm_inplace(df: pd.DataFrame, feature_cols: List[str], norm_stats: Dict[str, Dict[str, float]]) -> None:
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"В CSV отсутствует feature колонка '{c}', ожидаемая из чекпойнта.")
        m = float(norm_stats.get(c, {}).get("mean", 0.0))
        s = float(norm_stats.get(c, {}).get("std", 1.0))
        if not np.isfinite(s) or s < 1e-12:
            s = 1.0
        df[c] = (df[c].astype(np.float32) - m) / s
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)


def _build_volume_for_name(
    rows: pd.DataFrame,
    *,
    feature_cols: List[str],
) -> Tuple[torch.Tensor, Tuple[int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    xs = rows["x"].to_numpy(dtype=np.int64, copy=False)
    ys = rows["y"].to_numpy(dtype=np.int64, copy=False)
    zs = rows["z"].to_numpy(dtype=np.int64, copy=False)

    x0 = int(xs.min()) if xs.size else 0
    y0 = int(ys.min()) if ys.size else 0
    z0 = int(zs.min()) if zs.size else 0

    xi = (xs - x0).astype(np.int64, copy=False)
    yi = (ys - y0).astype(np.int64, copy=False)
    zi = (zs - z0).astype(np.int64, copy=False)

    W = int(xi.max() + 1) if xi.size else 1
    H = int(yi.max() + 1) if yi.size else 1
    D = int(zi.max() + 1) if zi.size else 1

    feats = rows[feature_cols].to_numpy(dtype=np.float32, copy=False)
    C = int(feats.shape[1])

    acc = np.zeros((C, D, H, W), dtype=np.float32)
    cnt = np.zeros((1, D, H, W), dtype=np.float32)
    for i in range(len(rows)):
        zz, yy, xx = int(zi[i]), int(yi[i]), int(xi[i])
        acc[:, zz, yy, xx] += feats[i]
        cnt[0, zz, yy, xx] += 1.0

    vol = np.zeros((C, D, H, W), dtype=np.float32)
    mask = cnt > 0
    vol[:, mask[0]] = (acc[:, mask[0]] / cnt[0, mask[0]])

    return torch.from_numpy(vol), (x0, y0, z0), (xi, yi, zi)


def _pad_volume_to_multiple(vol: torch.Tensor, multiple: int = 4) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """
    vol: (C, D, H, W) -> паддим справа до кратности multiple по D/H/W.
    Возвращает (padded_vol, (D,H,W) исходные размеры).
    """
    multiple = max(1, int(multiple))
    C, D, H, W = vol.shape
    if multiple == 1:
        return vol, (int(D), int(H), int(W))

    def _ceil(v: int) -> int:
        return int(((v + multiple - 1) // multiple) * multiple)

    Dp, Hp, Wp = _ceil(int(D)), _ceil(int(H)), _ceil(int(W))
    pd = Dp - int(D)
    ph = Hp - int(H)
    pw = Wp - int(W)
    if pd == 0 and ph == 0 and pw == 0:
        return vol, (int(D), int(H), int(W))

    # F.pad: (W_left, W_right, H_left, H_right, D_left, D_right)
    vol_p = F.pad(vol, (0, pw, 0, ph, 0, pd), mode="constant", value=0.0)
    return vol_p, (int(D), int(H), int(W))


@torch.no_grad()
def infer_to_csv(
    features_csv: str,
    checkpoint_path: str,
    out_csv: str,
    *,
    batch_names: int = 1,
) -> str:
    if not os.path.exists(features_csv):
        raise FileNotFoundError(features_csv)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    df = pd.read_csv(features_csv, sep=";")
    required = {"x", "y", "z", "name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В CSV не хватает столбцов: {sorted(missing)}")

    df["name"] = df["name"].astype(str)
    df["x"] = df["x"].astype(int)
    df["y"] = df["y"].astype(int)
    df["z"] = df["z"].astype(int)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    feature_cols = _infer_feature_columns_from_ckpt(ckpt)
    norm_stats = ckpt.get("norm_stats", {}) or {}
    ignore_index = int(ckpt.get("ignore_index", -1))

    label_mapping = ckpt.get("label_mapping", {}) or {}
    contiguous_to_raw = label_mapping.get("contiguous_to_raw", {}) or {}
    # ключи могут быть строками после JSON-совместимости
    contiguous_to_raw = {int(k): int(v) for k, v in contiguous_to_raw.items()}

    in_channels = int(ckpt["in_channels"])
    num_classes = int(ckpt["num_classes"])
    base_channels = int(ckpt.get("base_channels", 16))
    if in_channels != len(feature_cols):
        raise ValueError(f"Несоответствие каналов: ckpt.in_channels={in_channels}, feature_cols={len(feature_cols)}")

    _apply_norm_inplace(df, feature_cols, norm_stats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3DLight(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # Готовим колонки вероятностей
    raw_labels_sorted = [contiguous_to_raw[i] for i in range(num_classes)]
    proba_cols = [f"proba_{raw}" for raw in raw_labels_sorted]
    for c in proba_cols:
        if c not in df.columns:
            df[c] = 0.0
    if "pred" not in df.columns:
        df["pred"] = 0

    # Инференс по name
    names = sorted(df["name"].unique().tolist())
    for nm in tqdm(names, desc="cnn3d infer"):
        rows_idx = df.index[df["name"] == nm].to_numpy()
        rows = df.loc[rows_idx]

        vol, _, (xi, yi, zi) = _build_volume_for_name(rows, feature_cols=feature_cols)
        vol_p, (D0, H0, W0) = _pad_volume_to_multiple(vol, multiple=4)
        xb = vol_p.unsqueeze(0).to(device, non_blocking=True)  # (1, C, D, H, W)

        logits = model(xb)  # (1, K, D, H, W)
        probs = torch.softmax(logits, dim=1)[0]  # (K, Dp, Hp, Wp)
        probs = probs[:, :D0, :H0, :W0]  # кроп обратно к исходному объёму
        pred = torch.argmax(probs, dim=0)  # (D, H, W)

        # Берём предсказания только в координатах, которые есть в исходном CSV
        # Векторизуем через flatten
        _, D, H, W = probs.shape
        flat_idx = (torch.from_numpy(zi) * (H * W) + torch.from_numpy(yi) * W + torch.from_numpy(xi)).long()

        pred_flat = pred.reshape(-1)[flat_idx].cpu().numpy()
        pred_raw = np.array([contiguous_to_raw.get(int(c), int(c)) for c in pred_flat], dtype=np.int64)
        df.loc[rows_idx, "pred"] = pred_raw

        probs_flat = probs.reshape(num_classes, -1)[:, flat_idx].T.cpu().numpy()  # (N, K)
        for j, raw in enumerate(raw_labels_sorted):
            df.loc[rows_idx, f"proba_{raw}"] = probs_flat[:, j].astype(np.float32)

    df.to_csv(out_csv, index=False, sep=";")
    return out_csv


@torch.no_grad()
def infer_for_features_df(
    df_features: pd.DataFrame,
    checkpoint_path: str,
    *,
    name: str = "sample",
) -> pd.DataFrame:
    """
    Инференс для одного объекта (одного дерева), когда у вас уже есть df признаков из VOXELGRIDFEATURES.

    df_features должен содержать x,y,z + те feature_cols, что были при обучении.
    """
    df = df_features.copy()
    required = {"x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В df_features не хватает столбцов: {sorted(missing)}")
    if "name" not in df.columns:
        df["name"] = str(name)

    model, ckpt, device = load_model_from_checkpoint(checkpoint_path)
    feature_cols = _infer_feature_columns_from_ckpt(ckpt)
    norm_stats = ckpt.get("norm_stats", {}) or {}

    label_mapping = ckpt.get("label_mapping", {}) or {}
    contiguous_to_raw = label_mapping.get("contiguous_to_raw", {}) or {}
    contiguous_to_raw = {int(k): int(v) for k, v in contiguous_to_raw.items()}
    num_classes = int(ckpt["num_classes"])
    raw_labels_sorted = [contiguous_to_raw[i] for i in range(num_classes)]

    df["name"] = df["name"].astype(str)
    df["x"] = df["x"].astype(int)
    df["y"] = df["y"].astype(int)
    df["z"] = df["z"].astype(int)
    _apply_norm_inplace(df, feature_cols, norm_stats)

    # гарантируем колонки
    for raw in raw_labels_sorted:
        c = f"proba_{raw}"
        if c not in df.columns:
            df[c] = 0.0
    if "pred" not in df.columns:
        df["pred"] = 0

    # один name
    rows_idx = df.index.to_numpy()
    vol, _, (xi, yi, zi) = _build_volume_for_name(df, feature_cols=feature_cols)
    vol_p, (D0, H0, W0) = _pad_volume_to_multiple(vol, multiple=4)
    xb = vol_p.unsqueeze(0).to(device, non_blocking=True)

    logits = model(xb)
    probs = torch.softmax(logits, dim=1)[0]  # (K, Dp, Hp, Wp)
    probs = probs[:, :D0, :H0, :W0]
    pred = torch.argmax(probs, dim=0)  # (D,H,W)

    _, D, H, W = probs.shape
    flat_idx = (torch.from_numpy(zi) * (H * W) + torch.from_numpy(yi) * W + torch.from_numpy(xi)).long()

    pred_flat = pred.reshape(-1)[flat_idx].cpu().numpy()
    pred_raw = np.array([contiguous_to_raw.get(int(c), int(c)) for c in pred_flat], dtype=np.int64)
    df.loc[rows_idx, "pred"] = pred_raw

    probs_flat = probs.reshape(num_classes, -1)[:, flat_idx].T.cpu().numpy()  # (N, K)
    for j, raw in enumerate(raw_labels_sorted):
        df.loc[rows_idx, f"proba_{raw}"] = probs_flat[:, j].astype(np.float32)

    return df


@torch.no_grad()
def apply_predictions_to_voxelgridfeatures(
    grid: object,
    df_pred: pd.DataFrame,
    *,
    pred_col: str = "pred",
    proba_prefix: str = "proba_",
    proba_as_max: bool = True,
) -> object:
    """
    Записывает предсказания обратно в объекты вокселей внутри VOXELGRIDFEATURES.

    Предположение: строки df_pred идут в том же порядке, что и grid.voxels (как в grid.get_features_df()).
    """
    if not hasattr(grid, "voxels"):
        raise ValueError("grid не похож на VOXELGRIDFEATURES: нет атрибута voxels")
    if pred_col not in df_pred.columns:
        raise ValueError(f"В df_pred нет колонки '{pred_col}'")

    preds = df_pred[pred_col].to_numpy()
    if len(preds) != len(grid.voxels):
        # fallback: попробуем матчить по x/y/z
        if not all(c in df_pred.columns for c in ("x", "y", "z")):
            raise ValueError("Длина df_pred не совпала с числом вокселей, и нет x/y/z для матчинга.")
        idx_map = {(int(r.x), int(r.y), int(r.z)): i for i, r in df_pred[["x", "y", "z"]].iterrows()}
        for v in grid.voxels:
            x, y, z = map(int, getattr(v, "index"))
            i = idx_map.get((x, y, z))
            if i is None:
                continue
            v.label = int(df_pred.loc[i, pred_col])
        return grid

    proba_cols = [c for c in df_pred.columns if c.startswith(proba_prefix)]
    for i, v in enumerate(grid.voxels):
        v.label = int(preds[i])
        if proba_cols:
            if proba_as_max:
                v.proba = float(df_pred.loc[df_pred.index[i], proba_cols].max())
            else:
                # если это бинарная задача и есть proba_1 — можно выбрать её
                try:
                    v.proba = float(df_pred.loc[df_pred.index[i], proba_cols[0]])
                except Exception:
                    v.proba = None
    return grid


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Инференс 3D CNN по features.csv (добавляет pred/proba_*).")
    p.add_argument("--features_csv", required=True, help="Путь к features.csv (sep=';').")
    p.add_argument("--checkpoint", required=True, help="Путь к cnn3d_model.pt")
    p.add_argument("--out", default="features_pred.csv", help="Куда сохранить CSV с предсказаниями")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    out = infer_to_csv(args.features_csv, args.checkpoint, args.out)
    print(f"[cnn3d] saved: {out}")


if __name__ == "__main__":
    main()
