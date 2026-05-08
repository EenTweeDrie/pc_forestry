from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .dataset import normalize_xyz
from .model import PointNet2Segmenter


def _sanitize_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)


def _infer_feature_columns_from_ckpt(ckpt: dict) -> List[str]:
    cols = ckpt.get("feature_cols")
    if cols is None or not isinstance(cols, list):
        raise ValueError("В чекпойнте нет 'feature_cols'.")
    return [str(c) for c in cols]


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device | None = None,
) -> tuple[PointNet2Segmenter, dict, torch.device]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    feature_cols = _infer_feature_columns_from_ckpt(ckpt)
    in_channels = int(ckpt["in_channels"])
    num_classes = int(ckpt["num_classes"])
    model_kwargs = ckpt.get("model_kwargs", {}) or {}
    if in_channels != 3 + len(feature_cols):
        raise ValueError(
            f"Несоответствие каналов: ckpt.in_channels={in_channels}, feature_cols={len(feature_cols)}"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNet2Segmenter(
        in_channels=in_channels,
        num_classes=num_classes,
        **model_kwargs,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt, device


def _apply_norm_inplace(df: pd.DataFrame, feature_cols: List[str], norm_stats: Dict[str, Dict[str, float]]) -> None:
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"В таблице отсутствует feature колонка '{c}', ожидаемая из чекпойнта.")
        m = float(norm_stats.get(c, {}).get("mean", 0.0))
        s = float(norm_stats.get(c, {}).get("std", 1.0))
        if not np.isfinite(s) or s < 1e-12:
            s = 1.0
        df[c] = (df[c].astype(np.float32) - m) / s
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)


def _pad_sample_indices(indices: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == num_points:
        return indices
    if len(indices) > num_points:
        return indices[:num_points]
    extra = rng.choice(indices, size=num_points - len(indices), replace=True)
    return np.concatenate([indices, extra], axis=0)


def _build_inference_chunks(
    num_rows: int,
    num_points: int,
    num_votes: int,
    seed: int,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    chunks: List[np.ndarray] = []

    for _ in range(max(1, int(num_votes))):
        perm = rng.permutation(num_rows)
        if num_rows <= num_points:
            chunks.append(_pad_sample_indices(perm, num_points, rng))
            continue
        for start in range(0, num_rows, num_points):
            part = perm[start:start + num_points]
            chunks.append(_pad_sample_indices(part, num_points, rng))

    return chunks


@torch.no_grad()
def infer_for_points_df(
    df_features: pd.DataFrame,
    checkpoint_path: str,
    *,
    name: str = "sample",
    num_votes: int = 3,
) -> pd.DataFrame:
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
    num_points = int(ckpt.get("num_points", 1024))

    label_mapping = ckpt.get("label_mapping", {}) or {}
    contiguous_to_raw = label_mapping.get("contiguous_to_raw", {}) or {}
    contiguous_to_raw = {int(k): int(v) for k, v in contiguous_to_raw.items()}
    num_classes = int(ckpt["num_classes"])
    raw_labels_sorted = [contiguous_to_raw[i] for i in range(num_classes)]

    df["name"] = df["name"].astype(str)
    _apply_norm_inplace(df, feature_cols, norm_stats)

    for raw in raw_labels_sorted:
        c = f"proba_{raw}"
        if c not in df.columns:
            df[c] = 0.0
    if "pred" not in df.columns:
        df["pred"] = 0

    for nm in tqdm(sorted(df["name"].unique().tolist()), desc="pointnet2 infer"):
        rows_idx = df.index[df["name"] == nm].to_numpy()
        rows = df.loc[rows_idx]

        xyz = rows[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=False)
        xyz_norm, _, _ = normalize_xyz(xyz)
        feats = rows[feature_cols].to_numpy(dtype=np.float32, copy=False)

        accum = np.zeros((len(rows), num_classes), dtype=np.float32)
        counts = np.zeros(len(rows), dtype=np.float32)
        chunks = _build_inference_chunks(len(rows), num_points=num_points, num_votes=num_votes, seed=42)

        for chunk_idx in chunks:
            x_chunk = np.concatenate([xyz_norm[chunk_idx], feats[chunk_idx]], axis=1).T.astype(np.float32, copy=False)
            xb = torch.from_numpy(x_chunk).unsqueeze(0).to(device, non_blocking=True)
            xb = _sanitize_tensor(xb)

            logits = model(xb)
            logits = _sanitize_tensor(logits)
            probs = torch.softmax(logits, dim=1)[0].permute(1, 0).cpu().numpy()

            accum[chunk_idx] += probs
            counts[chunk_idx] += 1.0

        uncovered = np.where(counts <= 0)[0]
        if uncovered.size:
            accum[uncovered] = 0.0
            counts[uncovered] = 1.0

        probs_mean = accum / counts[:, None]
        pred_contig = probs_mean.argmax(axis=1)
        pred_raw = np.array([contiguous_to_raw.get(int(c), int(c)) for c in pred_contig], dtype=np.int64)

        df.loc[rows_idx, "pred"] = pred_raw
        for j, raw in enumerate(raw_labels_sorted):
            df.loc[rows_idx, f"proba_{raw}"] = probs_mean[:, j].astype(np.float32)

    return df


@torch.no_grad()
def infer_to_csv(
    features_csv: str,
    checkpoint_path: str,
    out_csv: str,
    *,
    num_votes: int = 3,
) -> str:
    if not os.path.exists(features_csv):
        raise FileNotFoundError(features_csv)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    df = pd.read_csv(features_csv, sep=";")
    out = infer_for_points_df(df, checkpoint_path, num_votes=num_votes)
    out.to_csv(out_csv, index=False, sep=";")
    return out_csv


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Инференс PointNet++ по point features CSV.")
    p.add_argument("--features_csv", required=True, help="Путь к point features CSV (sep=';').")
    p.add_argument("--checkpoint", required=True, help="Путь к pointnet2_model.pt")
    p.add_argument("--out", default="point_features_pred.csv", help="Куда сохранить CSV с предсказаниями")
    p.add_argument("--num_votes", type=int, default=3)
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    out = infer_to_csv(args.features_csv, args.checkpoint, args.out, num_votes=args.num_votes)
    print(f"[pointnet2] saved: {out}")


if __name__ == "__main__":
    main()
