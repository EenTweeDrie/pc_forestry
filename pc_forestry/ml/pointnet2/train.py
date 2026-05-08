from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (
    LabelMapping,
    PointCloudBlockDataset,
    build_label_mapping_from_df,
    infer_feature_columns,
    pointnet2_collate,
    split_names,
)
from .model import PointNet2Segmenter


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_norm_stats(train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for c in feature_cols:
        s = train_df[c].astype(np.float64)
        mean = float(np.nanmean(s.values))
        std = float(np.nanstd(s.values))
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        if not np.isfinite(mean):
            mean = 0.0
        stats[c] = {"mean": mean, "std": std}
    return stats


def _apply_norm(df: pd.DataFrame, feature_cols: List[str], stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        m = stats[c]["mean"]
        s = stats[c]["std"]
        out[c] = (out[c].astype(np.float32) - m) / s
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _sanitize_tensor_inplace(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)


@torch.no_grad()
def _eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    ignore_index: int,
) -> Tuple[float, float]:
    model.eval()
    ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    total_loss = 0.0
    total_correct = 0
    total_valid = 0

    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        xb = _sanitize_tensor_inplace(xb)

        logits = model(xb)
        logits = _sanitize_tensor_inplace(logits)
        valid = (yb != ignore_index)
        if int(valid.sum().item()) == 0:
            continue
        loss = ce(logits, yb)
        if not torch.isfinite(loss):
            continue
        total_loss += float(loss.item()) * xb.shape[0]

        pred = torch.argmax(logits, dim=1)
        total_correct += int((pred[valid] == yb[valid]).sum().item())
        total_valid += int(valid.sum().item())

    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = (total_correct / max(1, total_valid)) if total_valid else 0.0
    return avg_loss, acc


def train_from_df(
    df: pd.DataFrame,
    out_path: str,
    *,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    num_points: int = 1024,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    ignore_index: int = -1,
    ignore_label_values: Tuple[int, ...] = (2, -1),
    samples_per_epoch: int = 128,
    val_samples_per_epoch: int | None = None,
    amp: bool = False,
    sa1_npoint: int = 256,
    sa2_npoint: int = 64,
    sa3_npoint: int = 16,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    _set_seed(seed)

    required = {"x", "y", "z", "name", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В DataFrame не хватает столбцов: {sorted(missing)}")

    df_local = df.copy()
    df_local["name"] = df_local["name"].astype(str)
    df_local["target"] = df_local["target"].astype(int)
    if "point_index" not in df_local.columns:
        df_local["point_index"] = np.arange(len(df_local), dtype=np.int64)

    feature_cols = infer_feature_columns(df_local)
    names = sorted(df_local["name"].unique().tolist())
    train_names, val_names = split_names(names, val_ratio=val_ratio, seed=seed)

    train_df = df_local[df_local["name"].isin(train_names)].reset_index(drop=True)
    val_df = df_local[df_local["name"].isin(val_names)].reset_index(drop=True)

    norm_stats = _compute_norm_stats(train_df, feature_cols)
    train_df_n = _apply_norm(train_df, feature_cols, norm_stats)
    val_df_n = _apply_norm(val_df, feature_cols, norm_stats) if len(val_df) else val_df.copy()

    label_mapping: LabelMapping = build_label_mapping_from_df(
        train_df_n,
        label_col="target",
        ignore_values=ignore_label_values,
    )
    num_classes = len(label_mapping.contiguous_to_raw)

    if val_samples_per_epoch is None:
        val_samples_per_epoch = max(1, len(val_names)) if val_names else 1

    ds_train = PointCloudBlockDataset(
        train_df_n,
        feature_cols=feature_cols,
        label_mapping=label_mapping,
        ignore_index=ignore_index,
        ignore_label_values=ignore_label_values,
        num_points=num_points,
        samples_per_epoch=samples_per_epoch,
        seed=seed,
    )
    ds_val = PointCloudBlockDataset(
        val_df_n if len(val_df_n) else train_df_n.iloc[: min(len(train_df_n), num_points)].copy(),
        feature_cols=feature_cols,
        label_mapping=label_mapping,
        ignore_index=ignore_index,
        ignore_label_values=ignore_label_values,
        num_points=num_points,
        samples_per_epoch=val_samples_per_epoch,
        seed=seed + 10000,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pointnet2_collate,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=max(1, min(batch_size, 4)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pointnet2_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNet2Segmenter(
        in_channels=3 + len(feature_cols),
        num_classes=num_classes,
        sa1_npoint=sa1_npoint,
        sa2_npoint=sa2_npoint,
        sa3_npoint=sa3_npoint,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=bool(amp and device.type == "cuda"))
    ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        num_train_batches = 0
        pbar = tqdm(dl_train, desc=f"PointNet++ epoch {epoch}/{epochs}", leave=False)
        for xb, yb, _ in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            xb = _sanitize_tensor_inplace(xb)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
                logits = model(xb)
                logits = _sanitize_tensor_inplace(logits)
                valid = (yb != ignore_index)
                if int(valid.sum().item()) == 0:
                    continue
                loss = ce(logits, yb)
            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            running += float(loss.item()) * xb.shape[0]
            num_train_batches += xb.shape[0]
            pbar.set_postfix(loss=float(loss.item()))

        train_loss = running / max(1, num_train_batches)
        val_loss, val_acc = _eval_epoch(model, dl_val, device=device, ignore_index=ignore_index)
        print(f"[pointnet2] epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "model_state": model.state_dict(),
                "in_channels": 3 + len(feature_cols),
                "num_classes": num_classes,
                "feature_cols": feature_cols,
                "num_points": int(num_points),
                "ignore_index": int(ignore_index),
                "ignore_label_values": list(ignore_label_values),
                "label_mapping": {
                    "raw_to_contiguous": label_mapping.raw_to_contiguous,
                    "contiguous_to_raw": label_mapping.contiguous_to_raw,
                },
                "norm_stats": norm_stats,
                "train_names": train_names,
                "val_names": val_names,
                "model_kwargs": {
                    "sa1_npoint": int(sa1_npoint),
                    "sa2_npoint": int(sa2_npoint),
                    "sa3_npoint": int(sa3_npoint),
                },
            }
            torch.save(ckpt, out_path)

    meta_path = os.path.splitext(out_path)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": out_path,
                "feature_cols": feature_cols,
                "num_classes": num_classes,
                "num_points": int(num_points),
                "ignore_index": int(ignore_index),
                "label_mapping": label_mapping.raw_to_contiguous,
                "norm_stats": norm_stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return out_path


def train_from_features_csv(
    features_csv: str,
    out_path: str,
    **kwargs,
) -> str:
    if not os.path.exists(features_csv):
        raise FileNotFoundError(features_csv)
    df = pd.read_csv(features_csv, sep=";")
    return train_from_df(df, out_path, **kwargs)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Обучение PointNet++ для point-wise сегментации.")
    p.add_argument("--features_csv", required=True, help="Путь к point features CSV (sep=';').")
    p.add_argument("--out", default="checkpoints/pointnet2_model.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--samples_per_epoch", type=int, default=128)
    p.add_argument("--no_amp", action="store_true")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    out = train_from_features_csv(
        args.features_csv,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_points=args.num_points,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        samples_per_epoch=args.samples_per_epoch,
        amp=not args.no_amp,
    )
    print(f"[pointnet2] saved: {out}")


if __name__ == "__main__":
    main()
