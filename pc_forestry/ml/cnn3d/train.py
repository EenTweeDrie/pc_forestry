from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import Features3DVolumeDataset, LabelMapping, build_label_mapping_from_df, pad_collate_3d
from .model import UNet3DLight


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_names(names: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(seed)
    names = list(names)
    rng.shuffle(names)
    n_val = max(1, int(round(len(names) * float(val_ratio))))
    val_names = names[:n_val]
    train_names = names[n_val:]
    if not train_names:
        train_names, val_names = val_names, train_names
    return train_names, val_names


def _infer_feature_columns(df: pd.DataFrame) -> List[str]:
    meta = {"x", "y", "z", "name", "target"}
    cols = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        raise ValueError("Не найдено числовых признаков (кроме x/y/z/name/target).")
    return cols


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


def _align_logits_and_target(
    logits: torch.Tensor,  # (B, K, D, H, W)
    target: torch.Tensor,  # (B, D, H, W)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Страховка на случай несовпадения размеров после pool/upsample при нечётных D/H/W.
    Центр-кроп до общих минимальных размеров.
    """
    if logits.dim() != 5 or target.dim() != 4:
        return logits, target
    B, K, D1, H1, W1 = logits.shape
    b2, D2, H2, W2 = target.shape
    if B != b2:
        return logits, target
    D = min(int(D1), int(D2))
    H = min(int(H1), int(H2))
    W = min(int(W1), int(W2))
    if (D1, H1, W1) == (D2, H2, W2):
        return logits, target

    sd1 = max((int(D1) - D) // 2, 0)
    sh1 = max((int(H1) - H) // 2, 0)
    sw1 = max((int(W1) - W) // 2, 0)
    sd2 = max((int(D2) - D) // 2, 0)
    sh2 = max((int(H2) - H) // 2, 0)
    sw2 = max((int(W2) - W) // 2, 0)

    logits_c = logits[:, :, sd1:sd1 + D, sh1:sh1 + H, sw1:sw1 + W]
    target_c = target[:, sd2:sd2 + D, sh2:sh2 + H, sw2:sw2 + W]
    return logits_c, target_c


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
        logits = model(xb)
        logits, yb = _align_logits_and_target(logits, yb)
        loss = ce(logits, yb)
        total_loss += float(loss.item()) * xb.shape[0]

        pred = torch.argmax(logits, dim=1)  # (B, D, H, W)
        valid = (yb != ignore_index)
        total_correct += int((pred[valid] == yb[valid]).sum().item())
        total_valid += int(valid.sum().item())

    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = (total_correct / max(1, total_valid)) if total_valid else 0.0
    return avg_loss, acc


def train_from_features_csv(
    features_csv: str,
    out_path: str,
    *,
    epochs: int = 30,
    batch_size: int = 2,
    lr: float = 2e-3,
    base_channels: int = 16,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    ignore_index: int = -1,
    ignore_label_values: Tuple[int, ...] = (2, -1),
    amp: bool = True,
) -> str:
    if not os.path.exists(features_csv):
        raise FileNotFoundError(features_csv)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    _set_seed(seed)

    df = pd.read_csv(features_csv, sep=";")
    required = {"x", "y", "z", "name", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В CSV не хватает столбцов: {sorted(missing)}")

    df["name"] = df["name"].astype(str)
    df["target"] = df["target"].astype(int)

    feature_cols = _infer_feature_columns(df)

    names = sorted(df["name"].unique().tolist())
    train_names, val_names = _split_names(names, val_ratio=val_ratio, seed=seed)

    train_df = df[df["name"].isin(train_names)].reset_index(drop=True)
    val_df = df[df["name"].isin(val_names)].reset_index(drop=True)

    # Нормализация по train, сохраняем в чекпойнт
    norm_stats = _compute_norm_stats(train_df, feature_cols)
    train_df_n = _apply_norm(train_df, feature_cols, norm_stats)
    val_df_n = _apply_norm(val_df, feature_cols, norm_stats)

    # Маппинг меток (из train)
    label_mapping: LabelMapping = build_label_mapping_from_df(
        train_df_n, label_col="target", ignore_values=ignore_label_values
    )
    num_classes = len(label_mapping.contiguous_to_raw)

    ds_train = Features3DVolumeDataset(
        train_df_n,
        feature_cols=feature_cols,
        label_mapping=label_mapping,
        ignore_index=ignore_index,
        ignore_label_values=ignore_label_values,
    )
    ds_val = Features3DVolumeDataset(
        val_df_n,
        feature_cols=feature_cols,
        label_mapping=label_mapping,
        ignore_index=ignore_index,
        ignore_label_values=ignore_label_values,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda b: pad_collate_3d(b, ignore_index=ignore_index),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda b: pad_collate_3d(b, ignore_index=ignore_index),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3DLight(in_channels=len(feature_cols), num_classes=num_classes, base_channels=base_channels).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=bool(amp and device.type == "cuda"))
    ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{epochs}", leave=False)
        running = 0.0
        for xb, yb, _ in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
                logits = model(xb)
                logits, yb = _align_logits_and_target(logits, yb)
                loss = ce(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.item()) * xb.shape[0]
            pbar.set_postfix(loss=float(loss.item()))

        train_loss = running / max(1, len(dl_train.dataset))
        val_loss, val_acc = _eval_epoch(model, dl_val, device=device, ignore_index=ignore_index)

        print(f"[cnn3d] epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "model_state": model.state_dict(),
                "in_channels": len(feature_cols),
                "num_classes": num_classes,
                "base_channels": base_channels,
                "feature_cols": feature_cols,
                "ignore_index": ignore_index,
                "ignore_label_values": list(ignore_label_values),
                "label_mapping": {
                    "raw_to_contiguous": label_mapping.raw_to_contiguous,
                    "contiguous_to_raw": label_mapping.contiguous_to_raw,
                },
                "norm_stats": norm_stats,
                "train_names": train_names,
                "val_names": val_names,
            }
            torch.save(ckpt, out_path)

    # Параллельно сохраняем читаемое мета-описание
    meta_path = os.path.splitext(out_path)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": out_path,
                "feature_cols": feature_cols,
                "num_classes": num_classes,
                "ignore_index": ignore_index,
                "label_mapping": label_mapping.raw_to_contiguous,
                "norm_stats": norm_stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return out_path


def train_from_df(
    df: pd.DataFrame,
    out_path: str,
    *,
    epochs: int = 30,
    batch_size: int = 2,
    lr: float = 2e-3,
    base_channels: int = 16,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    ignore_index: int = -1,
    ignore_label_values: Tuple[int, ...] = (2, -1),
    amp: bool = True,
) -> str:
    """
    То же самое, что train_from_features_csv, но принимает уже загруженный DataFrame.

    df должен содержать: x,y,z,name,target и числовые признаки.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    _set_seed(seed)

    required = {"x", "y", "z", "name", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В DataFrame не хватает столбцов: {sorted(missing)}")

    df_local = df.copy()
    df_local["name"] = df_local["name"].astype(str)
    df_local["target"] = df_local["target"].astype(int)

    feature_cols = _infer_feature_columns(df_local)
    names = sorted(df_local["name"].unique().tolist())
    train_names, val_names = _split_names(names, val_ratio=val_ratio, seed=seed)

    train_df = df_local[df_local["name"].isin(train_names)].reset_index(drop=True)
    val_df = df_local[df_local["name"].isin(val_names)].reset_index(drop=True)

    norm_stats = _compute_norm_stats(train_df, feature_cols)
    train_df_n = _apply_norm(train_df, feature_cols, norm_stats)
    val_df_n = _apply_norm(val_df, feature_cols, norm_stats)

    label_mapping: LabelMapping = build_label_mapping_from_df(
        train_df_n, label_col="target", ignore_values=ignore_label_values
    )
    num_classes = len(label_mapping.contiguous_to_raw)

    ds_train = Features3DVolumeDataset(
        train_df_n,
        feature_cols=feature_cols,
        label_mapping=label_mapping,
        ignore_index=ignore_index,
        ignore_label_values=ignore_label_values,
    )
    ds_val = Features3DVolumeDataset(
        val_df_n,
        feature_cols=feature_cols,
        label_mapping=label_mapping,
        ignore_index=ignore_index,
        ignore_label_values=ignore_label_values,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda b: pad_collate_3d(b, ignore_index=ignore_index),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda b: pad_collate_3d(b, ignore_index=ignore_index),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3DLight(in_channels=len(feature_cols), num_classes=num_classes, base_channels=base_channels).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=bool(amp and device.type == "cuda"))
    ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{epochs}", leave=False)
        running = 0.0
        for xb, yb, _ in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
                logits = model(xb)
                logits, yb = _align_logits_and_target(logits, yb)
                loss = ce(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.item()) * xb.shape[0]
            pbar.set_postfix(loss=float(loss.item()))

        train_loss = running / max(1, len(dl_train.dataset))
        val_loss, val_acc = _eval_epoch(model, dl_val, device=device, ignore_index=ignore_index)
        print(f"[cnn3d] epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "model_state": model.state_dict(),
                "in_channels": len(feature_cols),
                "num_classes": num_classes,
                "base_channels": base_channels,
                "feature_cols": feature_cols,
                "ignore_index": ignore_index,
                "ignore_label_values": list(ignore_label_values),
                "label_mapping": {
                    "raw_to_contiguous": label_mapping.raw_to_contiguous,
                    "contiguous_to_raw": label_mapping.contiguous_to_raw,
                },
                "norm_stats": norm_stats,
                "train_names": train_names,
                "val_names": val_names,
            }
            torch.save(ckpt, out_path)

    meta_path = os.path.splitext(out_path)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": out_path,
                "feature_cols": feature_cols,
                "num_classes": num_classes,
                "ignore_index": ignore_index,
                "label_mapping": label_mapping.raw_to_contiguous,
                "norm_stats": norm_stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return out_path


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Обучение 3D CNN по features.csv (воксельная сегментация).")
    p.add_argument("--features_csv", required=True, help="Путь к features.csv (sep=';').")
    p.add_argument("--out", default="checkpoints/cnn3d_model.pt", help="Куда сохранить чекпойнт .pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--base_channels", type=int, default=16)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
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
        base_channels=args.base_channels,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        amp=not args.no_amp,
    )
    print(f"[cnn3d] saved: {out}")


if __name__ == "__main__":
    main()
