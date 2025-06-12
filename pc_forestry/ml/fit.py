# теперь у меня есть датасеты train.csv , val.csv, test.py

# нужно обучить классификатор
# используя все современные методы

# подбор гиперпараметров и так далее

# нужно проверить несколько классифиционный моделей

# CatboostClassifier, CatboostRanker, MLP и другие самые перспективные на твой выбор

# Сложность заключается в том, что ты должен делать test немного по другому

# В тестовых данных нет файла normalized_df. На тест будут подаваться непосредственно файлы

# например

# import numpy as np
# pc = TREE.read(os.path.join(folder_dir, 'tree_0001.pcd'))
# pc.shift_to_zero()
# pc.calculate_illuminance()
# pc.estimate_normals()
# pc.estimate_coordinate()
# vg = VOXELGRID.create(pc, 0.5, verbose=True)

# index = np.array([voxel.index for voxel in vg.voxels])
# max_layer = max([index[i][2] for i in range(len(index))])
# voxels_total = []
# for i in range(0, max_layer):
#     vg.calculate_distances_to_previous_layer_by_layer(pc.coordinate, layer = i)
#     voxels = vg.get_voxels_by_layer(layer = i)
#     voxels_total += voxels
#     vg_total = VOXELGRID(PC = None, voxel_size = vg.voxel_size, voxels = voxels_total)
#     vg_total.calculate_distances_to_coordinate(pc.coordinate)
#     # делается предикт только для одного слоя
#     # например все сделаю 0 (ствол)

# # вот здесь сделай предикт модели
#     for voxel in voxels:
#         voxel.label = 0


# потом посчитай auc, logloss, qauc (по разбивке по дереву)

import os
import argparse
import joblib
from typing import Dict, List

import pandas as pd
import numpy as np
from loguru import logger

from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from catboost import CatBoostClassifier
except ImportError as e:
    logger.error(
        "CatBoost не установлен. Пожалуйста, добавьте зависимость `catboost` в окружение.")
    raise e

# -----------------------------------------------------------------------------
# УТИЛИТАРНЫЕ ФУНКЦИИ
# -----------------------------------------------------------------------------


def qauc_by_group(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    """Среднее ROC-AUC, рассчитанное отдельно для каждой группы.

    Параметры
    ----------
    y_true : np.ndarray
        Истинные метки.
    y_pred : np.ndarray
        Предсказанные вероятности (для положительного класса).
    groups : np.ndarray
        Идентификатор дерева/файла для каждого примера.
    """
    aucs: List[float] = []
    unique_groups = np.unique(groups)
    for gid in unique_groups:
        mask = groups == gid
        # Если в группе присутствует только один класс, метрика не считается
        if len(np.unique(y_true[mask])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[mask], y_pred[mask]))
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def evaluate(model, X: pd.DataFrame, y: pd.Series, group_ids: pd.Series = None) -> Dict[str, float]:
    """Вычисляет AUC, лог-лосс и qAUC (если переданы группы)."""
    proba = model.predict_proba(X)[:, 1]
    metrics = {
        "auc": roc_auc_score(y, proba),
        "logloss": log_loss(y, proba),
    }
    if group_ids is not None:
        metrics["qauc"] = qauc_by_group(y.values, proba, group_ids.values)
    return metrics


# -----------------------------------------------------------------------------
# ОБУЧЕНИЕ МОДЕЛЕЙ
# -----------------------------------------------------------------------------


def train_catboost(X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series) -> CatBoostClassifier:
    """Обучение CatBoostClassifier с простой GridSearchCV по выбранным гиперпараметрам."""
    param_grid = {
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5],
        "iterations": [300, 500],
    }
    base_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        random_state=42,
        thread_count=os.cpu_count(),
    )

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    best_model: CatBoostClassifier = grid.best_estimator_

    logger.info(f"Лучшие параметры CatBoost: {grid.best_params_}")
    return best_model


def train_mlp(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Обучение MLPClassifier внутри Pipeline со StandardScaler."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(max_iter=300, random_state=42)),
    ])

    param_grid = {
        "mlp__hidden_layer_sizes": [(64,), (128, 64), (256, 128)],
        "mlp__alpha": [1e-4, 1e-3, 1e-2],
        "mlp__learning_rate_init": [1e-3, 5e-3],
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    logger.info(f"Лучшие параметры MLP: {grid.best_params_}")
    return grid.best_estimator_


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Простая модель RandomForest как baseline."""
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


MODEL_TRAINERS = {
    "catboost": train_catboost,
    "mlp": train_mlp,
    "rf": train_random_forest,
}

# -----------------------------------------------------------------------------
# PREDICT ДЛЯ ТЕСТА (РАБОТА С PCD ФАЙЛАМИ)
# -----------------------------------------------------------------------------

try:
    from pc_forestry.pcd.TREE import TREE
    from pc_forestry.pcd.VOXEL import VOXELGRID
except ImportError:
    # Скрипт может запускаться без установленных внутренних пакетов
    TREE = None  # type: ignore
    VOXELGRID = None  # type: ignore


# type: ignore
def extract_features_from_voxels(voxels: List["VOXELGRID".voxel]) -> pd.DataFrame:
    """Построение DataFrame признаков для списка вокселей."""
    vg_tmp = VOXELGRID(
        PC=None, voxel_size=voxels[0].voxel_size, voxels=voxels)  # type: ignore
    df = vg_tmp.normalized_df.drop(columns=["label"], errors="ignore")
    return df


def predict_for_tree(model, file_path: str, voxel_size: float = 0.5):
    """Пример инференса модели для одного дерева .pcd/.txt.

    Пошагово строится воксельная сетка, последовательно для каждого слоя
    вычисляются расстояния до предыдущего слоя, затем вызывается модель.
    Полученные предсказания сохраняются в атрибут `label` каждого вокселя.
    """
    if TREE is None or VOXELGRID is None:
        raise RuntimeError(
            "Модули TREE/VOXELGRID недоступны. Проверьте PYTHONPATH.")

    import numpy as np  # локальный импорт, чтобы избежать конфликтов при отсутствии numpy

    pc = TREE.read(file_path)
    pc.shift_to_zero()
    pc.calculate_illuminance()
    pc.estimate_normals()
    pc.estimate_coordinate()

    vg = VOXELGRID.create(pc, voxel_size, verbose=False)

    index = np.array([voxel.index for voxel in vg.voxels])
    max_layer = int(np.max(index[:, 2]))

    voxels_total: List = []

    for layer in range(max_layer + 1):
        vg.calculate_distances_to_previous_layer_by_layer(
            pc.coordinate, layer=layer)
        voxels_layer = vg.get_voxels_by_layer(layer=layer)
        voxels_total += voxels_layer

        vg_total = VOXELGRID(PC=None, voxel_size=vg.voxel_size,
                             voxels=voxels_total)  # type: ignore
        vg_total.calculate_distances_to_coordinate(pc.coordinate)

        # Подготовка признаков
        df_features = vg_total.normalized_df.drop(
            columns=["label"], errors="ignore")
        proba = model.predict_proba(df_features)[:, 1]

        # Присваиваем метку 1, если вероятность > 0.5 (порог можно настроить)
        for voxel, p in zip(voxels_layer, proba[-len(voxels_layer):]):
            voxel.label = int(p >= 0.5)

    # Возвращаем объект с обновлёнными метками
    return vg


# -----------------------------------------------------------------------------
# ОСНОВНАЯ ТОЧКА ВХОДА
# -----------------------------------------------------------------------------

def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("В датасете отсутствует столбец 'label'.")
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)
    groups = df["source_file"] if "source_file" in df.columns else None
    return X, y, groups


def main():
    parser = argparse.ArgumentParser(
        description="Обучение классификаторов для ForestryVoxel задач.")
    parser.add_argument("--train_csv", required=True,
                        type=str, help="Путь к train CSV")
    parser.add_argument("--val_csv", required=True,
                        type=str, help="Путь к val CSV")
    parser.add_argument("--models", nargs="*", default=["catboost", "mlp", "rf"],
                        choices=list(MODEL_TRAINERS.keys()), help="Список моделей для обучения")
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints", help="Каталог для сохранения моделей")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    X_train, y_train, _ = load_dataset(args.train_csv)
    X_val, y_val, groups_val = load_dataset(args.val_csv)

    summary: List[Dict[str, str]] = []

    for model_name in args.models:
        logger.info(f"Обучение модели: {model_name}")
        trainer = MODEL_TRAINERS[model_name]

        model = trainer(X_train, y_train, X_val,
                        y_val) if model_name == "catboost" else trainer(X_train, y_train)

        # Оценка на валидации
        metrics = evaluate(model, X_val, y_val, groups_val)
        logger.info(f"Метрики {model_name}: {metrics}")

        # Сохранение модели
        model_path = os.path.join(args.output_dir, f"{model_name}_model.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Модель сохранена в {model_path}")

        summary.append(
            {"model": model_name, **{k: f"{v:.4f}" for k, v in metrics.items()}})

    # Итоговая таблица
    summary_df = pd.DataFrame(summary)
    summary_csv = os.path.join(args.output_dir, "training_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"Сводка сохранена в {summary_csv}")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# ПРИМЕР ЗАПУСКА (из корня проекта):
#
#   python -m pc_forestry.ml.fit \
#       --train_csv data/train/train_dataset.csv \
#       --val_csv data/val/val_dataset.csv \
#       --output_dir pc_forestry/predict/checkpoints
# -----------------------------------------------------------------------------
