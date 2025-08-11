import os
from typing import List, Dict, Optional
import pandas as pd
import joblib
from loguru import logger
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:
    TabNetClassifier = None

from .ml_validator import MLValidator

# Константа для количества ядер CPU
CPU_CORES = 4


def train_catboost(X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, use_gridsearch: bool = True) -> CatBoostClassifier:
    """Обучение CatBoostClassifier с опциональной GridSearchCV по выбранным гиперпараметрам."""
    base_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        verbose=False,
        random_state=42,
        task_type="GPU",  # Используем GPU для ускорения
        devices="0",      # Используем первую GPU
    )

    from catboost import Pool, cv

    # Задаем параметры для модели. Большое количество итераций компенсируется ранней остановкой.
    params = {
        'iterations': 2000,
        'learning_rate': 0.1,
        'depth': 6,
        'l2_leaf_reg': 3,
        'early_stopping_rounds': 50,
    }
    base_model.set_params(**params)

    if X_val is not None and y_val is not None:
        # Если есть валидационный набор, используем его для ранней остановки.
        logger.info("Обучение CatBoost с ранней остановкой на предоставленном валидационном наборе.")
        base_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=100,  # Печатать прогресс каждые 100 итераций
            use_best_model=True  # Вернуть модель с лучшей итерации
        )
        logger.info(f"Модель обучена. Лучшая итерация: {base_model.get_best_iteration()}")
        return base_model
    else:
        # Если валидационного набора нет, используем K-fold CV для подбора оптимального числа итераций.
        logger.info("Валидационный набор не предоставлен. Используем 3-fold CV для определения оптимального количества итераций.")

        cv_params = base_model.get_params()
        train_pool = Pool(data=X_train, label=y_train)

        # Запускаем кросс-валидацию для поиска лучшего числа итераций
        cv_results = cv(
            pool=train_pool,
            params=cv_params,
            fold_count=3,
            shuffle=True,
            stratified=True,  # Важно для сбалансированности классов в фолдах
            verbose=False,
        )

        # Находим оптимальное количество итераций по результатам CV
        best_iteration_count = cv_results['test-Logloss-mean'].idxmin() + 1
        logger.info(f"Оптимальное количество итераций по CV: {best_iteration_count}")

        # Переобучаем модель на всех тренировочных данных с найденным числом итераций
        final_params = base_model.get_params()
        final_params['iterations'] = best_iteration_count
        # Ранняя остановка больше не нужна
        final_params.pop('early_stopping_rounds', None)

        final_model = CatBoostClassifier(**final_params)
        final_model.fit(X_train, y_train, verbose=False)

        logger.info("Финальная модель CatBoost обучена на всех данных.")
        return final_model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                  X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, use_gridsearch: bool = True):
    """Обучение XGBoost - один из лучших градиентных бустингов."""
    from xgboost import XGBClassifier

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=CPU_CORES,
        tree_method="gpu_hist",  # GPU ускорение
        gpu_id=0,
    )

    if use_gridsearch:
        param_grid = {
            "n_estimators": [300, 500, 800],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [1, 1.5, 2],
        }

        if X_val is not None:
            # Объединяем train и val для GridSearchCV
            X_combined = pd.concat([X_train, X_val], ignore_index=True)
            y_combined = pd.concat([y_train, y_val], ignore_index=True)
            cv = [(list(range(len(X_train))), list(range(len(X_train), len(X_train) + len(X_val))))]
        else:
            X_combined = X_train
            y_combined = y_train
            cv = KFold(n_splits=3, shuffle=True, random_state=42)

        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=CPU_CORES,
            verbose=1,
        )
        grid.fit(X_combined, y_combined)
        logger.info(f"Лучшие параметры XGBoost: {grid.best_params_}")
        return grid.best_estimator_
    else:
        # Используем базовые параметры без поиска
        base_model.set_params(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.5
        )
        base_model.fit(X_train, y_train)
        logger.info("XGBoost обучен с базовыми параметрами")
        return base_model


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, use_gridsearch: bool = True):
    """Обучение LightGBM - быстрый и эффективный градиентный бустинг."""
    from lightgbm import LGBMClassifier

    base_model = LGBMClassifier(
        objective="binary",
        metric="auc",
        random_state=42,
        n_jobs=CPU_CORES,
        device="gpu",  # GPU ускорение
        gpu_platform_id=0,
        gpu_device_id=0,
    )

    if use_gridsearch:
        param_grid = {
            "n_estimators": [300, 500, 800],
            "max_depth": [4, 6, 8, -1],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [31, 63, 127],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [0, 0.1, 1],
        }

        if X_val is not None:
            # Объединяем train и val для GridSearchCV
            X_combined = pd.concat([X_train, X_val], ignore_index=True)
            y_combined = pd.concat([y_train, y_val], ignore_index=True)
            cv = [(list(range(len(X_train))), list(range(len(X_train), len(X_train) + len(X_val))))]
        else:
            X_combined = X_train
            y_combined = y_train
            cv = KFold(n_splits=3, shuffle=True, random_state=42)

        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=CPU_CORES,
            verbose=1,
        )
        grid.fit(X_combined, y_combined)
        logger.info(f"Лучшие параметры LightGBM: {grid.best_params_}")
        return grid.best_estimator_
    else:
        # Используем базовые параметры без поиска
        base_model.set_params(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.1
        )
        base_model.fit(X_train, y_train)
        logger.info("LightGBM обучен с базовыми параметрами")
        return base_model


def train_tabnet(X_train: pd.DataFrame, y_train: pd.Series,
                 X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
    """Обучение TabNet - нейронная сеть специально для табличных данных."""
    import torch
    import gc

    # Принудительно запускаем сборщик мусора и очищаем кэш CUDA.
    # Это решает проблему с состоянием CUDA при повторных вызовах в одной сессии.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # TabNet работает с numpy массивами
    # Удаляем столбец 'source_file', если он есть, до преобразования в numpy
    if "source_file" in X_train.columns:
        X_train = X_train.drop(columns=["source_file"])
    if X_val is not None and "source_file" in X_val.columns:
        X_val = X_val.drop(columns=["source_file"])

    X_train_np = X_train.values.astype(np.float32)
    y_train_np = y_train.values.astype(int)

    model = TabNetClassifier(
        n_d=64,  # Размерность представления
        n_a=64,  # Размерность внимания
        n_steps=5,  # Количество шагов принятия решений
        gamma=1.5,  # Коэффициент для разреженности
        n_independent=2,  # Количество независимых GLU блоков
        n_shared=2,  # Количество разделяемых GLU блоков
        lambda_sparse=1e-4,  # Регуляризация разреженности
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type="entmax",  # Тип маски внимания
        scheduler_params={"step_size": 50, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=1,
        device_name="cuda" if torch.cuda.is_available() else "cpu",
    )

    if X_val is not None and y_val is not None:
        X_val_np = X_val.values.astype(np.float32)
        y_val_np = y_val.values.astype(int)
        eval_set = [(X_val_np, y_val_np)]
        eval_name = ["val"]
    else:
        eval_set = None
        eval_name = None

    model.fit(
        X_train_np, y_train_np,
        eval_set=eval_set,
        eval_name=eval_name,
        eval_metric=["logloss", "auc"] if eval_set else None,
        max_epochs=200,
        patience=20,
        batch_size=2048,
        virtual_batch_size=256,
        drop_last=False,
    )

    logger.info("TabNet обучен успешно")
    return model


def train_mlp(X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, use_gridsearch: bool = True) -> Pipeline:
    """Обучение MLPClassifier внутри Pipeline со StandardScaler."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            max_iter=500,  # Увеличиваем количество итераций для лучшей сходимости
            random_state=42,
            solver='adam',  # Adam оптимизатор хорошо работает с GPU
            early_stopping=True,  # Ранняя остановка для предотвращения переобучения
            validation_fraction=0.1,
        )),
    ])

    if use_gridsearch:
        param_grid = {
            "mlp__hidden_layer_sizes": [(64,), (128, 64), (256, 128), (512, 256)],  # Добавляем больше архитектур
            "mlp__alpha": [1e-5, 1e-4, 1e-3, 1e-2],  # Расширяем диапазон регуляризации
            "mlp__learning_rate_init": [1e-4, 1e-3, 5e-3, 1e-2],  # Больше вариантов learning rate
            "mlp__batch_size": [32, 64, 128],  # Добавляем настройку размера батча
        }

        if X_val is not None:
            # Объединяем train и val для GridSearchCV
            X_combined = pd.concat([X_train, X_val], ignore_index=True)
            y_combined = pd.concat([y_train, y_val], ignore_index=True)
            cv = [(list(range(len(X_train))), list(range(len(X_train), len(X_train) + len(X_val))))]
        else:
            X_combined = X_train
            y_combined = y_train
            cv = KFold(n_splits=3, shuffle=True, random_state=42)

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=CPU_CORES,  # Используем 4 ядра
            verbose=1,
        )
        grid.fit(X_combined, y_combined)
        logger.info(f"Лучшие параметры MLP: {grid.best_params_}")
        return grid.best_estimator_
    else:
        # Используем базовые параметры без поиска
        pipe.set_params(
            mlp__hidden_layer_sizes=(128, 64),
            mlp__alpha=1e-4,
            mlp__learning_rate_init=1e-3,
            mlp__batch_size=64
        )
        pipe.fit(X_train, y_train)
        logger.info("MLP обучен с базовыми параметрами")
        return pipe


def train_svm(X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, use_gridsearch: bool = True) -> Pipeline:
    """Обучение SVM с RBF ядром - классический мощный алгоритм."""
    from sklearn.svm import SVC

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            probability=True,  # Для получения вероятностей
            random_state=42,
        )),
    ])

    if use_gridsearch:
        param_grid = {
            "svm__C": [0.1, 1, 10, 100],
            "svm__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
            "svm__kernel": ["rbf", "poly"],
        }

        if X_val is not None:
            # Объединяем train и val для GridSearchCV
            X_combined = pd.concat([X_train, X_val], ignore_index=True)
            y_combined = pd.concat([y_train, y_val], ignore_index=True)
            cv = [(list(range(len(X_train))), list(range(len(X_train), len(X_train) + len(X_val))))]
        else:
            X_combined = X_train
            y_combined = y_train
            cv = KFold(n_splits=3, shuffle=True, random_state=42)

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=CPU_CORES,
            verbose=1,
        )
        grid.fit(X_combined, y_combined)
        logger.info(f"Лучшие параметры SVM: {grid.best_params_}")
        return grid.best_estimator_
    else:
        # Используем базовые параметры без поиска
        pipe.set_params(
            svm__C=1,
            svm__gamma="scale",
            svm__kernel="rbf"
        )
        pipe.fit(X_train, y_train)
        logger.info("SVM обучен с базовыми параметрами")
        return pipe


def train_extra_trees(X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, use_gridsearch: bool = True):
    """Обучение ExtraTreesClassifier - улучшенная версия Random Forest."""
    from sklearn.ensemble import ExtraTreesClassifier

    base_model = ExtraTreesClassifier(
        random_state=42,
        n_jobs=CPU_CORES,
        bootstrap=True,
    )

    if use_gridsearch:
        param_grid = {
            "n_estimators": [300, 500, 800],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        }

        if X_val is not None:
            # Объединяем train и val для GridSearchCV
            X_combined = pd.concat([X_train, X_val], ignore_index=True)
            y_combined = pd.concat([y_train, y_val], ignore_index=True)
            cv = [(list(range(len(X_train))), list(range(len(X_train), len(X_train) + len(X_val))))]
        else:
            X_combined = X_train
            y_combined = y_train
            cv = KFold(n_splits=3, shuffle=True, random_state=42)

        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=CPU_CORES,
            verbose=1,
        )
        grid.fit(X_combined, y_combined)
        logger.info(f"Лучшие параметры ExtraTrees: {grid.best_params_}")
        return grid.best_estimator_
    else:
        # Используем базовые параметры без поиска
        base_model.set_params(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt"
        )
        base_model.fit(X_train, y_train)
        logger.info("ExtraTrees обучен с базовыми параметрами")
        return base_model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> RandomForestClassifier:
    """Простая модель RandomForest как baseline."""
    rf = RandomForestClassifier(
        n_estimators=500,  # Увеличиваем количество деревьев
        max_depth=None,
        random_state=42,
        n_jobs=CPU_CORES,  # Используем 4 ядра
        max_features='sqrt',  # Оптимальная настройка для классификации
        min_samples_split=5,  # Добавляем регуляризацию
        min_samples_leaf=2,
    )
    rf.fit(X_train, y_train)
    return rf


def train_extended_ensemble(X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
    """Расширенный ансамбль из лучших моделей с использованием предобученных моделей."""
    from sklearn.ensemble import VotingClassifier
    import os

    # Пути к предобученным моделям
    model_paths = {
        'catboost': 'checkpoints/catboost_model.pkl',
        'xgboost': 'checkpoints/xgboost_model.pkl',
        'lightgbm': 'checkpoints/lightgbm_model.pkl',
        'rf': 'checkpoints/rf_model.pkl',
        'extra_trees': 'checkpoints/extra_trees_model.pkl',
        'svm': 'checkpoints/svm_model.pkl',
        'mlp': 'checkpoints/mlp_model.pkl',
    }

    # Загружаем предобученные модели
    estimators = []
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                estimators.append((name, model))
                logger.info(f"Загружена предобученная модель {name} из {path}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить модель {name} из {path}: {e}")
        else:
            logger.warning(f"Файл модели {path} не найден")

    if not estimators:
        logger.error("Не удалось загрузить ни одной предобученной модели")
        raise ValueError("Отсутствуют предобученные модели для создания ансамбля")

    # Создаем расширенный ансамбль из загруженных моделей
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',  # Используем вероятности
        n_jobs=CPU_CORES,
    )

    # Обучаем только ансамбль (базовые модели уже обучены)
    ensemble.fit(X_train, y_train)
    logger.info(f"Расширенный ансамбль из {len(estimators)} предобученных моделей создан успешно")
    return ensemble


def train_voting_ensemble(X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
    """Ансамбль из лучших моделей с использованием предобученных моделей."""
    from sklearn.ensemble import VotingClassifier
    import os

    # Пути к предобученным моделям
    model_paths = {
        'catboost': 'checkpoints/catboost_model.pkl',
        'xgboost': 'checkpoints/xgboost_model.pkl',
        'lightgbm': 'checkpoints/lightgbm_model.pkl',
        'rf': 'checkpoints/rf_model.pkl',
        'extra_trees': 'checkpoints/extra_trees_model.pkl',
        'svm': 'checkpoints/svm_model.pkl',
        'mlp': 'checkpoints/mlp_model.pkl',
    }

    # Загружаем предобученные модели
    estimators = []
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                estimators.append((name, model))
                logger.info(f"Загружена предобученная модель {name} из {path}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить модель {name} из {path}: {e}")
        else:
            logger.warning(f"Файл модели {path} не найден")

    if not estimators:
        logger.error("Не удалось загрузить ни одной предобученной модели")
        raise ValueError("Отсутствуют предобученные модели для создания ансамбля")

    # Создаем ансамбль из загруженных моделей
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',  # Используем вероятности
        n_jobs=CPU_CORES,
    )

    # Обучаем только ансамбль (базовые модели уже обучены)
    ensemble.fit(X_train, y_train)
    logger.info(f"Voting ансамбль из {len(estimators)} предобученных моделей создан успешно")
    return ensemble


def train_stacking_ensemble(X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
    """Стекинг ансамбль с использованием предобученных моделей."""
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    import os

    # Пути к предобученным моделям
    model_paths = {
        'catboost': 'checkpoints/catboost_model.pkl',
        'xgboost': 'checkpoints/xgboost_model.pkl',
        'lightgbm': 'checkpoints/lightgbm_model.pkl',
        'rf': 'checkpoints/rf_model.pkl',
        'extra_trees': 'checkpoints/extra_trees_model.pkl',
    }

    # Загружаем предобученные модели
    base_models = []
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                base_models.append((name, model))
                logger.info(f"Загружена предобученная модель {name} из {path}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить модель {name} из {path}: {e}")
        else:
            logger.warning(f"Файл модели {path} не найден")

    if not base_models:
        logger.error("Не удалось загрузить ни одной предобученной модели")
        raise ValueError("Отсутствуют предобученные модели для создания стекинг ансамбля")

    # Мета-модель
    meta_model = LogisticRegression(random_state=42, max_iter=1000)

    # Создаем стекинг ансамбль
    if X_val is not None:
        cv = [(list(range(len(X_train))), list(range(len(X_train), len(X_train) + len(X_val))))]
        X_combined = pd.concat([X_train, X_val], ignore_index=True)
        y_combined = pd.concat([y_train, y_val], ignore_index=True)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        X_combined = X_train
        y_combined = y_train

    stacking_ensemble = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=cv,  # Кросс-валидация для обучения мета-модели
        n_jobs=CPU_CORES,
    )

    stacking_ensemble.fit(X_combined, y_combined)
    logger.info(f"Стекинг ансамбль из {len(base_models)} предобученных моделей обучен успешно")
    return stacking_ensemble


MODEL_TRAINERS = {
    "catboost": train_catboost,
    "xgboost": train_xgboost,
    "lightgbm": train_lightgbm,
    "mlp": train_mlp,
    "rf": train_random_forest,
    "extra_trees": train_extra_trees,
    "svm": train_svm,
    "tabnet": train_tabnet,
    "voting_ensemble": train_voting_ensemble,
    "extended_ensemble": train_extended_ensemble,
    "stacking_ensemble": train_stacking_ensemble,
}


def load_dataset(csv_path: str):
    """Загружает датасет из CSV файла."""
    logger.info(f"Загрузка датасета из {csv_path}")
    df = pd.read_csv(csv_path, sep=';')
    if "label" not in df.columns:
        raise ValueError("В датасете отсутствует столбец 'label'.")
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)
    groups = df["source_file"] if "source_file" in df.columns else None
    return X, y, groups


class MLTrainer:
    """
    Класс для выполнения процесса обучения моделей.
    """

    def __init__(self, output_dir: str = "checkpoints"):
        """
        Инициализирует тренер.

        Args:
            output_dir (str): Каталог для сохранения моделей.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def train(self, train_csv: str, val_csv: Optional[str] = None, models: List[str] = None):
        """
        Запускает обучение для указанных моделей.

        Args:
            train_csv (str): Путь к обучающему CSV.
            val_csv (Optional[str]): Путь к валидационному CSV (если None, используется KFold).
            models (List[str]): Список названий моделей для обучения.
        """
        X_train, y_train, _ = load_dataset(train_csv)

        if val_csv is not None:
            X_val, y_val, _ = load_dataset(val_csv)
            # Удаляем столбец 'source_file', если он есть
            if "source_file" in X_val.columns:
                X_val = X_val.drop(columns=["source_file"])
        else:
            X_val, y_val = None, None

        # Удаляем столбец 'source_file', если он есть
        if "source_file" in X_train.columns:
            X_train = X_train.drop(columns=["source_file"])

        for model_name in models:
            logger.info(f"Обучение модели: {model_name}")
            trainer_func = MODEL_TRAINERS[model_name]

            model = trainer_func(X_train, y_train, X_val, y_val)

            # TabNet требует специального сохранения
            if TabNetClassifier and isinstance(model, TabNetClassifier):
                # pytorch-tabnet автоматически добавляет .zip, поэтому мы его убираем из пути
                base_model_path = os.path.join(self.output_dir, f"{model_name}_model")
                model.save_model(base_model_path)
                model_path = f"{base_model_path}.zip"
            else:
                model_path = os.path.join(self.output_dir, f"{model_name}_model.pkl")
                joblib.dump(model, model_path)

            logger.info(f"Модель сохранена в {model_path}")

        return self
