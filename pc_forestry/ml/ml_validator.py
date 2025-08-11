import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:
    TabNetClassifier = None


class MLValidator:
    """
    Класс для оценки (валидации) моделей машинного обучения.
    """

    def evaluate(self, model, X: pd.DataFrame, y: pd.Series, group_ids: pd.Series = None) -> Dict[str, float]:
        """
        Вычисляет AUC, лог-лосс и qAUC (если переданы группы).

        Args:
            model: Обученная модель с методом `predict_proba`.
            X (pd.DataFrame): Признаки для оценки.
            y (pd.Series): Истинные метки.
            group_ids (pd.Series, optional): Идентификаторы групп для qAUC. Defaults to None.

        Returns:
            Dict[str, float]: Словарь с метриками.
        """
        if TabNetClassifier and isinstance(model, TabNetClassifier):
            X_np = X.values.astype(np.float32)
            proba = model.predict_proba(X_np)[:, 1]
        else:
            proba = model.predict_proba(X)[:, 1]

        metrics = {
            "auc": roc_auc_score(y, proba),
            "logloss": log_loss(y, proba),
            "accuracy": accuracy_score(y, (proba > 0.5).astype(int)),
            "precision": precision_score(y, (proba > 0.5).astype(int)),
            "recall": recall_score(y, (proba > 0.5).astype(int)),
            "f1": f1_score(y, (proba > 0.5).astype(int)),
            "mcc": matthews_corrcoef(y, (proba > 0.5).astype(int)),
        }
        if group_ids is not None:
            metrics["qauc"] = self._qauc_by_group(
                y.values, proba, group_ids.values)
        return metrics

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Вычисляет метрики на основе истинных и предсказанных меток.

        Args:
            y_true (np.ndarray): Истинные метки.
            y_pred (np.ndarray): Предсказанные метки.

        Returns:
            Dict[str, float]: Словарь с метриками.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
        }
        return metrics

    def _qauc_by_group(self, y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
        """
        Среднее ROC-AUC, рассчитанное отдельно для каждой группы.
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
