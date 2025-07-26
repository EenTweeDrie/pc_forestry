import os
import glob
import pandas as pd
import joblib
from tqdm import tqdm
from typing import Optional, Any
import logging

from .ml_trainer import MLTrainer, load_dataset, MODEL_TRAINERS
from .ml_validator import MLValidator
from .fit import predict_for_tree  # fit пока остается
from .dataset_builder import DatasetBuilder
from ..path_manager import PathManager
from ..pcd.TREE import TREE
from ..pcd.VOXEL import VOXELGRID
from ..utils.timer import Timer

logger = logging.getLogger(__name__)


class MLPipeline:
    """
    Класс для обучения и инференса моделей классификации вокселей.
    Предоставляет текучий интерфейс для настройки и запуска процессов.
    """

    def __init__(self, path: str):
        """
        Инициализирует MLPipeline с использованием PathManager.
        """
        self._model_type: Optional[str] = None
        self._model: Optional[Any] = None
        self.path_manager = PathManager().set_base_dir(path)
        self.validator = MLValidator()
        self._datasets_config = {}

    def set_model_type(self, model_type: str) -> 'MLPipeline':
        """
        Устанавливает модель для обучения.

        Args:
            model_type (str): Название модели ('catboost', 'mlp', 'rf').

        Returns:
            MLPipeline: Экземпляр для цепочки вызовов.
        """
        assert model_type in MODEL_TRAINERS, (
            f"Модель {model_type} не поддерживается. "
            f"Поддерживаемые модели: {list(MODEL_TRAINERS.keys())}")
        self._model_type = model_type
        return self

    def set_datasets_config(self, config: dict):
        self._datasets_config = config
        return self

    def get_model_binary(self) -> Optional[Any]:
        assert self._model is not None, "Модель еще не обучена или не загружена."
        return self._model

    def compute_datasets(self, types=['train', 'val', 'test'], force=True) -> 'MLPipeline':
        builder = DatasetBuilder(self.path_manager, self._datasets_config)
        builder.build(types, force)
        return self

    def train(self) -> 'MLPipeline':
        assert self._model_type is not None, "Модель не выбрана. Используйте set_model()."

        train_csv = self.path_manager.get_computed_dataset_path('train')
        val_csv = self.path_manager.get_computed_dataset_path('val')

        assert os.path.exists(train_csv), "Обучающий CSV файл не найден. Запустите compute_datasets() перед обучением."

        # Проверяем существование валидационного файла
        val_csv_exists = os.path.exists(val_csv)
        if not val_csv_exists:
            print("Валидационный CSV файл не найден. Обучение будет проводиться только на обучающем наборе с использованием кросс-валидации.")
            val_csv = None

        checkpoints_dir = os.path.dirname(
            self.path_manager.get_model_path(self._model_type))
        os.makedirs(checkpoints_dir, exist_ok=True)

        # 1. Обучение
        trainer = MLTrainer(output_dir=checkpoints_dir)
        trainer.train(
            train_csv=train_csv,
            val_csv=val_csv,
            models=[self._model_type]
        )

        # 2. Загрузка модели
        model_path = os.path.join(
            checkpoints_dir, f"{self._model_type}_model.pkl")
        self._model = joblib.load(model_path)
        print(f"Модель '{self._model_type}' обучена и загружена.")

        # 3. Оценка на валидационном наборе (если он существует)
        if val_csv_exists:
            X_val, y_val, groups_val = load_dataset(val_csv)
            if "source_file" in X_val.columns:
                X_val = X_val.drop(columns=["source_file"])

            metrics = self.validator.evaluate(
                self._model, X_val, y_val, groups_val)
            print("\nОценка на валидационном наборе:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        else:
            print("\nВалидационный набор недоступен. Оценка пропущена.")

        return self

    def eval(self) -> 'MLPipeline':
        """
        Оценивает обученную модель на тестовом наборе данных.
        """
        assert self._model is not None, "Модель не обучена. Вызовите train() сначала."

        test_csv_path = self.path_manager.get_computed_dataset_path('test')

        if not os.path.exists(test_csv_path):
            print("Тестовый CSV не найден. Пропуск оценки на тестовом наборе.")
            print("Чтобы создать его, используйте compute_datasets().")
            return self

        X_test, y_test, groups_test = load_dataset(test_csv_path)

        if "source_file" in X_test.columns:
            X_test = X_test.drop(columns=["source_file"])

        metrics = self.validator.evaluate(
            self._model, X_test, y_test, groups_test)
        print("\nОценка на тестовом наборе:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        return self

    def fit(self, file_path: str) -> Any:
        """
        Выполняет инференс (предсказание) для одного файла с данными о дереве.

        Args:
            file_path (str): Путь к файлу (.pcd, .las, .txt).

        Returns:
            VOXELGRID: Объект воксельной сетки с результатами предсказания.
                       Каждый воксель будет иметь атрибут `label`.
        """
        assert self._model is not None, "Модель не обучена. Вызовите train() сначала."
        assert os.path.exists(file_path), f"Файл не найден: {file_path}"

        print(f"Выполнение предсказания для файла: {file_path}")
        voxelgrid_with_predictions = predict_for_tree(self._model, file_path)
        print("Предсказание завершено.")
        return voxelgrid_with_predictions
