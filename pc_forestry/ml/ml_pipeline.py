import os
import glob
import pandas as pd
import joblib
from tqdm import tqdm
from typing import Optional, Any, Tuple
import logging
import numpy as np

from .ml_trainer import MLTrainer, load_dataset, MODEL_TRAINERS
from .ml_validator import MLValidator
from .ml_inferencer import MLInferencer
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

    def set_model(self, model_path: str) -> 'MLPipeline':
        """
        Загружает предварительно обученную модель из файла.

        Args:
            model_path (str): Путь к файлу модели (обычно .pkl).

        Returns:
            MLPipeline: Экземпляр для цепочки вызовов.
        """
        assert os.path.exists(model_path), f"Файл модели не найден: {model_path}"

        self._model = joblib.load(model_path)
        print(f"Модель успешно загружена из: {model_path}")

        # Попытка определить тип модели из имени файла для согласованности
        model_filename = os.path.basename(model_path)
        for model_type_key in MODEL_TRAINERS.keys():
            if model_type_key in model_filename:
                self._model_type = model_type_key
                print(f"Тип модели определен как '{self._model_type}'.")
                break
        else:
            logger.warning("Не удалось автоматически определить тип модели из имени файла.")

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
        inferencer = MLInferencer(self._model)
        voxelgrid_with_predictions = inferencer.predict_for_tree(file_path)
        print("Предсказание завершено.")
        return voxelgrid_with_predictions

    def eval(self) -> 'MLPipeline':
        """
        Оценивает обученную модель на тестовом наборе данных, используя end-to-end логику `fit`.
        Для каждого файла из тестового набора выполняется инференс,
        а затем предсказанные метки сравниваются с истинными.
        """
        assert self._model is not None, "Модель не обучена. Вызовите train() сначала."
        test_dir = self.path_manager.get_prepared_files_dir('test')
        test_files = self.path_manager.get_file_paths(test_dir)

        all_true_labels = []
        all_pred_labels = []

        print(f"\nЗапуск E2E оценки на {len(test_files)} тестовых файлах...")
        for file_path in tqdm(test_files, desc="Оценка тестовых файлов"):
            # 1. Получаем предсказания
            predicted_voxelgrid = self.fit(file_path)
            pred_labels = np.array([voxel.label for voxel in predicted_voxelgrid.voxels])

            # 2. Загружаем исходные данные с истинными метками
            true_tree = TREE.read(file_path)
            true_tree.shift_to_coordinate()
            true_voxelgrid = VOXELGRID.create(true_tree, voxel_size=predicted_voxelgrid.voxel_size)
            true_labels = np.array([voxel.label for voxel in true_voxelgrid.voxels])

            # Убедимся, что количество вокселей совпадает
            if len(pred_labels) != len(true_labels):
                logger.warning(
                    f"Пропуск файла {os.path.basename(file_path)}: "
                    f"несоответствие количества вокселей "
                    f"({len(pred_labels)} предсказано, {len(true_labels)} истинных)."
                )
                continue

            all_pred_labels.extend(pred_labels)
            all_true_labels.extend(true_labels)

        if not all_true_labels:
            print("Не удалось обработать ни одного файла для оценки.")
            return self

        # 3. Считаем метрики
        metrics = self.validator.calculate_metrics(np.array(all_true_labels), np.array(all_pred_labels))

        print("\nИтоговая оценка на тестовом наборе (E2E):")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        return self
