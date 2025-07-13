import os
import glob
import pandas as pd
from tqdm import tqdm
from loguru import logger

from ..path_manager import PathManager
from ..pcd.TREE import TREE
from ..pcd.VOXEL import VOXELGRID
from ..utils.timer import Timer


class DatasetBuilder:
    """
    Отвечает за построение датасетов из исходных файлов облаков точек.
    """

    def __init__(self, path_manager: PathManager, datasets_config: dict):
        self.path_manager = path_manager
        self._datasets_config = datasets_config

    @Timer("Подготовка файла")
    def _prepare_tree_file(self, file_path: str) -> TREE:
        pc = TREE.read(file_path)
        pc.estimate_normals()
        pc.calculate_illuminance_fast()
        return pc

    @Timer("Подготовка файлов")
    def _save_prepared_file(self, dataset_type: str) -> None:
        initial_files_dir = self.path_manager.get_dataset_dir(dataset_type)
        prepared_files_dir = self.path_manager.get_prepared_files_dir(dataset_type)

        file_paths = []
        file_paths = self.path_manager.get_file_paths(initial_files_dir)

        for file_path in file_paths:
            pc = self._prepare_tree_file(file_path)
            pc.save(os.path.join(prepared_files_dir, os.path.basename(file_path)))

    @Timer("Обработка файла")
    def _process_tree_file(self, file_path: str, voxel_size: float) -> pd.DataFrame:
        pc = TREE.read(file_path)
        pc.shift_to_coordinate()
        pc.estimate_normals()
        vg = VOXELGRID.create(pc, voxel_size, verbose=False)
        vg.calculate_distances_to_previous_layer(pc.coordinate)
        vg.calculate_distances_to_coordinate(pc.coordinate)
        df = vg.normalized_df
        return df

    @Timer("Объединение всех DataFrame'ов")
    def _combine_individual_datasets(self, dataset_type: str) -> pd.DataFrame:
        individual_output_dir = self.path_manager.get_individual_dataset_dir(dataset_type)
        all_dfs = []
        for file in os.listdir(individual_output_dir):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(individual_output_dir, file), sep=';')
                all_dfs.append(df)
        combined_df = pd.concat(all_dfs, ignore_index=True)
        computed_dataset_path = self.path_manager.get_computed_dataset_path(dataset_type)
        combined_df.to_csv(computed_dataset_path, index=False)
        logger.info(f"Сборный датасет сохранен в: {computed_dataset_path}")
        return combined_df

    @Timer("Построение датасета")
    def _build_dataset(self, dataset_type: str, voxel_size: float):
        input_dir = self.path_manager.get_prepared_files_dir(dataset_type)

        file_paths = []
        file_paths = self.path_manager.get_file_paths(input_dir)
        if not file_paths:
            logger.warning(f"Не найдено файлов в директории: {input_dir}. Запускаем подготовку файлов.")
            self._save_prepared_file(dataset_type)
            file_paths = self.path_manager.get_file_paths(input_dir)
            assert file_paths, f"После подготовки файлов не найдено файлов в директории: {input_dir}"

        all_dfs = []
        for file_path in tqdm(file_paths, desc=f"Обработка {dataset_type} датасета"):
            df = self._process_tree_file(file_path, voxel_size)
            if df is not None:
                df['source_file'] = os.path.basename(file_path)
                all_dfs.append(df)
                output_filename = os.path.basename(file_path).replace('.las', '.csv')
                individual_save_path = os.path.join(self.path_manager.get_individual_dataset_dir(dataset_type), output_filename)
                df.to_csv(individual_save_path, index=False, sep=';')
                logger.debug(f"Индивидуальный DataFrame сохранен в: {individual_save_path}")
        assert all_dfs, "Не удалось обработать ни одного файла. Сборный датасет не будет создан."

        self._combine_individual_datasets(dataset_type)

    def build(self, types=['train', 'val', 'test']):
        """
        Запускает процесс построения датасетов для указанных типов.
        """
        voxel_size = self._datasets_config.get('voxel_size', 0.5)
        if 'voxel_size' not in self._datasets_config:
            logger.warning("Параметр 'voxel_size' не задан явно, используется значение по умолчанию 0.5.")
        for dataset_type in types:
            self._build_dataset(dataset_type, voxel_size)
