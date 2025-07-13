import os
import glob


class EnabledFileTypes:
    LAS = 'las'
    PCD = 'pcd'
    TXT = 'txt'
    LAZ = 'laz'


class PathManager:
    def __init__(self):
        self._base_dir: str | None = None

    def set_base_dir(self, path: str) -> 'PathManager':
        self._base_dir = path
        os.makedirs(self._base_dir, exist_ok=True)
        return self

    def get_dataset_dir(self, type) -> str:
        assert self._base_dir, "Базовая директория не установлена"
        path = os.path.join(self._base_dir, type)
        os.makedirs(path, exist_ok=True)
        return path

    def get_individual_dataset_dir(self, type) -> str:
        individual_dir = os.path.join(self.get_dataset_dir(type), 'individual')
        os.makedirs(individual_dir, exist_ok=True)
        return individual_dir

    def get_computed_dataset_path(self, type) -> str:
        dir = self.get_dataset_dir(type)
        path = os.path.join(dir, f'{type}_computed.csv')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def get_model_path(self, model_name) -> str:
        path = os.path.join(self._base_dir, 'models', model_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def get_prepared_files_dir(self, type: str) -> str:
        path = os.path.join(self.get_dataset_dir(type), 'prepared')
        os.makedirs(path, exist_ok=True)
        return path

    def get_file_paths(self, input_dir: str) -> list[str]:
        file_paths = []
        for ext in vars(EnabledFileTypes).values():
            search_pattern = os.path.join(input_dir, f'*{ext}')
            file_paths.extend(glob.glob(search_pattern))
        return file_paths
