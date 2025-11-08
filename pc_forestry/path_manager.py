import os
import glob
from typing import Iterable


class EnabledFileTypes:
    LAS = 'las'
    PCD = 'pcd'
    TXT = 'txt'
    LAZ = 'laz'
    STL = 'stl'


class PathManager:
    def __init__(self):
        self._base_dir: str | None = None

    def set_base_dir(self, path: str) -> 'PathManager':
        self._base_dir = path
        os.makedirs(self._base_dir, exist_ok=True)
        return self

    def _ensure_base_dir(self) -> str:
        assert self._base_dir, "Базовая директория не установлена"
        return self._base_dir

    def resolve(self, *relative_parts: str) -> str:
        base_dir = self._ensure_base_dir()
        return os.path.join(base_dir, *relative_parts)

    def ensure_dir(self, *relative_parts: str) -> str:
        path = self.resolve(*relative_parts)
        os.makedirs(path, exist_ok=True)
        return path

    def ensure_directories(self, relative_parts: Iterable[str]) -> list[str]:
        return [self.ensure_dir(part) for part in relative_parts]

    def resolve_file(self, relative_path: str) -> str:
        return self.resolve(relative_path)

    def get_dataset_dir(self, type) -> str:
        path = self.ensure_dir(type)
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

    def get_area_file_path(self, area_name: str) -> str:
        path = os.path.join(self._base_dir, f'{area_name}')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def get_cut_area_file_path(self, area_name: str) -> str:
        path = os.path.join(self._base_dir, f'cut_{area_name}')
        return path

    def get_cells_dir(self) -> str:
        path = os.path.join(self._base_dir, 'cells')
        os.makedirs(path, exist_ok=True)
        return path

    def get_cells_borders_dir(self) -> str:
        path = os.path.join(self.get_cells_dir(), 'borders')
        os.makedirs(path, exist_ok=True)
        return path

    def get_cells_data_dir(self) -> str:
        path = os.path.join(self.get_cells_dir(), 'data')
        os.makedirs(path, exist_ok=True)
        return path

    def get_stumps_dir(self, intensity_cut) -> str:
        path = os.path.join(self._base_dir, str(intensity_cut) + '_stumps')
        os.makedirs(path, exist_ok=True)
        return path

    def get_file_paths(self, input_dir: str) -> list[str]:
        file_paths = []
        for ext in vars(EnabledFileTypes).values():
            search_pattern = os.path.join(input_dir, f'*{ext}')
            file_paths.extend(glob.glob(search_pattern))
        return file_paths
