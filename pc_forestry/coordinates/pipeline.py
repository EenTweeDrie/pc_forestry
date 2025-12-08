import os
from tqdm import tqdm
import numpy as np
from typing import Any, Dict, Optional
from ..path_manager import PathManager
from ..pcd.PCD import PCD
from .mesh_adapter import MeshAdapter
from .utils import shp_create
from .VOR_TES import VOR_TES
from .make_stumps.utils import save_stumps_results
from .make_stumps.algorithm_intensity import process_cell_file
from .make_stumps.algorithm_rgb import process_cell_file_rgb


class CoordinatesPipeline:
    def __init__(self, base_path: str, file_path: str) -> None:
        self.params: Dict[str, Any] = {}
        self.base_path = base_path
        self.path_manager = PathManager().set_base_dir(base_path)
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.mesh_adapter: Optional[MeshAdapter] = None

    def set_params(self, params: Dict[str, Any]) -> "CoordinatesPipeline":
        self.params = dict(params)
        return self

    def update_params(self, params: Dict[str, Any]) -> "CoordinatesPipeline":
        self.params.update(params)
        return self

    def set_mesh(self, mesh_path: str) -> "CoordinatesPipeline":
        """
        Устанавливает mesh адаптер для адаптивного определения высоты срезов.

        Args:
            mesh_path: Путь к файлу mesh'а (по умолчанию 'mesh.stl')
        """
        try:
            self.mesh_adapter = MeshAdapter(mesh_path)
            print(f"Загружен mesh из файла: {mesh_path}")

            stats = self.mesh_adapter.get_mesh_statistics()
            if stats:
                print(f"Статистика mesh'а:")
                print(f"  - Вершин: {stats['num_vertices']}")
                print(f"  - Треугольников: {stats['num_triangles']}")
                print(f"  - Высота: {stats['z_min']:.2f} - {stats['z_max']:.2f} м")
                print(f"  - Размеры: {stats['x_extent']:.2f} x {stats['y_extent']:.2f} м")

        except Exception as e:
            print(f"Предупреждение: Не удалось загрузить mesh файл {mesh_path}: {e}")
            print("Будет использоваться фиксированная высота среза")
            self.mesh_adapter = None

        return self

    def cut_mesh_data(self, force: bool = True) -> "CoordinatesPipeline":
        if not force and os.path.exists(self.path_manager.get_cut_area_file_path(os.path.splitext(self.file_name)[0] + '.pcd')):
            return self

        pc_area = PCD.read(self.file_path)

        if len(pc_area.points) == 0:
            raise ValueError(f"Облако точек пустое: {self.file_path}")

        print(f"Загружено точек: {len(pc_area.points)}")
        print(f"Диапазон Z исходного облака: [{pc_area.points[:, 2].min():.2f}, {pc_area.points[:, 2].max():.2f}]м")

        shift_vector = self.params.get('shift_vector', pc_area.calculate_auto_shift_vector())
        # print(f"Применяю сдвиг: {shift_vector}")
        # pc_area.shift_with_vector(shift_vector=[shift_vector[0], shift_vector[1], 0])

        if self.mesh_adapter is not None:
            height_from = self.params.get('mesh_height_from')
            height_to = self.params.get('mesh_height_to')

            mesh_stats = self.mesh_adapter.get_mesh_statistics()
            print(f"Статистика mesh: {mesh_stats}")
            print(f"Фильтрация по высоте относительно mesh: [{height_from}, {height_to}]м")

            height_mask = self.mesh_adapter.create_relative_height_slice(
                pc_area.points,
                height_from=height_from,
                height_to=height_to,
            )

            idx_labels = np.where(height_mask)[0]

            if len(idx_labels) == 0:
                raise ValueError(
                    f"После применения фильтра высот [{height_from}, {height_to}]м относительно mesh "
                    f"не осталось ни одной точки!\n"
                    f"Исходное облако: {len(pc_area.points)} точек, "
                    f"Z: [{pc_area.points[:, 2].min():.2f}, {pc_area.points[:, 2].max():.2f}]м\n"
                    f"Mesh Z: [{mesh_stats.get('z_min', 'N/A'):.2f}, {mesh_stats.get('z_max', 'N/A'):.2f}]м\n"
                    f"Проверьте параметры mesh_height_from и mesh_height_to, а также shift_vector."
                )

            print(f"После фильтрации осталось точек: {len(idx_labels)}")
            pc_area.index_cut(idx_labels)

        else:
            raise Exception("Mesh adapter not found")

        # pc_area.shift_with_vector(shift_vector=[-shift_vector[0], -shift_vector[1], 0])
        # pc_area.show(color_field='rgb')
        # kostyl
        self.file_name = os.path.splitext(self.file_name)[0] + '.pcd'
        # kostyl
        pc_area.save(self.path_manager.get_cut_area_file_path(self.file_name))
        return self

    def cut_slice_data(self) -> "CoordinatesPipeline":
        pc_area = PCD.read(self.path_manager.get_area_file_path(self.file_name))
        # shift_vector = self.params.get('shift_vector', pc_area.calculate_auto_shift_vector())
        # pc_area.shift_with_vector(shift_vector=shift_vector)

        idx_labels = np.where((pc_area.points[:, 2] > self.params['low_height'] + pc_area.points[:, 2].min()) &
                              (pc_area.points[:, 2] <= self.params['high_height'] + pc_area.points[:, 2].min()))
        pc_area.index_cut(idx_labels)
        # kostyl
        self.file_name = os.path.splitext(self.file_name)[0] + '.pcd'
        # kostyl
        pc_area.save(self.path_manager.get_cut_area_file_path(self.file_name))
        return self

    def make_cells(self, force: bool = True) -> "CoordinatesPipeline":
        if not force and os.path.exists(self.path_manager.get_cells_data_dir()) and os.listdir(self.path_manager.get_cells_data_dir()):
            return self

        # kostyl
        self.file_name = os.path.splitext(self.file_name)[0] + '.pcd'
        # kostyl
        pc_area = PCD.read(self.path_manager.get_cut_area_file_path(self.file_name))

        shp_poly = shp_create(pc_area.points)

        vortes = VOR_TES(pc=pc_area, algo=self.params['algo'], n_clusters=self.params['n_clusters'],
                         percent=self.params['intensity_cut_vor_tes_percent'])
        vortes.select_borders(self.path_manager.get_cells_borders_dir(), shp_poly, verbose=False)
        vortes.select_clusters(
            path_folder_from=self.path_manager.get_cells_borders_dir(),
            path_folder_to=self.path_manager.get_cells_data_dir()
        )
        return self

    def make_stumps(self, force: bool = True) -> "CoordinatesPipeline":
        """
        Создание пней из данных ячеек с использованием выбранного алгоритма.
        Алгоритм выбирается параметром 'stump_algorithm' ('intensity' или 'rgb').
        """
        algo = self.params.get('stump_algorithm', 'intensity')

        stumps_id = self.params.get('stumps_id')
        process_function = process_cell_file

        if not force and os.path.exists(self.path_manager.get_stumps_dir(stumps_id)) and os.listdir(self.path_manager.get_stumps_dir(stumps_id)):
            print(f"Папка с результатами для '{stumps_id}' уже существует. Пропускаю шаг.")
            return self

        path_file_cells = self.path_manager.get_cells_data_dir()
        file_paths = self.path_manager.get_file_paths(path_file_cells)

        # Инициализация списков для хранения результатов
        counter = 0
        all_names = []
        all_x_coords = []
        all_y_coords = []
        all_diameters = []

        for filename in file_paths:
            if filename.endswith('.pcd'):
                pc_cells = PCD.read(filename)

                if pc_cells.points.shape[0] == 0:
                    continue

                if algo == 'intensity':
                    stumps_data = process_function(pc_cells, counter, self.params, self.path_manager, self.mesh_adapter)
                else:
                    stumps_data = process_function(pc_cells, counter, self.params, self.path_manager)

                # Обновляем счетчик и добавляем данные
                counter = stumps_data['counter']
                all_names.extend(stumps_data['names'])
                all_x_coords.extend(stumps_data['x_coords'])
                all_y_coords.extend(stumps_data['y_coords'])
                all_diameters.extend(stumps_data['diameters'])

        save_stumps_results(all_names, all_x_coords, all_y_coords, all_diameters, self.params, self.path_manager)
        return self
