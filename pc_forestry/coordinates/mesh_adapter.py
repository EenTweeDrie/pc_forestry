import numpy as np
import open3d as o3d
from typing import Tuple, Optional
from loguru import logger
from scipy.spatial import KDTree
import os


class MeshAdapter:
    """
    Класс для работы с mesh сетками и адаптивного определения высоты срезов
    на основе топографии mesh'а.
    """

    def __init__(self, mesh_path: str):
        """
        Инициализация адаптера mesh'а.

        Args:
            mesh_path: Путь к STL файлу mesh'а
        """
        self.mesh_path = mesh_path
        self.mesh = None
        self.vertices = None
        self.kdtree = None
        self._load_mesh()

    def _load_mesh(self) -> None:
        """Загружает mesh из STL файла."""
        if not os.path.exists(self.mesh_path):
            raise FileNotFoundError(f"STL файл не найден: {self.mesh_path}")

        try:
            self.mesh = o3d.io.read_triangle_mesh(self.mesh_path)
            if len(self.mesh.vertices) == 0:
                raise ValueError("Mesh не содержит вершин")

            self.vertices = np.asarray(self.mesh.vertices)
            # Создаем KD-tree для быстрого поиска ближайших точек
            self.kdtree = KDTree(self.vertices[:, :2])  # Используем только X, Y координаты

            logger.info(f"Загружен mesh с {len(self.vertices)} вершинами из {self.mesh_path}")

        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке mesh файла: {e}")

    def get_mesh_height_at_point(self, x: float, y: float, search_radius: float = 1.0) -> Optional[float]:
        """
        Получает высоту mesh'а в заданной точке (x, y).

        Args:
            x: X координата
            y: Y координата
            search_radius: Радиус поиска ближайших точек mesh'а

        Returns:
            Высота mesh'а в данной точке или None если не найдено
        """
        try:
            # Найдем ближайшие точки mesh'а в радиусе поиска
            indices = self.kdtree.query_ball_point([x, y], search_radius)

            if not indices:
                # Если в радиусе ничего не найдено, найдем ближайшую точку
                _, nearest_idx = self.kdtree.query([x, y], k=1)
                return self.vertices[nearest_idx, 2]

            # Вычисляем средневзвешенную высоту по расстоянию
            nearby_vertices = self.vertices[indices]
            distances = np.sqrt((nearby_vertices[:, 0] - x)**2 + (nearby_vertices[:, 1] - y)**2)

            # Избегаем деления на ноль
            distances = np.maximum(distances, 1e-8)
            weights = 1.0 / distances
            weighted_height = np.average(nearby_vertices[:, 2], weights=weights)

            return weighted_height

        except Exception as e:
            logger.warning(f"Ошибка при получении высоты mesh'а в точке ({x}, {y}): {e}")
            return None

    def get_adaptive_low_height(self, points: np.ndarray, height_offset: float = 0.1,
                                grid_resolution: float = 1.0) -> np.ndarray:
        """
        Вычисляет адаптивную нижнюю высоту для каждой точки облака на основе mesh'а.

        Args:
            points: Массив точек облака (N x 3)
            height_offset: Смещение от поверхности mesh'а (в метрах)
            grid_resolution: Разрешение сетки для оптимизации вычислений

        Returns:
            Массив адаптивных нижних высот для каждой точки
        """
        if self.mesh is None:
            raise RuntimeError("Mesh не загружен")

        # Получаем уникальные координаты X, Y с заданным разрешением
        x_coords = points[:, 0]
        y_coords = points[:, 1]

        # Создаем сетку для оптимизации вычислений
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        x_grid = np.arange(x_min, x_max + grid_resolution, grid_resolution)
        y_grid = np.arange(y_min, y_max + grid_resolution, grid_resolution)

        # Создаем словарь высот для сетки
        height_cache = {}

        logger.info(f"Вычисляю адаптивные высоты для сетки {len(x_grid)}x{len(y_grid)}")

        for x in x_grid:
            for y in y_grid:
                mesh_height = self.get_mesh_height_at_point(x, y)
                if mesh_height is not None:
                    grid_key = (round(x / grid_resolution), round(y / grid_resolution))
                    height_cache[grid_key] = mesh_height + height_offset

        # Вычисляем адаптивные высоты для каждой точки
        adaptive_heights = np.zeros(len(points))

        for i, point in enumerate(points):
            x, y = point[0], point[1]
            grid_key = (round(x / grid_resolution), round(y / grid_resolution))

            if grid_key in height_cache:
                adaptive_heights[i] = height_cache[grid_key]
            else:
                # Если нет в кэше, вычисляем напрямую
                mesh_height = self.get_mesh_height_at_point(x, y)
                adaptive_heights[i] = (mesh_height + height_offset) if mesh_height is not None else 0.0

        return adaptive_heights

    def create_relative_height_slice(self, points: np.ndarray, height_from: float, height_to: float,
                                     grid_resolution: float = 1.0) -> np.ndarray:
        """
        Создает маску для выбора кусочка облака точек в диапазоне высот относительно mesh'а.

        Args:
            points: Массив точек облака (N x 3)
            height_from: Нижняя граница относительно mesh'а (в метрах)
            height_to: Верхняя граница относительно mesh'а (в метрах)
            grid_resolution: Разрешение сетки для оптимизации

        Returns:
            Булева маска для фильтрации точек
        """
        if height_from >= height_to:
            raise ValueError("height_from должно быть меньше height_to")

        if self.mesh is None:
            raise RuntimeError("Mesh не загружен")

        # Создаем сетку для оптимизации вычислений
        x_coords = points[:, 0]
        y_coords = points[:, 1]

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        x_grid = np.arange(x_min, x_max + grid_resolution, grid_resolution)
        y_grid = np.arange(y_min, y_max + grid_resolution, grid_resolution)

        # Создаем кэш высот mesh'а для сетки
        height_cache = {}
        for x in x_grid:
            for y in y_grid:
                mesh_height = self.get_mesh_height_at_point(x, y)
                if mesh_height is not None:
                    grid_key = (round(x / grid_resolution), round(y / grid_resolution))
                    height_cache[grid_key] = mesh_height

        # Вычисляем маску для каждой точки
        mask = np.zeros(len(points), dtype=bool)

        for i, point in enumerate(points):
            x, y, z = point[0], point[1], point[2]
            grid_key = (round(x / grid_resolution), round(y / grid_resolution))

            # Получаем высоту mesh'а
            if grid_key in height_cache:
                mesh_height = height_cache[grid_key]
            else:
                mesh_height = self.get_mesh_height_at_point(x, y)
                if mesh_height is None:
                    mesh_height = 0.0

            # Проверяем, попадает ли точка в диапазон относительно mesh'а
            absolute_low = mesh_height + height_from
            absolute_high = mesh_height + height_to

            mask[i] = (z > absolute_low) and (z <= absolute_high)

        logger.info(f"Создан срез [{height_from:.2f}, {height_to:.2f}]м от mesh'а: {mask.sum()} из {len(mask)} точек")

        return mask

    def get_mesh_statistics(self) -> dict:
        """
        Возвращает статистику по mesh'у.

        Returns:
            Словарь со статистикой mesh'а
        """
        if self.mesh is None or self.vertices is None:
            return {}

        return {
            'num_vertices': len(self.vertices),
            'num_triangles': len(self.mesh.triangles),
            'z_min': self.vertices[:, 2].min(),
            'z_max': self.vertices[:, 2].max(),
            'z_mean': self.vertices[:, 2].mean(),
            'z_std': self.vertices[:, 2].std(),
            'x_extent': self.vertices[:, 0].max() - self.vertices[:, 0].min(),
            'y_extent': self.vertices[:, 1].max() - self.vertices[:, 1].min(),
            'z_extent': self.vertices[:, 2].max() - self.vertices[:, 2].min(),
        }
