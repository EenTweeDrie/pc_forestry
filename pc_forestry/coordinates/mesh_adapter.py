import numpy as np
import open3d as o3d
from typing import Tuple, Optional
from loguru import logger
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
        self.scene = None
        self._load_mesh()

    def _load_mesh(self) -> None:
        """Загружает mesh из STL файла."""
        if not os.path.exists(self.mesh_path):
            raise FileNotFoundError(f"STL файл не найден: {self.mesh_path}")

        try:
            self.mesh = o3d.io.read_triangle_mesh(self.mesh_path)
            if not self.mesh.has_vertices():
                raise ValueError("Mesh не содержит вершин")

            self.vertices = np.asarray(self.mesh.vertices)

            logger.info("Создание RaycastingScene для ускорения вычислений...")
            self.scene = o3d.t.geometry.RaycastingScene()
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
            self.scene.add_triangles(mesh_t)
            logger.info("RaycastingScene успешно создана.")

            logger.info(f"Загружен mesh с {len(self.vertices)} вершинами из {self.mesh_path}")

        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке mesh файла: {e}")

    def create_relative_height_slice(self, points: np.ndarray, height_from: float, height_to: float) -> np.ndarray:
        """
        Создает маску для выбора кусочка облака точек в диапазоне высот относительно mesh'а,
        используя трассировку лучей для определения высоты mesh'а.

        Args:
            points: Массив точек облака (N x 3)
            height_from: Нижняя граница относительно mesh'а (в метрах)
            height_to: Верхняя граница относительно mesh'а (в метрах)

        Returns:
            Булева маска для фильтрации точек
        """
        if height_from >= height_to:
            raise ValueError("height_from должно быть меньше height_to")

        if self.scene is None or self.vertices is None:
            raise RuntimeError("Mesh и RaycastingScene не загружены")

        if len(points) == 0:
            logger.warning("Передан пустой массив точек в create_relative_height_slice")
            return np.array([], dtype=bool)

        # Подготовка лучей для трассировки
        # Начальные точки лучей устанавливаем выше самой высокой точки меша
        z_max = self.vertices[:, 2].max()
        rays_origin = np.copy(points[:, :3])
        rays_origin[:, 2] = z_max + 1.0

        # Направление лучей - строго вниз
        rays_dir = np.array([0.0, 0.0, -1.0])
        rays_dir_tiled = np.tile(rays_dir, (len(points), 1))

        rays = np.hstack([rays_origin, rays_dir_tiled]).astype(np.float32)

        # Выполняем трассировку лучей
        ans = self.scene.cast_rays(o3d.core.Tensor(rays))

        t_hit = ans['t_hit'].numpy()

        # Вычисляем высоту меша в точках пересечения
        # Если пересечения нет, t_hit будет inf, и высота будет -inf
        mesh_heights = rays_origin[:, 2] - t_hit

        # Определяем границы среза для каждой точки
        absolute_low = mesh_heights + height_from
        absolute_high = mesh_heights + height_to

        # Создаем маску: точка должна быть выше нижней границы и ниже или равна верхней
        mask = (points[:, 2] > absolute_low) & (points[:, 2] <= absolute_high)

        # Точки, для которых не найдено пересечение с мешем, не включаем в маску
        no_hit_mask = np.isinf(t_hit)
        mask[no_hit_mask] = False

        points_selected = mask.sum()
        logger.info(f"Создан срез [{height_from:.2f}, {height_to:.2f}]м от mesh'а: {points_selected} из {len(mask)} точек")

        if points_selected == 0:
            logger.warning(f"ВНИМАНИЕ: После применения фильтра высот не осталось точек! "
                           f"Диапазон высот: [{height_from:.2f}, {height_to:.2f}]м")
            logger.warning(f"Статистика исходных точек - Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]м")
            if self.vertices is not None:
                logger.warning(f"Статистика mesh - Z: [{self.vertices[:, 2].min():.2f}, {self.vertices[:, 2].max():.2f}]м")

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
