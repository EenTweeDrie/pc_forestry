import numpy as np
from matplotlib import cm


def apply_color_palette(values, palette_name: str = 'Grey'):
    """Применяет цветовую палитру к нормализованным значениям."""
    colors = np.zeros((len(values), 3))

    if palette_name == "Blue > Green > Yellow > Red":
        # Создаем кастомную палитру: синий -> зеленый -> желтый -> красный
        colors[:, 0] = np.where(values < 0.5, 0, 2 * values - 1)  # красный компонент
        colors[:, 1] = np.where(values < 0.5, 2 * values, 2 - 2 * values)  # зеленый компонент
        colors[:, 2] = np.where(values < 0.5, 1 - 2 * values, 0)  # синий компонент
    elif palette_name == "Grey":
        colors[:, 0] = values  # r
        colors[:, 1] = values  # g
        colors[:, 2] = values  # b
    elif palette_name == "Viridis":
        colormap = cm.viridis(values)
        colors = colormap[:, :3]
    elif palette_name == "Brown > Yellow":
        # Коричневый -> желтый
        colors[:, 0] = 0.4 + 0.6 * values  # r: от коричневого к желтому
        colors[:, 1] = 0.2 + 0.8 * values  # g: от коричневого к желтому
        colors[:, 2] = 0.1 * (1 - values)  # b: убираем синий
    elif palette_name == "Yellow > Brown":
        # Желтый -> коричневый (обратный)
        colors[:, 0] = 1.0 - 0.6 * values  # r: от желтого к коричневому
        colors[:, 1] = 1.0 - 0.8 * values  # g: от желтого к коричневому
        colors[:, 2] = 0.1 * values        # b: добавляем синий
    elif palette_name == "Topo landserf":
        # Топографическая палитра: синий -> зеленый -> коричневый -> белый
        if len(values) > 0:
            colors[:, 0] = np.where(
                values < 0.33,
                0.2,
                np.where(
                    values < 0.66,
                    0.4 + 0.6 * (values - 0.33) / 0.33,
                    0.8 + 0.2 * (values - 0.66) / 0.34
                )
            )
            colors[:, 1] = np.where(
                values < 0.33,
                0.4 + 0.6 * values / 0.33,
                np.where(
                    values < 0.66,
                    0.8 - 0.4 * (values - 0.33) / 0.33,
                    0.6 + 0.4 * (values - 0.66) / 0.34
                )
            )
            colors[:, 2] = np.where(
                values < 0.33,
                0.8 - 0.6 * values / 0.33,
                np.where(
                    values < 0.66,
                    0.2,
                    0.2 + 0.8 * (values - 0.66) / 0.34
                )
            )
    elif palette_name == "High contrast":
        # Высококонтрастная палитра
        colors[:, 0] = np.where(values < 0.5, 0, 1)  # черный/красный
        colors[:, 1] = np.where(values < 0.5, values * 2, 1 - (values - 0.5) * 2)  # градиент зеленого
        colors[:, 2] = np.where(values < 0.5, 1 - values * 2, 0)  # синий/черный
    elif palette_name == "Cividis":
        colormap = cm.cividis(values)
        colors = colormap[:, :3]
    elif palette_name == "Blue > White > Red":
        # Синий -> белый -> красный (coolwarm style)
        colormap = cm.coolwarm(values)
        colors = colormap[:, :3]
    elif palette_name == "Red > Yellow":
        # Красный -> желтый
        colors[:, 0] = 1.0  # r всегда максимальный
        colors[:, 1] = values  # g растет от 0 до 1
        colors[:, 2] = 0.0  # b всегда 0
    else:
        # По умолчанию серый
        colors[:, 0] = values
        colors[:, 1] = values
        colors[:, 2] = values

    return colors
