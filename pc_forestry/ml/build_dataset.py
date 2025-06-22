import os
import argparse
import pandas as pd
from tqdm import tqdm
import glob
from loguru import logger

try:
    from pc_forestry.pcd.TREE import TREE
    from pc_forestry.pcd.VOXEL import VOXELGRID
except ImportError:
    # если скрипт запускается напрямую
    import sys
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../..')))
    from pc_forestry.pcd.TREE import TREE
    from pc_forestry.pcd.VOXEL import VOXELGRID


def process_tree_file(file_path, voxel_size=0.5):
    """
    Обрабатывает один файл с данными о дереве.
    """
    try:
        logger.info(f"Чтение файла: {file_path}")
        pc = TREE.read(file_path)

        logger.info("Сдвиг к нулю")
        pc.shift_to_zero()

        logger.info("Расчет освещенности")
        pc.calculate_illuminance()

        logger.info("Оценка нормалей")
        pc.estimate_normals()

        logger.info("Оценка координат")
        pc.estimate_coordinate()

        logger.info(f"Создание воксельной сетки с размером {voxel_size}")
        vg = VOXELGRID.create(pc, voxel_size, verbose=False)

        logger.info("Расчет расстояний до предыдущего слоя")
        vg.calculate_distances_to_previous_layer(pc.coordinate)

        logger.info("Расчет расстояний до координат")
        vg.calculate_distances_to_coordinate(pc.coordinate)

        logger.info("Получение нормализованного DataFrame")
        df = vg.normalized_df

        return df
    except Exception as e:
        logger.error(f"Ошибка при обработке файла {file_path}: {e}")
        return None


def build_dataset(input_dir, output_dir, dataset_type, voxel_size):
    """
    Создает датасет (train/val/test) из файлов деревьев.
    """
    individual_output_dir = os.path.join(
        output_dir, dataset_type, 'individual')
    combined_output_dir = os.path.join(output_dir, dataset_type)

    os.makedirs(individual_output_dir, exist_ok=True)
    os.makedirs(combined_output_dir, exist_ok=True)

    # Поиск файлов (предполагаем, что они .txt)
    search_pattern = os.path.join(input_dir, '*.las')
    file_paths = glob.glob(search_pattern)

    if not file_paths:
        logger.warning(f"Не найдено файлов .txt в директории: {input_dir}")
        return

    all_dfs = []

    for file_path in tqdm(file_paths, desc=f"Обработка {dataset_type} датасета"):
        df = process_tree_file(file_path, voxel_size)
        if df is not None:
            # Добавляем столбец с именем исходного файла
            df['source_file'] = os.path.basename(file_path)
            all_dfs.append(df)

            # Сохранение индивидуального файла
            output_filename = os.path.basename(
                file_path).replace('.las', '.csv')
            individual_save_path = os.path.join(
                individual_output_dir, output_filename)
            df.to_csv(individual_save_path, index=False, sep=';')
            logger.info(
                f"Индивидуальный DataFrame сохранен в: {individual_save_path}")

    if not all_dfs:
        logger.error(
            "Не удалось обработать ни одного файла. Сборный датасет не будет создан.")
        return

    # Объединение и сохранение полного датасета
    logger.info("Объединение всех DataFrame'ов...")
    combined_df = pd.concat(all_dfs, ignore_index=True)

    combined_filename = f"{dataset_type}_dataset.csv"
    combined_save_path = os.path.join(combined_output_dir, combined_filename)
    combined_df.to_csv(combined_save_path, index=False)

    logger.info(f"Сборный датасет сохранен в: {combined_save_path}")
    logger.info("Сборка датасета завершена.")


def main():
    parser = argparse.ArgumentParser(
        description="Сборка датасета для обучения классификатора деревьев.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Папка с исходными файлами деревьев (например, .txt)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Папка для сохранения обработанных датасетов')
    parser.add_argument('--dataset_type', type=str, required=True, choices=['train', 'val', 'test'],
                        help='Тип создаваемого датасета (train, val, or test)')
    parser.add_argument('--voxel_size', type=float, default=0.5,
                        help='Размер вокселя для создания сетки (по умолчанию: 0.5)')

    args = parser.parse_args()

    build_dataset(args.input_dir, args.output_dir,
                  args.dataset_type, args.voxel_size)


if __name__ == '__main__':
    main()
