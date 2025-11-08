import os
import shutil
from typing import Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..path_manager import PathManager
from .pipeline import CoordinatesPipeline
from ..predict.models.pointnet2_cls_ssg import get_model
from ..predict.utils import pointcloud_utils as pcu
import torch
from ..utils.fps import farthest_point_sample
from ..pcd.PCD import PCD


def predict_cluster(cluster, device):
    """ predict the cluster """
    model_name = 'int0000_7000-512-rlish-s4762'
    model_path = os.path.join(os.path.dirname(__file__),
                              '..', 'predict', 'checkpoints', model_name, 'models', 'model.t7')
    species_names = ['Trunk', 'Not_Trunk']

    points = torch.Tensor([cluster]).to(device)
    centroids = farthest_point_sample(points, 512)
    pc_sampled = points[0][centroids[0]].cpu().detach().numpy()
    X_test = pcu.tree_normalize(np.array([pc_sampled]))

    int2name = {i: name for i, name in enumerate(species_names)}
    NUM_CLASSES = len(int2name)

    model = get_model(NUM_CLASSES, normal_channel=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.eval()
    data = torch.tensor(X_test, device=device)
    data = data.permute(0, 2, 1)

    with torch.no_grad():
        logits, _ = model(data)
        # Получаем предсказанный класс
        predicted_class = torch.argmax(logits, dim=1)

    return predicted_class.item()


class MultiCoordinatesPipeline:
    """
    Класс для оркестрации нескольких запусков CoordinatesPipeline с разными параметрами
    и последующего объединения и обработки результатов.
    """

    def __init__(self, base_path: str, file_path: str) -> None:
        self.base_path = base_path
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.params: Dict[str, Any] = {}
        self.mesh_path: str | None = None
        self.path_manager = PathManager().set_base_dir(base_path)
        self.stumps_df = None
        # Новый режим: список конфигов
        self.param_sets: list[Dict[str, Any]] | None = None
        # Маппинг stumps_id -> priority
        self.stumps_priority_map: Dict[str, int] = {}

    def set_params(self, params: Dict[str, Any]) -> "MultiCoordinatesPipeline":
        """Устанавливает базовые параметры для всех пайплайнов."""
        self.params = dict(params)
        return self

    def set_param_sets(self, param_sets: list[Dict[str, Any]]) -> "MultiCoordinatesPipeline":
        """Устанавливает список конфигов для последовательных запусков.
        Если задан, режим intensity_cuts игнорируется.
        """
        # Клонируем, чтобы не модифицировать входные структуры
        self.param_sets = [dict(p) for p in param_sets]
        return self

    def set_mesh(self, mesh_path: str) -> "MultiCoordinatesPipeline":
        """Устанавливает путь к файлу меша."""
        self.mesh_path = mesh_path
        return self

    def run(self, force_cut: bool = True, force_cells: bool = True, force_stumps: bool = True) -> None:
        """
        Запускает полный процесс обработки:
        - либо по self.param_sets (если задано),
        - либо по self.intensity_cuts (обратная совместимость).
        """
        stumps_csv_paths = []

        print(f"Запуск обработки для набора конфигов: {len(self.param_sets)} шт.")
        for idx, cfg in enumerate(self.param_sets):
            current_params = dict(self.params)
            current_params.update(cfg)
            # Генерируем stumps_id, если не задан явно: ключевые параметры, влияющие на результат
            if 'stumps_id' not in current_params:
                import uuid
                current_params['stumps_id'] = str(uuid.uuid4())[:7]

            stumps_id = current_params['stumps_id']
            # Сохраняем приоритет (по умолчанию используем порядок определения)
            self.stumps_priority_map[stumps_id] = int(cfg.get('priority', idx))
            print(f"\n--- Обработка конфига #{idx+1}: stumps_id={stumps_id} ---")

            cp = CoordinatesPipeline(self.base_path, self.file_path).set_params(current_params)

            if self.mesh_path:
                cp.set_mesh(self.mesh_path)
                cp.cut_mesh_data(force=force_cut)
            else:
                # Если меш не задан, предполагается другой способ нарезки (в проекте отключён)
                raise Exception("Mesh adapter обязателен для текущего режима")

            cp.make_cells(force=force_cells).make_stumps(force=force_stumps)

            stumps_csv_path = os.path.join(self.path_manager.get_stumps_dir(stumps_id), f'stumps_{stumps_id}.csv')
            stumps_csv_paths.append(stumps_csv_path)

        # Создаем coordinates_paths.txt для объединения
        coord_paths_txt = os.path.join(self.base_path, "coordinates_paths.txt")
        with open(coord_paths_txt, "w") as f:
            for path in stumps_csv_paths:
                f.write(path + "\n")
        print(f"\nФайл {coord_paths_txt} создан.")

        print("\n--- Запуск объединения координат ---")
        self._merge_coordinates()
        print("Объединение координат завершено.")

        print("\n--- Запуск очистки лишних пней ---")
        self._clear_excess_stumps()
        print("Очистка завершена.")

        print("\n--- Выбор пней по приоритету ---")
        self._select_stumps_by_priority()
        print("Выбор по приоритету завершён.")

    def filter_selected_by_labels(self, n_labels: int) -> None:
        """
        Фильтрует файлы в selected_stumps на основе количества положительных меток.
        Читает файл *_Clear_Excess.csv, находит столбцы Labels_*, и если сумма единиц
        в строке больше n_labels, копирует соответствующий файл из selected_stumps в
        новую папку selected_stumps_filtered.

        Соответствие строки и имени файла берётся из selected_stumps/selection_summary.csv
        по полю row -> row{row_idx:05d}_... .
        """
        clear_csv_path = os.path.join(self.base_path, self.file_name.partition('.')[0] + "_Clear_Excess.csv")
        if not os.path.exists(clear_csv_path):
            print(f"Файл с метками не найден: {clear_csv_path}")
            return

        df = pd.read_csv(clear_csv_path, delimiter=';')
        label_columns = [c for c in df.columns if c.startswith('Labels_')]
        if not label_columns:
            print("В таблице нет столбцов Labels_*")
            return

        summary_csv = os.path.join(self.base_path, 'selected_stumps', 'selection_summary.csv')
        if not os.path.exists(summary_csv):
            print(f"Не найдена сводка выбранных файлов: {summary_csv}")
            return

        summary = pd.read_csv(summary_csv, delimiter=';')
        # Быстрый доступ: row -> copied_as
        row_to_copied = {int(r['row']): r['copied_as'] for _, r in summary.iterrows()}

        src_dir = os.path.join(self.base_path, 'selected_stumps')
        dst_dir = os.path.join(self.base_path, 'selected_stumps_filtered')
        os.makedirs(dst_dir, exist_ok=True)

        copied = 0
        for row_idx in range(df.shape[0]):
            row_vals = df.loc[row_idx, label_columns]
            # Считаем единицы (Trunk == 1)
            try:
                ones = int((row_vals == 1).sum())
            except Exception:
                # На случай строковых типов
                ones = sum(1 for v in row_vals.values if str(v) == '1')

            if ones > n_labels:
                copied_name = row_to_copied.get(row_idx)
                if not copied_name:
                    continue
                src_path = os.path.join(src_dir, copied_name)
                dst_path = os.path.join(dst_dir, copied_name)
                try:
                    shutil.copy2(src_path, dst_path)
                    copied += 1
                except FileNotFoundError:
                    print(f"Не найден файл для копирования: {src_path}")

        print(f"Отфильтровано и скопировано файлов: {copied}")

    def _merge_coordinates(self) -> None:
        """Объединяет файлы с координатами пней."""
        df = self._init_merge_file()
        save_pth = self.file_name.partition('.')[0] + "_Coordinates_Merged.csv"
        save_pth = os.path.join(self.base_path, save_pth)
        df.to_csv(save_pth, index=False, sep=';')
        print(f"Объединенные координаты сохранены в: {save_pth}")

    def _init_merge_file(self) -> pd.DataFrame:
        """Инициализирует и выполняет процесс слияния файлов."""
        txt_path = os.path.join(self.base_path, "coordinates_paths.txt")
        with open(txt_path, "r") as file:
            paths = [line.strip() for line in file if line.strip()]

        df = None
        iter_count = 0
        names_col = []
        array = []

        for i, file_path in enumerate(paths):
            splt_fn = os.path.basename(file_path).split('_')[-1].split('.')[0]

            current_df = pd.read_csv(file_path, delimiter=";")

            if i == 0:
                df = current_df
                names_col = ["Name_stump_" + splt_fn, "X", "Y", "Diameter_" + splt_fn]
                df.columns = names_col
                continue

            iter_count += 1
            names_col.insert(iter_count, "Name_stump_" + splt_fn)
            names_col.insert(len(names_col) - 1, "Diameter_" + splt_fn)

            file1 = df
            file2 = current_df
            file2.columns = ["Name_stump_" + splt_fn, "X", "Y", "Diameter_" + splt_fn]

            df = self._merge_step(file1, file2, iter_count, names_col)

        return df if df is not None else pd.DataFrame()

    @staticmethod
    def _merge_step(df1: pd.DataFrame, df2: pd.DataFrame, iter_val: int, new_cols: list) -> pd.DataFrame:
        """Шаг слияния двух DataFrame."""
        eps = 0.25

        merged_data = []
        df2_unmatched = df2.copy()

        for _, row1 in df1.iterrows():
            match_found = False
            for _, row2 in df2_unmatched.iterrows():
                dist = np.linalg.norm(row1[['X', 'Y']].values.astype(float) - row2[['X', 'Y']].values.astype(float))
                if dist < eps:
                    new_row = row1.tolist()
                    new_row.insert(iter_val, row2.iloc[0])
                    new_row.insert(len(new_row)-1, row2.iloc[3])
                    merged_data.append(new_row)
                    df2_unmatched = df2_unmatched.drop(row2.name)
                    match_found = True
                    break
            if not match_found:
                new_row = row1.tolist()
                new_row.insert(iter_val, "File__Not__Found")
                new_row.insert(len(new_row)-1, 0.0)
                merged_data.append(new_row)

        for _, row2_unmatched in df2_unmatched.iterrows():
            new_row = ["File__Not__Found"] * (iter_val) + [row2_unmatched.iloc[0]] + \
                      [row2_unmatched.X, row2_unmatched.Y] + [0.0] * (iter_val) + [row2_unmatched.iloc[3]]
            # Ensure correct number of columns
            while len(new_row) < len(new_cols):
                new_row.insert(iter_val + 2, 0.0)  # Add missing diameter columns

            merged_data.append(new_row)

        df = pd.DataFrame(data=merged_data, columns=new_cols)
        df = df.dropna(subset=['X', 'Y'])
        df = df[(df.X != 'nan')]
        df = df[(df.Y != 'nan')]
        return df

    def _clear_excess_stumps(self) -> None:
        """
        Фильтрует пни на основе предсказаний модели, аналогично
        clear_excess_stumps.py.
        """
        merged_csv_path = os.path.join(self.base_path, self.file_name.partition('.')[0] + "_Coordinates_Merged.csv")
        df = pd.read_csv(merged_csv_path, delimiter=";")

        # Число наборов = по количеству стобцов Name_stump_*
        name_columns = [col for col in df.columns if col.startswith('Name_stump_')]
        n = len(name_columns)
        path_merged_stumps = os.path.join(self.base_path, "merged_stumps")
        os.makedirs(path_merged_stumps, exist_ok=True)

        initial_labels = np.full((df.shape[0], 1), -1, dtype=int)
        new_label_cols = []

        for i in tqdm(range(n), desc="Processing stumps sets"):
            col_name = name_columns[i]
            stumps_id = col_name.replace('Name_stump_', '')
            labels = []

            new_label_cols.append("Labels_" + str(stumps_id))
            path_int = self.path_manager.get_stumps_dir(stumps_id)

            for j in tqdm(range(df.shape[0]), desc=f"Predicting for id={stumps_id}", leave=False):
                value = df.at[j, col_name]
                if pd.notna(value) and value != "File__Not__Found":
                    path_file = os.path.join(path_int, value)
                    path_save = os.path.join(path_merged_stumps, value)
                    try:
                        shutil.copy2(path_file, path_save)
                        cluster = PCD.read(path_save)
                        device = cluster.device
                        label = predict_cluster(cluster.points, device)
                        labels.append(label)
                    except FileNotFoundError:
                        print(f"File not found: {path_file}")
                        labels.append(-3)
                else:
                    labels.append(-2)

            initial_labels = np.hstack([initial_labels, np.array(labels).reshape(-1, 1)])

        initial_labels = initial_labels[:, 1:]

        df_labels = pd.DataFrame(data=initial_labels, columns=new_label_cols)
        df_result = pd.concat([df, df_labels], axis=1)

        save_pth = self.file_name.partition('.')[0] + "_Clear_Excess.csv"
        save_pth = os.path.join(self.base_path, save_pth)
        df_result.to_csv(save_pth, index=False, sep=';')
        print(f"Результаты с метками сохранены в: {save_pth}")

    def _select_stumps_by_priority(self) -> None:
        """
        Выбирает из объединённой таблицы по каждой строке один файл пня на основе
        приоритета конфигураций и копирует в папку selected_stumps.

        Приоритет берётся из self.stumps_priority_map. Если приоритет не задан,
        используется порядок следования колонок в CSV.
        """
        merged_csv_path = os.path.join(self.base_path, self.file_name.partition('.')[0] + "_Coordinates_Merged.csv")
        df = pd.read_csv(merged_csv_path, delimiter=';')

        name_columns = [col for col in df.columns if col.startswith('Name_stump_')]
        if not name_columns:
            print("Нет колонок Name_stump_ для выбора по приоритету.")
            return

        # Подготовка порядка выбора по приоритетам
        stumps_ids = [col.replace('Name_stump_', '') for col in name_columns]
        # Словарь колонка -> (priority, original_index)
        priority_pairs = {}
        for idx, sid in enumerate(stumps_ids):
            priority_pairs[sid] = (self.stumps_priority_map.get(sid, 10**6), idx)

        # Сортировка: меньший priority выше, при равенстве — по порядку колонок
        sorted_ids = sorted(stumps_ids, key=lambda sid: (priority_pairs[sid][0], priority_pairs[sid][1]))

        # Карта id -> column name для быстрого доступа
        id_to_col = {sid: f"Name_stump_{sid}" for sid in stumps_ids}

        source_dir = os.path.join(self.base_path, 'merged_stumps')
        target_dir = os.path.join(self.base_path, 'selected_stumps')
        os.makedirs(target_dir, exist_ok=True)

        selections = []
        for row_idx in range(df.shape[0]):
            chosen_sid = None
            chosen_name = None
            for sid in sorted_ids:
                col = id_to_col[sid]
                value = df.at[row_idx, col] if col in df.columns else None
                if pd.isna(value) or value == "File__Not__Found":
                    continue
                chosen_sid = sid
                chosen_name = str(value)
                break

            if chosen_sid is None or chosen_name is None:
                continue

            src_path = os.path.join(source_dir, chosen_name)
            # Уникализируем имя файла по индексу строки, чтобы избежать коллизий
            base, ext = os.path.splitext(chosen_name)
            dst_name = f"row{row_idx:05d}_{base}{ext}"
            dst_path = os.path.join(target_dir, dst_name)
            try:
                shutil.copy2(src_path, dst_path)
                selections.append({
                    'row': row_idx,
                    'stumps_id': chosen_sid,
                    'file': chosen_name,
                    'copied_as': dst_name,
                })
            except FileNotFoundError:
                print(f"Не найден исходный файл для копирования: {src_path}")

        # Сохраняем сводку выбора
        if selections:
            summary_csv = os.path.join(self.base_path, 'selected_stumps', 'selection_summary.csv')
            pd.DataFrame(selections).to_csv(summary_csv, index=False, sep=';')
            print(f"Сводка выбора сохранена: {summary_csv}")
