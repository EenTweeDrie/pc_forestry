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

    def __init__(self, base_path: str, file_name: str, intensity_cuts: list[int]) -> None:
        self.base_path = base_path
        self.file_name = file_name
        self.intensity_cuts = intensity_cuts
        self.params: Dict[str, Any] = {}
        self.mesh_name: str | None = None
        self.path_manager = PathManager().set_base_dir(base_path)

    def set_params(self, params: Dict[str, Any]) -> "MultiCoordinatesPipeline":
        """Устанавливает базовые параметры для всех пайплайнов."""
        self.params = dict(params)
        return self

    def set_mesh(self, mesh_name: str) -> "MultiCoordinatesPipeline":
        """Устанавливает имя файла меша."""
        self.mesh_name = mesh_name
        return self

    def run(self, force_cut: bool = True, force_cells: bool = True, force_stumps: bool = True) -> None:
        """
        Запускает полный процесс обработки для всех указанных intensity_cuts,
        включая запуск отдельных пайплайнов, объединение и очистку результатов.
        """
        print(f"Запуск обработки для intensity_cuts: {self.intensity_cuts}")
        stumps_csv_paths = []
        for intensity in self.intensity_cuts:
            print(f"\n--- Обработка для intensity_cut = {intensity} ---")
            current_params = self.params.copy()
            current_params['intensity_cut'] = intensity

            cp = CoordinatesPipeline(self.base_path, self.file_name).set_params(current_params)

            if self.mesh_name:
                cp.set_mesh(self.mesh_name)
                cp.cut_mesh_data(force=force_cut)
            else:
                # Предполагаем, что если меш не задан, используется cut_slice_data
                cp.cut_slice_data()

            cp.make_cells(force=force_cells).make_stumps(force=force_stumps)

            stumps_csv_path = os.path.join(self.path_manager.get_stumps_dir(intensity), f'stumps_{intensity}.csv')
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
            print('3', file_path)
            splt_fn = os.path.basename(file_path).split('_')[-1].split('.')[0]
            print('1', splt_fn)

            current_df = pd.read_csv(file_path, delimiter=";")

            if i == 0:
                df = current_df
                names_col = ["Name_stump_" + splt_fn, "X", "Y", "Diameter_" + splt_fn]
                df.columns = names_col
                continue

            iter_count += 1
            prev_names_col = names_col.copy()
            names_col.insert(iter_count, "Name_stump_" + splt_fn)
            names_col.insert(len(names_col) - 1, "Diameter_" + splt_fn)

            file1 = df
            file2 = current_df
            file2.columns = ["Name_stump_" + splt_fn, "X", "Y", "Diameter_" + splt_fn]

            df = self._merge_step(file1, file2, iter_count, names_col, prev_names_col)

        return df if df is not None else pd.DataFrame()

    @staticmethod
    def _merge_step(df1: pd.DataFrame, df2: pd.DataFrame, iter_val: int, new_cols: list, old_cols: list) -> pd.DataFrame:
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

        n = len(self.intensity_cuts)
        path_merged_stumps = os.path.join(self.base_path, "merged_stumps")
        os.makedirs(path_merged_stumps, exist_ok=True)

        # Колонки с именами файлов пней (например, Name_stump_7000)
        name_columns = [col for col in df.columns if col.startswith('Name_stump_')]

        initial_labels = np.full((df.shape[0], 1), -1, dtype=int)
        new_label_cols = []

        for i in tqdm(range(n), desc="Processing intensity levels"):
            col_name = name_columns[i]
            intensity = self.intensity_cuts[i]
            labels = []

            new_label_cols.append("Labels_" + str(intensity))
            path_int = self.path_manager.get_stumps_dir(intensity)

            for j in tqdm(range(df.shape[0]), desc=f"Predicting for int={intensity}", leave=False):
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
