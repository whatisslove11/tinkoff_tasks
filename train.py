import difflib
import re
from typing import Optional, Dict, List
from catboost import CatBoostRegressor, Pool
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from sentence_transformers import SentenceTransformer
from os import listdir
import ast, astunparse
import argparse
import pickle
from tqdm import tqdm


class Regressor():
    def __init__(self,
                 learning_rate: Optional[float] = 0.08705189500596719,
                 iterations: Optional[int] = 7216,
                 depth: Optional[int] = 8,
                 l2_leaf_reg: Optional[float] = 16.791048921323583,
                 logging_level: Optional[str] = "Silent",
                 task_type: Optional[str] = "GPU"
                 ):
        self.model = CatBoostRegressor(learning_rate=learning_rate,
                                       iterations=iterations,
                                       depth=depth,
                                       l2_leaf_reg=l2_leaf_reg,
                                       random_seed=42,
                                       logging_level=logging_level,
                                       task_type=task_type
                                       )

    def fit(self, train_data: pd.DataFrame) -> "Regressor":
        """
        Описание входных данных:

        :param train_data: датафрейм pandas, в моем случае 187к строк, 30 features (метод составления выше)
        Подается сразу целиком, модель сама удаляет ненужные столбцы
        """
        X, y = train_data.drop(columns=['similarity', 'name_file', 'name_plagiat']), \
               np.log1p(train_data['similarity'] * 100)

        # train_pool = Pool(X, y)
        self.model.fit(X, y)
        return self

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        Описание входных данных:

        :param test_data: датафрейм pandas, такой же, как и на вход в метод fit (составлен аналогично, только без target столбца)
        :return: float, предсказание модели (у данных был Long tail, я исправлял положение при помощи логарифмирования)
        """
        test_pool = Pool(test_data.drop(columns=['similarity', 'name_file', 'name_plagiat']))

        pred = self.model.predict(test_pool)
        pred = np.expm1(pred) / 100

        return pred


class PretrainedModel():
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def predict(self, a: str, b: str) -> float:
        """
        Описание входных данных:

        :param a: str, первый текст для сравнения
        :param b: str, второй текст для сравнения

        Оба текста могут подаваться не нормализованными (но у нас так не будет)
        (как оказалось, модели все равно на удаление комментариев, замены узлов на одинаковое имя и т.д.)

        :return: float, cosine similarity двух текстов
        """
        a = self.model.encode(a)
        b = self.model.encode(b)

        return dot(a, b) / (norm(a) * norm(b))


    """
    Код получения словаря с количеством определенных конструкций в файле
    Да, код просто огонь, извините(
    """
def count_constuctions(file_name: str) -> Dict[str, int]:
    with open(file_name, encoding="utf8") as f:
        src = f.read()
    cfg = {"if": 0,
           "cycles": 0,
           "class": 0,
           "def": 0,
           "module": 0,
           "args": 0,
           "name": 0,
           "count": 0,
           "constructions": 0,
           "operations": 0,
           "constants": 0,
           "call_functions": 0,
           "attributes": 0}

    tree = ast.parse(src)
    for node in ast.walk(tree):
        cfg["count"] += 1
        if isinstance(node, (ast.If, ast.IfExp)):
            cfg["if"] += 1
        elif isinstance(node, (ast.For, ast.While)):
            cfg["cycles"] += 1
        elif isinstance(node, ast.ClassDef):
            cfg["class"] += 1
        elif isinstance(node, (ast.FunctionDef, ast.Lambda)):
            cfg["def"] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            cfg["module"] += 1
        elif isinstance(node, ast.arg):
            cfg["args"] += 1
        elif isinstance(node, ast.Name):
            cfg["name"] += 1
        elif isinstance(node, (ast.List, ast.Dict, ast.Tuple, ast.Set, ast.Slice)):
            cfg["constructions"] += 1
        elif isinstance(node, (ast.Return, ast.Yield, ast.Call)):
            cfg["call_functions"] += 1
        elif isinstance(node, (ast.Expr, ast.UnaryOp, ast.BinOp, ast.BoolOp, ast.Compare)):
            cfg["operations"] += 1
        elif isinstance(node, ast.Constant):
            cfg["constants"] += 1
        elif isinstance(node, ast.Attribute):
            cfg["attributes"] += 1

    return cfg

#получение всех файлов по пути
def get_names(path: str) -> List[str]:
    return [f for f in listdir(path)]

def get_length_file(file_name: str) -> int:
    count_lines = 0
    with open(file_name, encoding = "utf8") as f:
        lines = astunparse.unparse(ast.parse(f.read())).split('\n')
        for line in lines:
            if line.lstrip()[:1] not in ("'", '"'):
                count_lines+=1
    return count_lines

def normalize_text(file_text: str) -> str:
    tree = ast.parse(file_text)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            node.id = 'x'
    new_tree = ast.unparse(tree)
    new_tree = re.sub('#.*', '', new_tree, len(new_tree))
    new_tree = re.sub('\n', '_n_', new_tree, len(new_tree))
    new_tree = re.sub('""".*"""', '', new_tree, len(new_tree))
    return re.sub('_n_', '\n', new_tree, len(new_tree))

def similarity(f1: str, f2: str) -> float:
    return difflib.SequenceMatcher(None, f1.lower(), f2.lower()).ratio()


def create_dataframe(file: str, plag1: str, plag2: str) -> pd.DataFrame:

    files = get_names(file)
    files_plag = get_names(plag1)
    files_plag2 = get_names(plag2)

    list_files = []
    list_plag = []
    list_similarity = []

    list_files_if = []
    list_files_cycles = []
    list_files_class = []
    list_files_def = []
    list_files_module = []
    list_files_args = []
    list_files_name = []
    list_files_count = []
    list_files_constructions = []
    list_files_operations = []
    list_files_constants = []
    list_files_call_functions = []
    list_files_attributes = []
    list_files_length = []

    list_plag_if = []
    list_plag_cycles = []
    list_plag_class = []
    list_plag_def = []
    list_plag_module = []
    list_plag_args = []
    list_plag_name = []
    list_plag_count = []
    list_plag_constructions = []
    list_plag_operations = []
    list_plag_constants = []
    list_plag_call_functions = []
    list_plag_attributes = []
    list_plag_length = []

    for f in tqdm(files):
        f_file = f"{file}/{f}"
        normalized_file = normalize_text(f_file)

        for f1 in (files_plag):
            f_file_1 = f"{plag1}/{f1}"
            normalized_plag1 = normalize_text(f_file_1)

            list_files.append(f_file)
            list_plag.append(f_file_1)

            list_similarity.append(similarity(
                normalized_file,
                normalized_plag1
            ))

            list_plag_if.append(count_constuctions(f_file_1)["if"])
            list_plag_cycles.append(count_constuctions(f_file_1)["cycles"])
            list_plag_class.append(count_constuctions(f_file_1)["class"])
            list_plag_def.append(count_constuctions(f_file_1)["def"])
            list_plag_module.append(count_constuctions(f_file_1)["module"])
            list_plag_args.append(count_constuctions(f_file_1)["args"])
            list_plag_name.append(count_constuctions(f_file_1)["name"])
            list_plag_count.append(count_constuctions(f_file_1)["count"])
            list_plag_constructions.append(count_constuctions(f_file_1)["constructions"])
            list_plag_length.append(get_length_file(f_file_1))
            list_plag_operations.append(count_constuctions(f_file_1)["operations"])
            list_plag_constants.append(count_constuctions(f_file_1)["constants"])
            list_plag_call_functions.append(count_constuctions(f_file_1)["call_functions"])
            list_plag_attributes.append(count_constuctions(f_file_1)["attributes"])

            list_files_if.append(count_constuctions(f_file)["if"])
            list_files_cycles.append(count_constuctions(f_file)["cycles"])
            list_files_class.append(count_constuctions(f_file)["class"])
            list_files_def.append(count_constuctions(f_file)["def"])
            list_files_module.append(count_constuctions(f_file)["module"])
            list_files_args.append(count_constuctions(f_file)["args"])
            list_files_name.append(count_constuctions(f_file)["name"])
            list_files_count.append(count_constuctions(f_file)["count"])
            list_files_constructions.append(count_constuctions(f_file)["constructions"])
            list_files_length.append(get_length_file(f_file))
            list_files_operations.append(count_constuctions(f_file)["operations"])
            list_files_constants.append(count_constuctions(f_file)["constants"])
            list_files_call_functions.append(count_constuctions(f_file)["call_functions"])
            list_files_attributes.append(count_constuctions(f_file)["attributes"])
        for f2 in files_plag2:
            f_file_2 = f"{plag2}/{f2}"
            normalized_plag2 = normalize_text(f_file_2)

            list_files.append(f_file)
            list_plag.append(f_file_2)

            list_similarity.append(similarity(
                normalized_file,
                normalized_plag2
            ))

            list_plag_if.append(count_constuctions(f_file_2)["if"])
            list_plag_cycles.append(count_constuctions(f_file_2)["cycles"])
            list_plag_class.append(count_constuctions(f_file_2)["class"])
            list_plag_def.append(count_constuctions(f_file_2)["def"])
            list_plag_module.append(count_constuctions(f_file_2)["module"])
            list_plag_args.append(count_constuctions(f_file_2)["args"])
            list_plag_name.append(count_constuctions(f_file_2)["name"])
            list_plag_count.append(count_constuctions(f_file_2)["count"])
            list_plag_constructions.append(count_constuctions(f_file_2)["constructions"])
            list_plag_length.append(get_length_file(f_file_2))
            list_plag_operations.append(count_constuctions(f_file_2)["operations"])
            list_plag_constants.append(count_constuctions(f_file_2)["constants"])
            list_plag_call_functions.append(count_constuctions(f_file_2)["call_functions"])
            list_plag_attributes.append(count_constuctions(f_file_2)["attributes"])

            list_files_if.append(count_constuctions(f_file)["if"])
            list_files_cycles.append(count_constuctions(f_file)["cycles"])
            list_files_class.append(count_constuctions(f_file)["class"])
            list_files_def.append(count_constuctions(f_file)["def"])
            list_files_module.append(count_constuctions(f_file)["module"])
            list_files_args.append(count_constuctions(f_file)["args"])
            list_files_name.append(count_constuctions(f_file)["name"])
            list_files_count.append(count_constuctions(f_file)["count"])
            list_files_constructions.append(count_constuctions(f_file)["constructions"])
            list_files_length.append(get_length_file(f_file))
            list_files_operations.append(count_constuctions(f_file)["operations"])
            list_files_constants.append(count_constuctions(f_file)["constants"])
            list_files_call_functions.append(count_constuctions(f_file)["call_functions"])
            list_files_attributes.append(count_constuctions(f_file)["attributes"])

    df_train = pd.DataFrame(
        {'name_file': list_files, 'name_plagiat': list_plag,

         "count_if_in_not_plagiat": list_files_if,
         "count_cycles_in_not_plagiat": list_files_cycles, "count_class_in_not_plagiat": list_files_class,
         "count_def_in_not_plagiat": list_files_def, "count_module_in_not_plagiat": list_files_module,
         "count_args_in_not_plagiat": list_files_args, "count_name_in_not_plagiat": list_files_name,
         "count_count_in_not_plagiat": list_files_count, "count_length_in_not_plagiat": list_files_length,
         "count_constructions_in_not_plagiat": list_files_constructions,
         "count_operations_in_not_plagiat": list_files_operations,
         "count_constants_in_not_plagiat": list_files_constants,
         "count_call_functions_in_not_plagiat": list_files_call_functions,
         "count_attributes_in_not_plagiat": list_files_attributes,

         "count_if_in_plagiat": list_plag_if,
         "count_cycles_in_plagiat": list_plag_cycles, "count_class_in_plagiat": list_plag_class,
         "count_def_in_plagiat": list_plag_def, "count_module_in_plagiat": list_plag_module,
         "count_args_in_plagiat": list_plag_args, "count_name_in_plagiat": list_plag_name,
         "count_count_in_plagiat": list_plag_count, "count_length_in_plagiat": list_plag_length,
         "count_constructions_in_plagiat": list_plag_constructions, "count_operations_in_plagiat": list_plag_operations,
         "count_constants_in_plagiat": list_plag_constants, "count_call_functions_in_plagiat": list_plag_call_functions,
         "count_attributes_in_plagiat": list_plag_attributes,

         'similarity': list_similarity}
    )

    df_train["mean_name_in_not_plagiat"] = df_train["count_name_in_not_plagiat"] / df_train["count_length_in_not_plagiat"]
    df_train["mean_name_in_plagiat"] = df_train["count_name_in_plagiat"] / df_train["count_length_in_plagiat"]

    df_train["mean_count_in_not_plagiat"] = df_train["count_count_in_not_plagiat"] / df_train["count_length_in_not_plagiat"]
    df_train["mean_count_in_plagiat"] = df_train["count_count_in_plagiat"] / df_train["count_length_in_plagiat"]

    return df_train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=str)
    parser.add_argument('plagiat1', type=str)
    parser.add_argument('plagiat2', type=str)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    df = create_dataframe(args.files, args.plagiat1, args.plagiat2)

    model = Regressor()
    model.fit(df)

    pretrained_model = PretrainedModel()

    with open(args.model, "wb") as f:
        pickle.dump(model, f)

    with open("pretrained_model.pkl", "wb") as f1:
        pickle.dump(pretrained_model, f1)

if __name__ == '__main__':
    main()
