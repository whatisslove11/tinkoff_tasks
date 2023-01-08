import ast
import pickle
import argparse
from typing import Dict

import astunparse
import numpy as np
import re
import pandas as pd

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

def levenshtein(seq1: str, seq2: str) -> int:

    size_x = len(seq1) + 1
    size_y = len(seq2) + 1

    matrix = np.zeros((size_x, size_y))
    for x in np.arange(size_x):
        matrix[x, 0] = x
    for y in np.arange(size_y):
        matrix[0, y] = y

    for x in np.arange(1, size_x):
        for y in np.arange(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                matrix[x - 1, y] + 1,
                matrix[x - 1, y - 1],
                matrix[x, y - 1] + 1
            )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    return matrix[size_x - 1, size_y - 1]

def get_length_file(file_name: str) -> int:
    count_lines = 0
    with open(file_name, encoding = "utf8") as f:
        lines = astunparse.unparse(ast.parse(f.read())).split('\n')
        for line in lines:
            if line.lstrip()[:1] not in ("'", '"'):
                count_lines+=1
    return count_lines

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

def create_dataframe(file: str, plag1: str) -> pd.DataFrame:

    cfg1, cfg2 = count_constuctions(file), count_constuctions(plag1)
    list_pd = []

    list_pd.append(file)
    list_pd.append(plag1)

    for key in cfg1.keys():
        list_pd.append(cfg1[key])

    for key in cfg2.keys():
        list_pd.append(cfg2[key])

    columns = ['name_file', 'name_plagiat', 'count_if_in_not_plagiat', 'count_cycles_in_not_plagiat', 'count_class_in_not_plagiat',
               'count_def_in_not_plagiat', 'count_module_in_not_plagiat', 'count_args_in_not_plagiat',
               'count_name_in_not_plagiat', 'count_count_in_not_plagiat', 'count_length_in_not_plagiat', 'count_constructions_in_not_plagiat',
               'count_operations_in_not_plagiat', 'count_constants_in_not_plagiat', 'count_call_functions_in_not_plagiat',
               'count_attributes_in_not_plagiat',
               'count_if_in_plagiat', 'count_cycles_in_plagiat', 'count_class_in_plagiat', 'count_def_in_plagiat',
               'count_module_in_plagiat', 'count_count_in_plagiat', 'count_length_in_plagiat', 'count_constructions_in_plagiat',
               'count_operations_in_plagiat', 'count_constants_in_plagiat', 'count_call_functions_in_plagiat',
               'count_attributes_in_plagiat']

    df = pd.DataFrame(list_pd, columns=columns)

    df["mean_name_in_not_plagiat"] = df["count_name_in_not_plagiat"] / df["count_length_in_not_plagiat"]
    df["mean_name_in_plagiat"] = df["count_name_in_plagiat"] / df["count_length_in_plagiat"]

    df["mean_count_in_not_plagiat"] = df["count_count_in_not_plagiat"] / df["count_length_in_not_plagiat"]
    df["mean_count_in_plagiat"] = df["count_count_in_plagiat"] / df["count_length_in_plagiat"]

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('scores', type=str)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    with open("pretrained_model.pkl", "rb") as f:
        pretrained_model = pickle.load(f)

    with open(args.input, 'r') as f:
        for_check = list(f.readlines())

    ans = []

    for line in for_check:
        name_file, name_plagiat = line.split()
        file_test_1, file_test_2 = open(name_file, encoding="utf8").read(), open(name_file, encoding="utf8").read()

        f1 = normalize_text(file_test_1)
        f2 = normalize_text(file_test_2)

        pred_data = create_dataframe(name_file, name_plagiat)

        res1 = model.predict(pred_data)
        res2 = pretrained_model.predict(f1, f2)
        res3 = 1 - levenshtein(f1, f2) / max(len(f1), len(f2))

        ans.append(round((0.8*res3 + 0.1*res2 + 0.1*res1), 3))

    with open(args.output, 'w+') as f:
        for elem in ans:
            f.write(str(elem) + "\n")

if __name__ == '__main__':
    main()




