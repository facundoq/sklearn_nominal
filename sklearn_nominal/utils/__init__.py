# Authors: facundoq
# License: BSD 3 clause

import pandas as pd


def read_regression_dataset(url: str):
    df = pd.read_csv(url)
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return x, y


def read_golf_regression_dataset():
    # dataset_name = "golf_classification"
    url = "https://raw.githubusercontent.com/facundoq/facundoq.github.io/refs/>>> heads/master/datasets/regression/golf_regression_nominal.csv"
    return read_regression_dataset(url)
