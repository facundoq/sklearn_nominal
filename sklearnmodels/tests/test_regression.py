import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
import sklearn.datasets
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from sklearnmodels import tree
import sklearn.tree
from sklearn.model_selection import train_test_split
import pytest

from sklearnmodels import SKLearnRegressionTree


def read_regression_dataset(path: Path):
    df = pd.read_csv(path)
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1].to_numpy().reshape(-1, 1)
    return x, y


def get_nominal_tree_regressor(x: pd.DataFrame, y: np.ndarray):
    n, m = x.shape
    max_height = min(max(int(np.log(m) * 3), 5), 30)
    min_samples_leaf = max(10, int(n * (0.05 / y.std())))
    min_samples_split = min_samples_leaf
    min_error_improvement = 0.05 * y.std()

    return SKLearnRegressionTree(
        criterion="std",
        max_depth=max_height,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        min_error_decrease=min_error_improvement,
        splitter=4,
    )


def train_test_classification_model(model_name: str, model_generator, dataset: Path):
    dataset_name = dataset.name.split(".")[0]
    x, y = read_regression_dataset(dataset)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, shuffle=True, random_state=0
    )
    model = model_generator(x_train, y_train)
    model.fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    score_train = mean_absolute_error(y_train, y_pred_train)
    y_pred_test = model.predict(x_test)
    score_test = mean_absolute_error(y_test, y_pred_test)
    return {
        "Model": model_name,
        "Dataset": dataset_name,
        "Train": score_train,
        "Test": score_test,
    }


def get_sklearn_pipeline(x: pd.DataFrame, model):
    numeric_features = x.select_dtypes(include=["int64", "float64"]).columns
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_features = x.select_dtypes(exclude=["int64", "float64"]).columns
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])


def get_sklearn_tree(x: pd.DataFrame, y: np.ndarray):
    n, m = x.shape
    max_height = min(max(int(np.log(m) * 3), 5), 30)
    min_samples_leaf = max(10, int(n * (0.05 / y.std())))
    min_samples_split = min_samples_leaf
    min_error_improvement = 0.05 * y.std()
    model = sklearn.tree.DecisionTreeRegressor(
        max_depth=max_height,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        min_impurity_decrease=min_error_improvement,
        criterion="squared_error",
    )
    return get_sklearn_pipeline(x, model)


path = Path("datasets/regression")
dataset_names = [
    "golf_regression_nominal.csv",
    "study_regression_small.csv",
    "study_regression_2d_small.csv",
    "who_no_missing_numeric.csv",
]


def test_performance_similar_sklearn(at_most_percent=1.5, dataset_names=dataset_names):

    datasets = [path / name for name in dataset_names]
    nominal_results_all = []
    numeric_results_all = []
    for dataset in tqdm(datasets, desc=f"Datasets"):
        nominal_results = train_test_classification_model(
            "NominalTree", get_nominal_tree_regressor, dataset
        )
        numeric_results = train_test_classification_model(
            "SklearnTree", get_sklearn_tree, dataset
        )
        for set in ["Train", "Test"]:
            numeric = numeric_results[set]
            nominal = nominal_results[set]
            percent = nominal / numeric
            assert (
                percent <= at_most_percent
            ), f"{set} score of nominal tree ({nominal:.2f}) should be at most {at_most_percent*100:.2f}% of sklearn.tree ({numeric:.2f}) on dataset {nominal_results["Dataset"]}, was {percent*100:.2f}% instead."
        nominal_results_all.append(nominal_results)
        numeric_results_all.append(numeric_results)
    print(pd.DataFrame.from_records(nominal_results_all))
    print(pd.DataFrame.from_records(numeric_results_all))


if __name__ == "__main__":
    test_performance_similar_sklearn()
