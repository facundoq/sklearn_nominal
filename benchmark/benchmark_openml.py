import os
import subprocess
import time
import typing
from pathlib import Path

import cpuinfo
import lets_plot as lp

import numpy as np
import openml
import pandas as pd
import sklearn
from sklearn import base
import sklearn.impute
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm

from sklearnmodels import SKLearnClassificationTree

basepath = Path("benchmark/openml_cc18/")


def get_tree_parameters(x: pd.DataFrame, classes: int):
    n, m = x.shape
    max_height = min(max(int(np.log(m) * 3), 5), 30)
    min_samples_leaf = max(10, int(n * (0.05 / classes)))
    min_samples_split = min_samples_leaf
    min_error_improvement = 0.05 / classes
    return max_height, min_samples_leaf, min_samples_split, min_error_improvement


def get_nominal_tree(x: pd.DataFrame, classes: int):
    n, m = x.shape
    max_height, min_samples_leaf, min_samples_split, min_error_improvement = (
        get_tree_parameters(x, classes)
    )

    return SKLearnClassificationTree(
        criterion="entropy",
        max_depth=max_height,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        min_error_decrease=min_error_improvement,
        splitter=4,
    )


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
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    return pipeline


def get_sklearn_tree(x: pd.DataFrame, classes: int):
    max_height, min_samples_leaf, min_samples_split, min_error_improvement = (
        get_tree_parameters(x, classes)
    )
    model = sklearn.tree.DecisionTreeClassifier(
        max_depth=max_height,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        min_impurity_decrease=min_error_improvement,
    )
    pipeline = get_sklearn_pipeline(x, model)
    return pipeline


def reduce_numeric_features(x: pd.DataFrame, max_numeric_features):
    x_numeric = x.select_dtypes(include="number")
    x_non_numeric = x.select_dtypes(exclude="number")
    if len(x_numeric.columns) > max_numeric_features:
        pca = sklearn.decomposition.PCA(n_components=max_numeric_features)
        imputer = SimpleImputer(strategy="median")
        x_numeric = imputer.fit_transform(x_numeric)
        x_numeric_reduced = pca.fit_transform(x_numeric)
        x_numeric_reduced = pd.DataFrame(
            data=x_numeric_reduced,
            columns=[
                f"v_{i}={v:.2f}" for i, v in enumerate(pca.explained_variance_ratio_)
            ],
        )
        x = pd.concat([x_numeric_reduced, x_non_numeric], axis=1)
    return x


def benchmark(
    model_generator: typing.Callable,
    model_name: str,
    model_df: pd.DataFrame,
    model_table_path: Path,
) -> pd.DataFrame:
    benchmark_suite = openml.study.get_suite(
        "OpenML-CC18"
    )  # obtain the benchmark suite

    # print("Running", benchmark_suite)

    pbar = tqdm(total=len(benchmark_suite.tasks))
    for i, task_id in enumerate(benchmark_suite.tasks):  # iterate over all tasks
        task = openml.tasks.get_task(task_id)  # download the OpenML task
        dataset = task.get_dataset()
        pbar.update(1)
        # dont run again if a result already exists
        if not model_df.empty:
            previous_runs = model_df[
                (model_df["model"] == model_name)
                & (model_df["dataset"] == dataset.name)
            ]
            if len(previous_runs) > 0:
                continue

        x, y = task.get_X_and_y(dataset_format="dataframe")  # get the data
        x = reduce_numeric_features(x, 32)
        n, m = x.shape

        # print(f"Running {task} for dataset {dataset.name}")
        le = LabelEncoder().fit(y)
        y = le.transform(y)
        classes = len(np.unique(y))

        pbar.set_postfix_str(f"{dataset.name}: input={n}x{m} => {classes} classes")
        model = model_generator(x, classes)
        start = time.time_ns()
        model.fit(x, y)
        train_elapsed = (time.time_ns() - start) / 10e9
        start = time.time_ns()
        y_pred = model.predict(x)
        test_elapsed = (time.time_ns() - start) / 10e9
        acc = sklearn.metrics.accuracy_score(y, y_pred)
        result = pd.DataFrame.from_records(
            [
                {
                    "model": model_name,
                    "dataset": dataset.name,
                    "train_accuracy": acc,
                    "train_time": train_elapsed,
                    "test_time": test_elapsed,
                    "samples": n,
                    "features": m,
                    "classes": classes,
                }
            ]
        )
        model_df = pd.concat([model_df, result], axis=0)
        model_df.to_csv(model_table_path, index=False)

        if isinstance(model, SKLearnClassificationTree):
            image_folderpath = basepath / "trees"
            image_folderpath.mkdir(exist_ok=True)
            image_filepath = image_folderpath / f"{dataset.name}.svg"
            model.export_image(image_filepath, le.classes_)
    return model_df


paths = []


def save_plot(plot, filename: str):
    path = str((basepath / filename).absolute())
    lp.ggsave(plot, filename=path, w=8, h=4, unit="in", dpi=300)
    paths.append(filename)
    return filename


def plot_results(df: pd.DataFrame):
    df.sort_values(by="features")

    axis_font = lp.theme(axis_text=lp.element_text(size=7, angle=90))
    x_scale = lp.scale_x_discrete(lablim=10)
    common_options = lp.geom_line() + lp.geom_point()
    condition = df["model"] == "sklearn.tree"
    df_reference = df.loc[condition]
    df_others = df.loc[~condition].copy()
    mix = df_others.merge(
        df_reference, on="dataset", how="left", suffixes=(None, "_ref")
    )

    for y in ["train_accuracy", "train_time", "test_time"]:
        plot = (
            lp.ggplot(df, lp.aes(x="dataset", y=y, color="model"))
            + common_options
            + x_scale
            + axis_font
        )
        save_plot(plot, f"{y}.png")
    for x in ["samples", "features"]:
        for y in ["train_time", "test_time"]:
            plot = lp.ggplot(df, lp.aes(x=x, y=y, color="model")) + common_options
            save_plot(plot, f"{x}_{y}.png")
        aes_speedup = lp.ylim(0, 1.5) + lp.geom_hline(
            yintercept=1, color="black", linetype="longdash"
        )
        for y in ["train_time", "test_time"]:
            speedup_y = f"speedup_{y}"
            df_others[speedup_y] = mix[f"{y}_ref"] / mix[y]
            plot = (
                lp.ggplot(df_others, lp.aes(x=x, y=speedup_y, color="model"))
                + common_options
                + aes_speedup
            )
            save_plot(plot, f"{x}_{speedup_y}.png")


def compute_results(platform: str, models: dict[str, tuple[typing.Callable, bool]]):
    model_dfs = []
    for model_name, (model, force) in models.items():
        model_table_path = basepath / f"{model_name}.csv"

        if force:
            model_df = pd.DataFrame()
        elif model_table_path.exists():
            model_df = pd.read_csv(model_table_path)
        else:
            model_df = pd.DataFrame()

        print(f"{model_name}: Running benchmarks ")
        benchmark(model, model_name, model_df, model_table_path)

        print(f"{model_name}: Done")

        model_dfs.append(model_df)
    df = pd.concat(model_dfs, ignore_index=True)
    return df


def export_md(df: pd.DataFrame, pdf=False):

    filepath = basepath / f"results.md"
    with open(filepath, "w") as f:
        f.write("\n## Graphs\n")
        f.write(" All times are specified in seconds \n")
        f.writelines([f"![alt]({p})\n\n" for p in paths])
        f.write("\n# Benchmark table\n")
        df.sort_values(by=["dataset", "model"])
        f.write(df.to_markdown(index=False))
    if pdf:
        pdf_filepath = basepath / "results.pdf"
        command = f"cd '{basepath.absolute()}' && pandoc -f gfm '{filepath.absolute()}' -o '{pdf_filepath.absolute()}'"
        print(command)
        subprocess.run(command, shell=True)


if __name__ == "__main__":
    models = {
        "sklearn.tree": (get_sklearn_tree, False),
        "sklearnmodels.tree[pandas]": (get_nominal_tree, False),
    }

    info = cpuinfo.get_cpu_info()
    platform = (
        "".join(info["brand_raw"].split(" "))
        .replace("/", "-")
        .replace("_", "-")
        .replace("(R)", "")
        .replace("(TM)", "")
    )
    basepath = basepath / platform
    basepath.mkdir(exist_ok=True, parents=True)
    print(f"Running on {platform}")
    df = compute_results(platform, models)
    print(df)

    plot_results(df)
    export_md(df, pdf=True)
