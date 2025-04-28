from pathlib import Path
import typing

import openml
import sklearn
from sklearnmodels import SKLearnClassificationTree
import sklearn.impute
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import lets_plot as lp
import cpuinfo

# studies = openml.study.list_suites(status = 'all',output_format="dataframe")
import time

basepath = Path("benchmark/outputs/")


import numpy as np
import pandas as pd
from sklearnmodels import SKLearnClassificationTree

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


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


def benchmark(model_generator: typing.Callable, model_name: str) -> pd.DataFrame:
    benchmark_suite = openml.study.get_suite(
        "OpenML-CC18"
    )  # obtain the benchmark suite

    print("Running", benchmark_suite)
    results = []
    pbar = tqdm(total=len(benchmark_suite.tasks))
    for i, task_id in enumerate(benchmark_suite.tasks):  # iterate over all tasks
        task = openml.tasks.get_task(task_id)  # download the OpenML task
        dataset = task.get_dataset()
        pbar.update(1)
        if i == 2:
            break
        x, y = task.get_X_and_y(dataset_format="dataframe")  # get the data

        n, m = x.shape
        if m > 1000:
            m = 32
            pca = sklearn.decomposition.PCA(n_components=m)
            x = pca.fit_transform(x)
            x = pd.DataFrame(
                data=x,
                columns=[
                    f"v_{i}={v:.2f}"
                    for i, v in enumerate(pca.explained_variance_ratio_)
                ],
            )

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

        # run = openml.runs.run_model_on_task(clf, task)  # run the classifier on the task
        acc = sklearn.metrics.accuracy_score(y, y_pred)

        # score = run.get_metric_fn(sklearn.metrics.accuracy_score)  # print accuracy score
        results.append(
            {
                "model": model_name,
                "dataset": dataset.name,
                "train_accuracy": acc,
                "train_time": train_elapsed,
                "test_time": test_elapsed,
                "samples": n,
                "features": m,
            }
        )

        if isinstance(model, SKLearnClassificationTree):
            image_filepath = basepath / f"trees/{dataset.name}.svg"
            model.export_image(image_filepath, le.classes_)

    results_df = pd.DataFrame.from_records(results)
    return results_df


paths = []


def save_plot(plot, filename: str):
    path = str((basepath / filename).absolute())
    lp.ggsave(plot, filename=path, w=8, h=4, unit="in", dpi=300)
    paths.append(filename)
    return filename


def plot_results(df: pd.DataFrame, platform: str):
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
        save_plot(plot, f"openml_cc18_{platform}_{y}.png")
    for x in ["samples", "features"]:
        for y in ["train_time", "test_time"]:
            plot = lp.ggplot(df, lp.aes(x=x, y=y, color="model")) + common_options
            save_plot(plot, f"openml_cc18_{platform}_{x}_{y}.png")
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
            save_plot(plot, f"openml_cc18_{platform}_{x}_{speedup_y}.png")
        # plot = lp.ggplot(df,lp.aes(x="samples",y="features",size="train_time",color="model"))+ lp.geom_point(alpha=0.3)+ lp.ggsize(800, 400)+axis_font
    # save_plot(plot,f"openml_cc18_{platform}_samples_features_{y}.png")


def compute_results(platform: str, models: dict[str, typing.Callable], force=False):
    table_path = basepath / f"openml_cc18_{platform}.csv"
    model_dfs = []
    for model_name, model in models.items():
        model_table_path = basepath / f"openml_cc18_{platform}_{model_name}.csv"
        if not model_table_path.exists() or force:
            print(f"Running benchmarks for {model_name}")
            model_df = benchmark(model, model_name)
            model_df.to_csv(model_table_path)
        else:
            print(f"Loading stored results for {model_name}")
            model_df = pd.read_csv(table_path)
        model_dfs.append(model_df)
    df = pd.concat(model_dfs, ignore_index=True)
    return df


def export_md(df: pd.DataFrame, platform: str):
    with open(basepath / f"openml_cc18_{platform}_benchmark.md", "w") as f:
        f.write("# Benchmark table\n")
        df.sort_values(by=["dataset", "model"])
        f.write(df.to_markdown(index=False))
        f.write("\n## Graphs\n")
        f.write(" All times are specified in seconds \n")
        f.writelines([f"![alt]({p})" for p in paths])


if __name__ == "__main__":
    models = {
        "sklearn.tree": get_sklearn_tree,
        "sklearnmodels.tree": get_nominal_tree,
    }

    info = cpuinfo.get_cpu_info()
    platform = (
        "".join(info["brand_raw"].split(" "))
        .replace("/", "-")
        .replace("_", "-")
        .replace("(R)", "")
        .replace("(TM)", "")
    )
    print(f"Running on {platform}")
    df = compute_results(platform, models, force=False)
    print(df)

    plot_results(df, platform)
    export_md(df, platform)
