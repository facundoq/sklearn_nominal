from pathlib import Path
import typing
import openml
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
import sklearn.impute
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearnmodels import tree,SKLearnClassificationTree,SKLearnClassificationTree
import numpy as np
import pandas as pd
from tqdm import tqdm
import subprocess
#import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
from lets_plot import *

# studies = openml.study.list_suites(status = 'all',output_format="dataframe")
import time
basepath = Path("benchmark/outputs/")


def get_nominal_tree(x:pd.DataFrame,classes:int):
    n,m=x.shape
    max_height = min(max(int(np.log(m)*3),5),30)
    min_samples_leaf = max(10,int(n*(0.05/classes)))
    min_samples_split = min_samples_leaf
    min_error_improvement = 0.05/classes    
    return SKLearnClassificationTree(criterion="entropy",max_depth=max_height,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split,min_error_decrease=min_error_improvement,splitter=4),"sklearnmodels.tree"

def get_sklearn_pipeline(x:pd.DataFrame,model):
    numeric_features = x.select_dtypes(include=['int64','float64']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = x.select_dtypes(exclude=['int64','float64']).columns
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    return Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])
    

def get_sklearn_tree(x:pd.DataFrame,classes:int):
    model = sklearn.tree.DecisionTreeClassifier(max_depth=20,min_samples_leaf=20)
    return get_sklearn_pipeline(x,model),"sklearn.tree"

def benchmark(model_generator:typing.Callable)->pd.DataFrame:
    benchmark_suite = openml.study.get_suite('OpenML-CC18')  # obtain the benchmark suite
    
    print("Running", benchmark_suite)
    results = []
    pbar = tqdm(total=len(benchmark_suite.tasks))

    for i,task_id in enumerate(benchmark_suite.tasks):  # iterate over all tasks
        task = openml.tasks.get_task(task_id)  # download the OpenML task
        dataset = task.get_dataset()
        
        x, y = task.get_X_and_y(dataset_format='dataframe')  # get the data
        n,m=x.shape
        # print(f"Running {task} for dataset {dataset.name}")
        le = LabelEncoder()
        y = le.fit_transform(y)

        classes = len(le.classes_)
        pbar.set_postfix_str(f"{dataset.name}: input={n}x{m} => {classes} classes")
        pbar.update(1)
        model,model_name = model_generator(x,classes)
        start = time.time_ns()
        model.fit(x,y)
        y_pred = model.predict(x)
        elapsed = (time.time_ns()-start)/10e9
        
        # run = openml.runs.run_model_on_task(clf, task)  # run the classifier on the task
        acc = sklearn.metrics.accuracy_score(y,y_pred)
        # score = run.get_metric_fn(sklearn.metrics.accuracy_score)  # print accuracy score
        
        results.append({"model":model_name,
                        "dataset":dataset.name,
                        "train_accuracy":acc,
                        "time": elapsed,
                        "samples":n,
                        "features":m,
                        })
        if i >2:
            break
        if isinstance(model,SKLearnClassificationTree):
            tree_model :tree.Tree = model.tree_
            filepath = basepath/f"trees/{dataset.name}.dot"
            image_filepath = basepath/f"trees/{dataset.name}.png"
            tree.export_dot_file(tree_model,filepath,dataset.name,list(le.classes_))
            subprocess.run(f"dot -Tpng {filepath} > {image_filepath}",shell=True)

    results_df = pd.DataFrame.from_records(results)
    return results_df
    
    
def save_results(df):
    # (
    # so.Plot(df, x="dataset",color="model")
    # .pair(y=["train_accuracy","time"])
    # .add(so.Line())
    # .add(so.Dots(color=".2"))

    # ).save(basepath/"openml_cc18.png")

    # (
    # so.Plot(df, y="time",color="model")
    # .pair(y=["features","samples"])
    # .add(so.Line())
    # .add(so.Dots(color=".2"))

    # ).save(basepath/"openml_cc18_time.png")
    
    for y in ["train_accuracy","time"]:
        path = str((basepath/f"openml_cc18_{y}.svg").absolute())
        ggsave(ggplot(df, aes(x='dataset', y=y, color='model')) + ggsize(700, 300)+geom_line()+geom_point(),filename=path)

    
if __name__ == "__main__":
    table_path = basepath /"openml_cc18.csv"    
    if not table_path.exists():
        nominal_df = benchmark(get_nominal_tree)
        numeric_df = benchmark(get_sklearn_tree)
        df = pd.concat([nominal_df,numeric_df], ignore_index=True)
        df.to_csv(table_path)
    else:
        df = pd.read_csv(table_path)
    print(df)
    save_results(df)
    