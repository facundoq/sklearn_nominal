import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,mean_absolute_error
from sklearnmodels import tree

import pytest

output_folderpath = Path("tests/outputs")



def get_regression_trainer():
    scorers = {
        "number":tree.DiscretizingNumericColumnSplitter(tree.OptimizingDiscretizationStrategy()),
        "object":tree.NominalColumnSplitter()
                                }
    prune_criteria = tree.PruneCriteria(max_height=5,min_samples_leaf=3,min_error_decrease=0.01)
    scorer = tree.MixedGlobalError(scorers,tree.DeviationMetric())
    trainer = tree.BaseTreeTrainer(scorer,prune_criteria)
    return trainer

def read_regression_dataset(url:str):
    df = pd.read_csv(url)
    x = df.iloc[:,:-1]
    y = df.iloc[:,-1].to_numpy().reshape(-1,1)
    return x,y





def test_classification_urls():
    datasets = [
        "https://raw.githubusercontent.com/facundoq/facundoq.github.io/refs/heads/master/datasets/classification/2_clases_simple.csv",
    "https://raw.githubusercontent.com/facundoq/facundoq.github.io/refs/heads/master/datasets/classification/6_clases_dificil.csv",
    "https://raw.githubusercontent.com/facundoq/facundoq.github.io/refs/heads/master/datasets/classification/diabetes.csv",
    "https://raw.githubusercontent.com/facundoq/facundoq.github.io/refs/heads/master/datasets/classification/ecoli.csv",
    "https://raw.githubusercontent.com/facundoq/facundoq.github.io/refs/heads/master/datasets/classification/golf_classification_nominal.csv",
    "datasets/trabajos_ej4_2024.csv",
    ]

    for dataset in datasets:
        dataset_name = dataset.split("/")[-1].split(".")[0]
        x,y,class_names=read_classification_dataset(dataset)
        test_classification(x,y,class_names,dataset_name,output_folderpath/f"{dataset_name}.dot")

def test_classification_sklearn():
    datasets = {"iris":(sklearn.datasets.load_iris,0.95)}
    for dataset_name,(loader,score) in datasets.items():
        dataset = loader()
        y=dataset.target
        x = pd.DataFrame(dataset.data,columns=dataset.feature_names)
        test_classification(x,y,dataset_name,output_folderpath/f"{dataset_name}.dot")

def test_regression(x:pd.DataFrame,y:np.ndarray,name:str,filepath:Path):
        
        trainer = get_regression_trainer()
        model = tree.SKLearnRegressionTree(trainer)
        model.fit(x,y)
        y_pred = model.predict(x)
        return mean_absolute_error

def test_regression_urls():        
    datasets = [
    ("https://raw.githubusercontent.com/facundoq/facundoq.github.io/refs/heads/master/datasets/regression/golf_regression_nominal.csv",2900)
    
    ]

    for dataset,max_error in datasets:
        x,y=read_regression_dataset(dataset)
        dataset_name = dataset.split("/")[-1].split(".")[0]
        error = test_regression(x,y,dataset_name)
        assert error < max_error, f"Dataset {dataset_name} must have at most {max_error} error (got {error})"


if __name__ == "__main__":

    test_classification_urls()  
    # test_regression_urls()  
    # test_classification_sklearn()
