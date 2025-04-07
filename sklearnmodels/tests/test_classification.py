import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,mean_absolute_error
from sklearnmodels import tree

import pytest

from sklearnmodels import SKLearnClassificationTree


def read_classification_dataset(url:str):
    df = pd.read_csv(url)
    x = df.iloc[:,:-1]
    le = LabelEncoder().fit(df.iloc[:,-1])
    y = le.transform(df.iloc[:,-1])
    y = y.reshape(len(y),1)
    return x,y,le.classes_

def get_nominal_tree_classifier(x:pd.DataFrame,classes:int):
    n,m=x.shape
    max_height = min(max(int(np.log(m)*3),5),30)
    min_samples_leaf = max(10,int(n*(0.05/classes)))
    min_samples_split = min_samples_leaf
    min_error_improvement = 0.05/classes    
    return SKLearnClassificationTree(criterion="entropy",max_depth=max_height,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split,min_error_decrease=min_error_improvement,splitter=4)


def test_classification(x:pd.DataFrame,y:np.ndarray,model):
    model.fit(x,y)
    y_pred = model.predict(x)
    return accuracy_score(y,y_pred)

def test_classification_model(model_name:str,model_generator):
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
        model = model_generator(x,len(class_names))
        score = test_classification(x,y,model)
        print(f"Model {model_name} | Dataset {dataset_name} | Score {score}")
        
        
        

if __name__ == "__main__":
    test_classification_model("NominalTree",get_nominal_tree_classifier)  