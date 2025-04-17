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
from sklearn.metrics import accuracy_score,mean_absolute_error
from tqdm import tqdm
from sklearnmodels import tree
import sklearn.tree
from sklearn.model_selection import train_test_split
import pytest

from sklearnmodels import SKLearnClassificationTree


def read_classification_dataset(path:Path):
    df = pd.read_csv(path)
    x = df.iloc[:,:-1]
    le = LabelEncoder().fit(df.iloc[:,-1])
    y = le.transform(df.iloc[:,-1])
    #y = y.reshape(len(y),1)
    return x,y,le.classes_

def get_nominal_tree_classifier(x:pd.DataFrame,classes:int):
    n,m=x.shape
    max_height = min(max(int(np.log(m)*3),5),30)
    min_samples_leaf = max(10,int(n*(0.05/classes)))
    min_samples_split = min_samples_leaf
    min_error_improvement = 0.05/classes

    return SKLearnClassificationTree(criterion="entropy",max_depth=max_height,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split,min_error_decrease=min_error_improvement,splitter=4)


def train_test_classification_model(model_name:str,model_generator,dataset:Path):
    dataset_name = dataset.name.split(".")[0]
    x,y,class_names = read_classification_dataset(dataset)
    x_train,x_test,y_train,y_test= train_test_split(x,y,train_size=0.8,stratify=y,shuffle=True,random_state=0)
    model = model_generator(x_train,len(class_names))
    model.fit(x_train,y_train)
    
    y_pred_train = model.predict(x_train)
    score_train = accuracy_score(y_train,y_pred_train)
    y_pred_test = model.predict(x_test)
    score_test = accuracy_score(y_test,y_pred_test)
    return { "Model":model_name, "Dataset":dataset_name, "Train":score_train, "Test":score_test}
        
        

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
    n,m=x.shape
    max_height = min(max(int(np.log(m)*3),5),30)
    min_samples_leaf = max(10,int(n*(0.05/classes)))
    min_samples_split = min_samples_leaf
    min_error_improvement = 0.05/classes  
    model = sklearn.tree.DecisionTreeClassifier(max_depth=max_height,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split,min_impurity_decrease=min_error_improvement,criterion="entropy",random_state=0)
    return get_sklearn_pipeline(x,model)

path = Path("datasets/classification")
dataset_names = [
    "2_clases_simple.csv",
    "6_clases_dificil.csv",
    "diabetes.csv",
    "ecoli.csv",
    "golf_classification_nominal.csv",
    "golf_classification_numeric.csv",
    "seeds.csv",
    "sonar.csv",
    "titanic.csv",
]
def test_performance_similar_sklearn(at_least_percent=0.8,dataset_names=dataset_names):
    
    datasets = [path/name for name in dataset_names]
    nominal_results_all = []
    numeric_results_all = []
    for dataset in tqdm(datasets,desc=f"Datasets"):
        nominal_results = train_test_classification_model("sklearnmodels.tree",get_nominal_tree_classifier,dataset)
        numeric_results = train_test_classification_model("sklearn.tree",get_sklearn_tree,dataset)
        for set in ["Train","Test"]:
            numeric=numeric_results[set]
            nominal=nominal_results[set]
            percent = nominal/numeric
            assert at_least_percent<=percent, f"{set} accuracy of nominal tree ({nominal:.2g}) should be at least {at_least_percent*100:.2g}% of sklearn.tree ({numeric:.2g}) on dataset {nominal_results["Dataset"]}, was only {percent*100:.2g}%."
        nominal_results_all.append(nominal_results)
        numeric_results_all.append(numeric_results)
    print(pd.DataFrame.from_records(nominal_results_all))
    print(pd.DataFrame.from_records(numeric_results_all))


if __name__ == "__main__":
    test_performance_similar_sklearn()
    