import typing
import openml
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
import sklearn.impute
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import sklearn.tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearnmodels import tree
import numpy as np
import pandas as pd
from tqdm import tqdm
import subprocess
# studies = openml.study.list_suites(status = 'all',output_format="dataframe")

def get_nominal_tree_classifier(x:pd.DataFrame,classes:int):
    n,m=x.shape
    scorers = {
        "number":tree.DiscretizingNumericColumnSplitter(tree.OptimizingDiscretizationStrategy(max_evals=3)),
        "object":tree.NominalColumnSplitter(),
        "category":tree.NominalColumnSplitter()
                                }
    scorer = tree.MixedGlobalError(scorers,tree.EntropyMetric(classes))
    max_height = min(max(int(np.log(m)*3),5),30)
    min_samples_leaf = max(10,int(n*(0.05/classes)))
    min_samples_split = min_samples_leaf
    min_error_improvement = 0.05/classes    
    prune_criteria = tree.PruneCriteria(max_height=max_height,min_samples_leaf=min_samples_leaf,min_error_improvement=min_error_improvement,min_samples_split=min_samples_split)
    trainer = tree.BaseTreeTrainer(scorer,prune_criteria)
    return tree.SKLearnClassificationTree(trainer)

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
    return get_sklearn_pipeline(x,model)

def test_benchmark(model_generator:typing.Callable):

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
        le.fit(y)

        classes = len(le.classes_)
        pbar.set_postfix_str(f"{dataset.name}: input={n}x{m} => {classes} classes")
        pbar.update(1)
        clf = model_generator(x,classes)
        if isinstance(clf,tree.SKLearnClassificationTree):
            y = le.transform(y)
            y = y.reshape(-1,1)
        clf.fit(x,y)
        y_pred = clf.predict(x)
        
        # run = openml.runs.run_model_on_task(clf, task)  # run the classifier on the task
        acc = sklearn.metrics.accuracy_score(y_pred,y)
        # score = run.get_metric_fn(sklearn.metrics.accuracy_score)  # print accuracy score
        pbar.write(f'{dataset.name:25} | Train Accuracy: {acc:.2f} | {clf}')
        results.append({"dataset":dataset.name,
                        "score":acc,
                        })
        if isinstance(clf,tree.SKLearnClassificationTree):
            tree_model :tree.Tree = clf.tree
            filepath = f"tests/outputs/{dataset.name}.dot"
            image_filepath = f"tests/outputs/{dataset.name}.png"
            tree.export_dot_file(tree_model,filepath,dataset.name,list(le.classes_))
            subprocess.run(f"dot -Tpng {filepath} > {image_filepath}",shell=True)

    results_df = pd.DataFrame.from_records(results)
    
    print(results_df)

    if isinstance(clf,tree.SKLearnClassificationTree):
        table_path = "tests/outputs/openml_nominal.csv"
    else:
        table_path = "tests/outputs/openml_sklearn.csv"
    results_df.to_csv(table_path)
    
if __name__ == "__main__":
    test_benchmark(get_nominal_tree_classifier)
    #test_benchmark(get_sklearn_tree)