import openml
import sklearn
import sklearn.impute
import sklearn.tree
from sklearnmodels import tree
import numpy as np


# studies = openml.study.list_suites(status = 'all',output_format="dataframe")

def get_classification_model(classes:int):
    
    scorers = {
        "number":tree.DiscretizingNumericColumnSplitter(tree.OptimizingDiscretizationStrategy()),
        "object":tree.NominalColumnSplitter()
                                }
    scorer = tree.MixedGlobalError(scorers,tree.EntropyMetric(classes))
    prune_criteria = tree.PruneCriteria(max_height=5,min_samples_leaf=3,min_error_improvement=0.01)
    trainer = tree.BaseTreeTrainer(scorer,prune_criteria)
    return tree.SKLearnClassificationTree(trainer)

def test_benchmark():

    benchmark_suite = openml.study.get_suite('OpenML-CC18')  # obtain the benchmark suite
    
    print("Running", benchmark_suite)
    for task_id in benchmark_suite.tasks:  # iterate over all tasks

        task = openml.tasks.get_task(task_id)  # download the OpenML task
        dataset = task.get_dataset()
        classes = dataset.retrieve_class_labels()
        x, Y = task.get_X_and_y(dataset_format='dataframe')  # get the data
        print(f"Running {task} for dataset {dataset.name}")
        
        y = Y.to_numpy().reshape(-1,1)
        print(y.shape,x.shape,y.dtype)
        clf = get_classification_model(len(classes))
        clf.fit(x,y)
        y_pred = clf.predict(x)
        
        # run = openml.runs.run_model_on_task(clf, task)  # run the classifier on the task
        acc = sklearn.metrics.accuracy_score(y_pred,y)
        # score = run.get_metric_fn(sklearn.metrics.accuracy_score)  # print accuracy score
        print(f'Train Accuracy: {acc:.2f}')
        break

if __name__ == "__main__":
    test_benchmark()