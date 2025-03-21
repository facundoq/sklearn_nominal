

def test_regression_model(model_name:str,model_generator:Callable[]):
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
        test_classification(x,y,class_names,dataset_name)