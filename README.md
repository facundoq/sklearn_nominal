# sklearn_nominal
Extra models for scikit-learn, including Tree, PRISM, CN2, OneR and ZeroR Classifiers and Regressors with support for **nominal values**.

## Colab Quickstart

Check our  [classification models notebook](https://colab.research.google.com/github/facundoq/sklearn_nominal/blob/main/examples/Classification%20Models.ipynb) and [regression models notebook](https://colab.research.google.com/github/facundoq/sklearn_nominal/blob/main/examples/Regression%20Models.ipynb) to see samples of `sklearn_nominal` models in action with simple datasets.

## Installation

To use `sklearn_nominal` in your project, you can install it from [pypi](https://pypi.org/project/sklearn-nominal/) (no conda-forge support yet):

Using `pip`:
````
pip install sklearn_nominal
````

Using `uv`:
````
uv add sklearn_nominal
````

## Installation with support for svg/png/pdf export for Tree models

To export tree graphs to those formats, you need `pygraphviz` (and in the future, possibly other dependencies). Regrettably, `pygraphviz` does not include its own binaries for `grpahviz`. Therefore, make sure to install `graphviz` (with headers) and `cairo`. In Ubuntu 24.04, that can be achieved with:

````
sudo apt install libgraphviz-dev
````

Then use the `export` extras version of `sklearn_nominal` installing:

````
pip install  "sklearn_nominal[export]"
````

