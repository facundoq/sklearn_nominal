# sklearn_nominal
Extra models for scikit-learn, including Decision/Regression Trees with support for nominal values



## Installation

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
sudo apt install libgraphviz-dev graphviz cairosvg 
````

Then use the `export` extras version of `sklearn_nominal` installing:

````
pip install  "sklearn_nominal[export]"
````

