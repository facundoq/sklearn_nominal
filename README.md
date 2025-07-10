# sklearn_nominal
Extra models for scikit-learn, including Decision/Regression Trees with support for nominal values



## Installation

## Exporting to svg/png/pdf
To export tree graphs to those formats, you need `pygraphviz` (and in the future, possibly other dependencies). Regrettably, `pygraphviz` does not include its own binaries for `grpahviz`. Therefore, make sure to install `graphviz` (with headers) and `cairo`. In Ubuntu 24.04, that can be achieved with:

````
sudo apt install libgraphviz-dev graphviz cairosvg 
````


#