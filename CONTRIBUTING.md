
# Developing sklearn_nominal

We use [uv](https://docs.astral.sh/uv/) for project management.

### First run

Install `graphviz` (with headers) and `cairo`. In Ubuntu 24.04:

````
sudo apt install libgraphviz-dev graphviz cairosvg 
````

Install deps:
````
uv sync --dev --extra export
````

Install pre-commit hooks
````
uv run pre-commit install
````

Install `poethepoet` for common tasks such as linting, formatting, building docs, etc
````
uv tool install poethepoet
````

### Running pre commit hooks

````
pre-commit run --all-files
````
### Install PoeThePoet task runner



### Running pytest, linter, formatter

````
poe lint
poe format
poe test
````

### Generating docs

To continually update docs, use:
````
poe serve-docs
````

To only build once, try:
````
poe docs
````

Compiled docs are not stored in the git repo. Instead, `readthedocs` pulls from the repo and rebuilds them whenever a new version is generated.


### Running benchmarks

In addition to the tests, we have benchmarks comparing `sklearn_nominal.TreeClassifier` to `sklearn.tree.DecisionTreeClassifier` to have a baseline comparison between the performance of these two models. After changes in the backend, make sure you run the benchmarks to ensure no regressions. 

````
uv run benchmark/benchmark_openml.py
````

### Publishing to pipy

When you have a new version to publish, make sure to upgrade the version number in `project.toml`, format, lint and run tests, and then commit and push that version to github. Afterwards, tag the commit and push those tags.

````

git tag v[version]
git push --tags
````
