import numpy as np
import pandas as pd

from sklearnmodels.backend.core import Model

from ..backend.conditions import Condition

type Branches = dict[Condition, Tree]


class TreeInfo:
    def __init__(
        self, column_names: list[str], categorical_values: dict[str, dict[int, str]]
    ):
        self.column_names = column_names
        self.categorical_values = categorical_values


class Tree(Model):
    def __init__(
        self,
        prediction: np.ndarray,
        error: float,
        samples: int,
        branches: Branches = None,
    ):
        if branches is None:
            branches = {}
        self.branches = branches
        self.prediction = prediction
        self.samples = samples
        self.column: str = None
        self.error = error

    def output_size(self):
        return len(self.prediction)

    def predict_sample(self, x: pd.Series):
        for condition, child in self.branches.items():
            result = condition(x)
            if result:
                return child.predict_sample(x)
        return self.prediction

    def children(self):
        return list(self.branches.values())

    def conditions(self):
        return list(self.branches.keys())

    @property
    def leaf(self):
        return len(self.branches) == 0

    def __repr__(self):
        if self.leaf:
            return f"🍁({self.prediction},n={self.samples})"
        else:
            return f"🪵({self.column})"

    def n_leafs(self):
        if self.leaf:
            return 1
        else:
            return sum([t.n_leafs() for t in self.children()])

    def n_nodes(self):
        return 1 + sum([t.n_nodes() for t in self.children()])

    def height(self):
        return 1 + max([t.height() for t in self.children()])

    def complexity(self):
        return self.n_leafs()

    def pretty_print(self, height=0, max_height=np.inf, class_names=None):

        result = ""
        if height == 0:
            result = "root"
        if self.leaf:
            result = f"{self.prediction}"
            if class_names is not None:

                klass = self.prediction.argmax()
                result = f"{class_names[klass]}"

        if height >= max_height:
            return ""
        base_sep = "|   "
        indent = base_sep * height
        if self.leaf:
            children = ""
        else:
            children = "\n" + "\n".join(
                [
                    f"{indent}{base_sep}🪵{c} => {t.pretty_print(height+1,max_height=max_height,class_names=class_names)}"
                    for c, t in self.branches.items()
                ]
            )

        return f"{result}{children}"
