import abc
from typing import Callable
import numpy as np
import pandas as pd

from .conditions import Condition

type Branches = dict[Condition, Tree]


class TreeInfo:
    def __init__(
        self, column_names: list[str], categorical_values: dict[str, dict[int, str]]
    ):
        self.column_names = column_names
        self.categorical_values = categorical_values


class Tree:
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

    def predict(self, x: pd.Series):
        for condition, child in self.branches.items():
            result = condition(x)
            if result:
                return child.predict(x)
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

    def height(self):
        if self.leaf:
            return 1
        else:
            return 1 + max([t.height() for t in self.children()])

    def pretty_print(self, height=0, max_height=np.inf):
        result = ""
        if height == 0:
            result = f"root"
        if self.leaf:
            result = f"{self}"

        if height >= max_height:
            return ""
        base_sep = "|   "
        indent = base_sep * height
        if self.leaf:
            children = ""
        else:
            children = "\n" + "\n".join(
                [
                    f"{indent}{base_sep}🪵{c} => {t.pretty_print(height+1)}"
                    for c, t in self.branches.items()
                ]
            )

        return f"{result}{children}"
