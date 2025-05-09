from __future__ import annotations

import abc

import numpy as np
import pandas as pd


# A condition can filter rows of a dataframe (pd.Series),
# Returns a new boolean series
class Condition(abc.ABC):

    def __init__(self, column: str):
        super().__init__()
        self.column = column

    @abc.abstractmethod
    def __call__(self, x: pd.Series) -> np.ndarray:
        pass

    @abc.abstractmethod
    def short_description(self) -> str:
        pass

    def na_to_false(self, s: bool | any):
        if not isinstance(s, (bool, np.bool_)):
            return False
        else:
            return s


class ValueCondition(Condition):
    def __init__(self, column: str, value):
        super().__init__(column)
        self.value = value

    def __call__(self, x: pd.Series):
        return self.na_to_false(x[self.column] == self.value)

    def __repr__(self):
        return f"{self.column}={self.value}"

    def short_description(self):
        return f"{self.value}"


class RangeCondition(Condition):
    def __init__(self, column: str, value: float, less: bool):
        super().__init__(column)
        self.value = value
        self.less = less

    @classmethod
    def make(cls, column, value):
        return [RangeCondition(column, value, t) for t in [True, False]]

    def __call__(self, x: pd.Series):

        if self.less:
            return self.na_to_false(x[self.column] <= self.value)
        else:
            return self.na_to_false(x[self.column] > self.value)

    def __repr__(self):
        op = "<=" if self.less else ">"
        return f"{self.column} {op} {self.value:.4g}"

    def short_description(self):
        op = "<=" if self.less else ">"
        return f"{op} {self.value:.4g}"


class Predicate(Condition):
    def __init__(self, conditions: list[Condition]):
        self.conditions = conditions

    def short_description(self):
        descriptions = [c.short_description() for c in self.conditions]
        descriptions = ",".join(descriptions)
        return f"({descriptions})"

    def __call__(self, x: pd.Series):
        for c in self.conditions:
            if not c(x):
                return False
        return True


class TrueCondition(Condition):
    def __init__(self):
        super().__init__("")

    def __call__(self, x):
        return True

    def short_description(self):
        return "()"
