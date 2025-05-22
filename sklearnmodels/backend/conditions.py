from __future__ import annotations

import abc

import numpy as np
import pandas as pd

from sklearnmodels.backend import InputSample


# A condition can filter rows of a Dataset
# Returns a new boolean series
class Condition(abc.ABC):

    def __init__(self, column: str):
        super().__init__()
        self.column = column

    @abc.abstractmethod
    def __call__(self, x: InputSample) -> bool:
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

    def __call__(self, x: InputSample):
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

    def __call__(self, x: InputSample):

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


class AndCondition(Condition):
    def __init__(self, conditions: list[Condition]):
        column = ",".join([c.column for c in conditions])
        super().__init__(column)
        self.conditions = conditions

    def short_description(self):
        descriptions = [c.short_description() for c in self.conditions]
        descriptions = ",".join(descriptions)
        return f"({descriptions})"

    def __call__(self, x: InputSample):
        for c in self.conditions:
            if not c(x):
                return False
        return True


class TrueCondition(Condition):
    def __init__(self):
        super().__init__("")

    def __call__(self, x: InputSample):
        return True

    def short_description(self):
        return "()"


class NotCondition(Condition):
    def __init__(self, condition: Condition):
        super().__init__(condition.column)
        self.condition = condition

    def __call__(self, x: InputSample):
        return ~self.condition(x)

    def short_description(self):
        return "()"
