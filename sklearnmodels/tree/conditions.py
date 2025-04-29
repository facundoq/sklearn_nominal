from __future__ import annotations

import abc
from typing import Iterable

import numpy as np
import pandas as pd
from numpy import number

type Partition = Iterable[tuple[pd.DataFrame, np.ndarray]]


# A condition can filter rows of a dataframe (pd.Series),
# Returns a new boolean series
class Condition(abc.ABC):

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
        self.column = column
        self.value = value

    def __call__(self, x: pd.Series):
        return self.na_to_false(x[self.column] == self.value)

    def __repr__(self):
        return f"{self.column}={self.value}"

    def short_description(self):
        return f"{self.value}"


class RangeCondition(Condition):
    def __init__(self, column: str, value: float, less: bool):
        self.column = column
        self.value = value
        self.less = less

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


class Split(abc.ABC):
    @property
    @abc.abstractmethod
    def conditions(self) -> list[Condition]:
        pass

    @property
    @abc.abstractmethod
    def partition(self) -> Partition:
        pass

    @abc.abstractmethod
    def split(self, x: pd.DataFrame, y: np.ndarray):
        pass


class ColumnSplit(Split):
    def __init__(self, column: str):
        super().__init__()
        self.column = column


class RangeSplit(ColumnSplit):
    def __init__(self, column: str, value: number, x: pd.DataFrame, y: np.ndarray):
        super().__init__(column)
        self.value = value
        self._partition = list(self.split(x, y))

    @property
    def partition(self):
        return self._partition

    @property
    def conditions(self):
        return [RangeCondition(self.column, self.value, t) for t in [True, False]]

    def split(self, x: pd.DataFrame, y: np.ndarray):
        idx = x[self.column] <= self.value
        not_idx = (~idx).fillna(False)
        idx = idx.fillna(False)

        yield x.loc[idx], y[idx]
        yield x.loc[not_idx], y[not_idx]


class ValueSplit(ColumnSplit):
    def __init__(self, column: str, values: list, x: pd.DataFrame, y: np.ndarray):
        super().__init__(column)
        self.values = values
        self._partition = list(self.split(x, y))

    @property
    def partition(self):
        return self._partition

    @property
    def conditions(self):
        return [ValueCondition(self.column, v) for v in self.values]

    def split(self, x: pd.DataFrame, y: np.ndarray):
        for value in self.values:
            idx = x[self.column].values == value
            if idx.any():
                yield x.loc[idx], y[idx]
