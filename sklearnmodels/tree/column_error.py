import abc
from typing import Callable

from numpy import isnan

import numpy as np
import pandas as pd

from sklearnmodels.tree.attribute_penalization import ColumnPenalization, NoPenalization
from sklearnmodels.tree.conditions import RangeSplit, Split, ValueSplit

from .target_error import TargetError
from .tree import Condition
from . import ValueCondition, RangeCondition


class SplitterResult:
    def __init__(self, error: float, split: Split, column: str, remove: bool):
        self.error = error
        self.split = split
        self.column = column
        self.remove = remove

    def __repr__(self):
        return (
            f"Score({self.column},{self.error},{len(self.split.conditions)} branches)"
        )


class Splitter(abc.ABC):

    def __init__(self, penalization: ColumnPenalization = NoPenalization()):
        self.penalization = penalization

    @abc.abstractmethod
    def error(
        self, x: pd.DataFrame, y: np.ndarray, column: str, metric: TargetError
    ) -> SplitterResult:
        pass

    def __repr__(self):
        return self.__class__.__name__


type ConditionEvaluationCallback = Callable[[str, np.ndarray, np.ndarray], None]

import pyarrow.compute as pc


class NumericSplitter(Splitter):

    def __init__(
        self,
        max_evals: int = np.iinfo(np.int64).max,
        callbacks: list[ConditionEvaluationCallback] = [],
    ):
        super().__init__()
        assert max_evals > 0
        self.max_evals = max_evals
        self.callbacks = callbacks

    def get_values(self, x: pd.DataFrame, column: str):
        values = x[column].sort_values()
        values = x[column].unique()
        values = values[~np.isnan(values)]
        n = len(values)
        if self.max_evals is not None:
            if n > self.max_evals:
                # subsample
                step = n // self.max_evals
                values = values[::step]
                n = len(values)
        if n > 1:
            values = values[:-1]
            n -= 1
        return values

    def optimize(
        self, x: pd.DataFrame, y: np.ndarray, column: str, metric: TargetError
    ):
        values = self.get_values(x, column)
        n = len(values)

        # find best split value based on unique values of column
        errors = np.zeros(n)
        candidate_splits = []
        for i, v in enumerate(values):
            split = RangeSplit(column, v, x, y)
            errors[i] = metric.average_split(split.partition)
            candidate_splits.append(split)
            penalization = self.penalization.penalize(x, split)
            errors[i] /= penalization

        for callback in self.callbacks:
            callback(column, values, errors)

        best_i = np.argmin(errors)
        return candidate_splits[best_i], errors[best_i]

    def error(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
        column: str,
        metric: TargetError,
    ) -> SplitterResult:
        conditions, error = self.optimize(x, y, column, metric)
        return SplitterResult(error, conditions, column, False)


class NominalSplitter(Splitter):

    def error(
        self, x: pd.DataFrame, y: np.ndarray, column: str, metric: TargetError
    ) -> SplitterResult:
        split = ValueSplit(column, x[column].unique(), x, y)
        error = metric.average_split(split.partition)
        penalization = self.penalization.penalize(x, split)
        error /= penalization
        return SplitterResult(error, split, column, True)
