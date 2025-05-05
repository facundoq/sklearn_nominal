import abc
from typing import Callable

from h11 import Data
import numpy as np
import pandas as pd

from sklearnmodels.backend.conditions import Condition, RangeCondition, ValueCondition
from sklearnmodels.backend.core import Dataset, Partition
from sklearnmodels.backend.split import RangeSplit, Split
from sklearnmodels.tree.attribute_penalization import ColumnPenalization, NoPenalization
from sklearnmodels.backend.split import ValueSplit

from .target_error import TargetError


class ColumnErrorResult:
    def __init__(
        self,
        column: str,
        error: float,
        conditions: list[Condition],
        partition: Partition,
        remove: bool = False,
    ):
        self.error = error
        self.conditions = conditions
        self.partition = partition
        self.column = column
        self.remove = remove

    def __repr__(self):
        return (
            f"Score({self.column},{self.error},{len(self.split.conditions)} branches)"
        )


class ColumnError(abc.ABC):

    def __init__(self, penalization: ColumnPenalization = NoPenalization()):
        self.penalization = penalization

    @abc.abstractmethod
    def error(self, d: Dataset, column: str, metric: TargetError) -> ColumnErrorResult:
        pass

    def __repr__(self):
        return self.__class__.__name__

    def evaluate(
        self,
        d: Dataset,
        conditions: list[Condition],
        column: str,
        metric: TargetError,
        remove=False,
    ):
        partition = d.split(conditions)
        error = metric.average_split(partition)
        error /= self.penalization.penalize(partition)
        return ColumnErrorResult(column, error, conditions, partition, remove)


type ConditionEvaluationCallbackResult = tuple[str, np.ndarray, np.ndarray]
type ConditionEvaluationCallback = Callable[[ConditionEvaluationCallbackResult], None]


class NumericColumnError(ColumnError):

    def __init__(
        self,
        max_evals: int = np.iinfo(np.int64).max,
        callbacks: list[ConditionEvaluationCallback] = [],
    ):
        super().__init__()
        assert max_evals > 0
        self.max_evals = max_evals
        self.callbacks = callbacks

    def get_values(self, d: Dataset, column: str):
        values = d.unique_values(column, sorted=True)
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

    def optimize(self, d: Dataset, column: str, metric: TargetError):
        values = self.get_values(d, column)
        # find best split value based on unique values of column
        best = None
        for i, v in enumerate(values):
            conditions = RangeCondition.make(self.column, self.value)
            result = self.evaluate(d, conditions, column, metric, True)
            if best is None or result.error <= best.error:
                best = result
        return best

    def error(
        self,
        d: Dataset,
        column: str,
        metric: TargetError,
    ) -> ColumnErrorResult:
        conditions, error = self.optimize(d, column, metric)
        return ColumnErrorResult(error, conditions, column, False)


class NominalColumnError(ColumnError):

    def error(self, d: Dataset, column: str, metric: TargetError) -> ColumnErrorResult:
        conditions = [ValueCondition(column, v) for v in d.unique_values(column)]
        return self.evaluate(d, conditions, column, metric)
