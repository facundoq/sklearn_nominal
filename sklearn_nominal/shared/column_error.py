import abc
from typing import Callable

import numpy as np
import pandas as pd

from sklearn_nominal.backend.conditions import Condition, RangeCondition, ValueCondition, NotCondition
from sklearn_nominal.backend.core import Dataset, Partition
from sklearn_nominal.backend.split import RangeSplit, Split, ValueSplit

from .attribute_penalization import ColumnPenalization, NoPenalization
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
        return f"Score({self.column},{self.error},{len(self.conditions)} branches)"


ColumnCallback = Callable[[ColumnErrorResult], None]


class ColumnError(abc.ABC):
    def __init__(
        self,
        metric: TargetError,
        penalization: ColumnPenalization = NoPenalization(),
        callback=None,
    ):
        self.penalization = penalization
        self.metric = metric
        self.callback = callback

    def do_callback(self, result: ColumnErrorResult):
        if self.callback is not None:
            self.callback(result)

    @abc.abstractmethod
    def error(self, d: Dataset, column: str) -> ColumnErrorResult | None:
        pass

    def __repr__(self):
        return self.__class__.__name__

    def evaluate_conditions(
        self,
        d: Dataset,
        conditions: list[Condition],
        column: str,
        remove=False,
    ) -> ColumnErrorResult:
        partition = d.split(conditions)
        error = self.metric.average_split(partition)
        error /= self.penalization.penalize(partition)
        return ColumnErrorResult(column, error, conditions, partition, remove)


class NumericColumnError(ColumnError):
    def __init__(
        self,
        metric: TargetError,
        penalization: ColumnPenalization = NoPenalization(),
        callback=None,
        max_evals: int = np.iinfo(np.int64).max,
    ):
        super().__init__(metric, penalization, callback=callback)
        if max_evals <= 0:
            raise ValueError("max_evals must be greater than 0")
        self.max_evals = max_evals

    def get_split_points(self, d: Dataset, column: str) -> np.ndarray:
        """
        Identifies potential split points for a numeric column.
        """
        # Get unique sorted values for potential split points
        values = d.unique_values(column, sorted=True)
        n = len(values)

        if n <= 1:
            return np.array([])

        # Subsample if there are too many unique values to evaluate
        if self.max_evals is not None and n > self.max_evals:
            step = n // self.max_evals
            values = values[::step]
            n = len(values)

        # Use unique values as split points (except the last one)
        # This creates splits like: x <= v and x > v
        return values[:-1]

    def error(
        self,
        d: Dataset,
        column: str,
    ) -> ColumnErrorResult | None:
        """
        Finds the best split point for a numeric column by minimizing the metric.
        """
        split_points = self.get_split_points(d, column)
        if len(split_points) == 0:
            return None

        best = None
        for v in split_points:
            # RangeCondition.make(column, v) returns [x <= v, x > v]
            conditions = RangeCondition.make(column, v)
            result = self.evaluate_conditions(d, conditions, column)
            self.do_callback(result)

            if best is None or result.error < best.error:
                best = result
        return best


class NominalColumnError(ColumnError):
    def error(self, d: Dataset, column: str) -> ColumnErrorResult | None:
        conditions: list[Condition] = [ValueCondition(column, v) for v in d.unique_values(column, False)]
        result = self.evaluate_conditions(d, conditions, column, remove=True)
        self.do_callback(result)
        return result


class BinaryNominalColumnError(ColumnError):
    """
    Finds the best binary split (One-vs-Rest) for a nominal column.
    This prevents extremely wide trees by creating binary branches.
    """

    def error(self, d: Dataset, column: str) -> ColumnErrorResult | None:
        unique_values = d.unique_values(column, False)
        if len(unique_values) <= 2:
            # For 1 or 2 values, use the standard nominal error
            return NominalColumnError(self.metric, self.penalization, self.callback).error(d, column)

        best = None
        for v in unique_values:
            # Create a One-vs-Rest split
            conditions = [
                ValueCondition(column, v),
                NotCondition(ValueCondition(column, v)),
            ]
            result = self.evaluate_conditions(d, conditions, column)
            self.do_callback(result)
            if best is None or result.error < best.error:
                best = result
        return best
