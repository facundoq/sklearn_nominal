from __future__ import annotations
from typing import Generator, Iterable


from numpy import dtype, ndarray
from scipy.special import y1
from .conditions import Condition, RangeCondition, ValueCondition
from .core import ColumnType, Dataset
import pandas as pd
import numpy as np


class PandasDataset(Dataset):

    def __init__(self, x: pd.DataFrame, y: np.ndarray, idx=None):
        super().__init__()
        self._x: pd.DataFrame = x
        self._y: np.ndarray = y
        self.idx = idx
        self._x_subset = None
        self._y_subset = None

    @property
    def x(self) -> pd.DataFrame:
        if self.idx is None:
            return self._x
        else:
            if self._x_subset is None:
                self._x_subset: pd.DataFrame = self._x.loc[self.idx]
            return self._x_subset

    @property
    def y(self) -> pd.ndarray:
        if self.idx is None:
            return self._y
        else:  # lazy filtering
            if self._y_subset is None:
                self._y_subset: np.ndarray = self._y[self.idx]
            return self._y_subset

    def split(self, conditions: list[Condition]):
        return [self.filter(c) for c in conditions]

    def values(self, column: str):
        result: pd.Series = self.x[column].dropna()
        return result

    def unique_values(self, column: str, sorted=False) -> np.ndarray:
        result = self.values(column).unique()
        if sorted:
            result.sort()
        return result

    def filter(self, condition: Condition):
        if isinstance(condition, RangeCondition):
            rc: RangeCondition = condition
            if rc.less:
                idx = self.x[rc.column] <= rc.value
            else:
                idx = self.x[rc.column] > rc.value
            idx.fillna(False, inplace=True)
            return PandasDataset(self.x, self.y, idx=idx)
        elif isinstance(condition, ValueCondition):
            vc: ValueCondition = condition
            idx = self.x[vc.column] == vc.value
            idx.fillna(False, inplace=True)
            return PandasDataset(self.x, self.y, idx=idx)
        else:
            raise ValueError(f"Invalid condition: {condition}")

    @property
    def n(self):
        return self.y.shape[0]

    @property
    def types(self) -> list[ColumnType]:
        numeric = self.x.select_dtypes(include="number").columns

        def to_type(column):
            if column in numeric:
                return ColumnType.Numeric
            else:
                return ColumnType.Nominal

        return map(to_type, self.columns)

    @property
    def columns(self) -> list[str]:
        return self.x.columns

    def drop(self, columns: list[str]) -> PandasDataset:
        x = self.x.drop(columns=columns)
        return PandasDataset(x, self.y)

    def classes(self):
        values = np.unique(self.y)
        values.sort()
        return values

    def filter_by_class(self, c) -> Dataset:
        idx = self.y == c
        idx.fillna(False, inplace=True)
        return PandasDataset(self.x, self.y, idx)

    def class_distribution(self, class_weight: np.ndarray) -> np.ndarray:
        p: np.ndarray = np.bincount(self.y)
        result = p / p.sum()
        if class_weight is not None:
            result *= class_weight
            result /= result.sum()
        return result

    def mean_y(
        self,
    ) -> np.ndarray:
        return self.y.mean(axis=0)

    def std_y(
        self,
    ) -> float:
        if self.y.shape[0] == 0:
            return np.inf
        return np.sum(np.std(self.y, axis=0))
