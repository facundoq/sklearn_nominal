from __future__ import annotations
from typing import Generator, Iterable


from scipy.special import y1
from .conditions import Condition, RangeCondition, ValueCondition
from .core import ColumnType, Dataset
import pandas as pd
import numpy as np


class PandasDataset(Dataset):

    def __init__(self, x: pd.DataFrame, y: np.ndarray):
        super().__init__()
        self.x: pd.DataFrame = x
        self.y: np.ndarray = y
        self.cache = None

    def split(self, conditions: list[Condition]) -> list[PandasDataset]:
        return [self.filter(c) for c in conditions]

    def values(self, column: str) -> np.ndarray:
        return self.x[column].dropna()

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
            return PandasDataset(self.x.loc[idx], self.y[idx])
        elif isinstance(condition, ValueCondition):
            vc: ValueCondition = condition
            idx = self.x[vc.column] == vc.value
            return PandasDataset(self.x.loc[idx], self.y[idx])
        else:
            raise ValueError(f"Invalid condition: {condition}")

    @property
    def n(self):
        return self.y.shape[0]

    def types(self) -> list[ColumnType]:
        numeric = self.x.select_dtypes(include="number").columns

        def to_type(column):
            if column in numeric:
                return ColumnType.Numeric
            else:
                return ColumnType.Nominal

        return map(to_type, self.columns)

    def columns(self) -> list[str]:
        return self.x.columns

    def drop(self, columns: list[str]) -> PandasDataset:
        x = self.x.drop(columns=columns)
        return PandasDataset(x, self.y)
