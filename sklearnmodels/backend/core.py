from __future__ import annotations

import abc
import enum
from typing import Generator, Iterable

import numpy as np
import pandas as pd

from .conditions import Condition


class ColumnType(enum.Enum):
    Numeric = 0
    Nominal = 1


type Partition = list[Dataset]


class Dataset(abc.ABC):

    @abc.abstractmethod
    def split(self, conditions: list[Condition]) -> Partition:
        pass

    @property
    @abc.abstractmethod
    def x(
        self,
    ) -> pd.DataFrame:
        pass

    @property
    @abc.abstractmethod
    def y(
        self,
    ) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def n(
        self,
    ) -> int:
        pass

    @property
    @abc.abstractmethod
    def types(
        self,
    ) -> list[ColumnType]:
        pass

    @property
    @abc.abstractmethod
    def columns(
        self,
    ) -> list[str]:
        pass

    @abc.abstractmethod
    def drop(self, columns: list[str]) -> Dataset:
        pass

    @abc.abstractmethod
    def values(self, column: str) -> np.ndarray:
        pass

    @abc.abstractmethod
    def unique_values(self, column: str, sorted: bool) -> np.ndarray:
        pass

    @abc.abstractmethod
    def classes(self) -> list:
        pass

    @abc.abstractmethod
    def filter_by_class(self, c) -> Dataset:
        pass

    @abc.abstractmethod
    def class_distribution(self, class_weight: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def mean_y(
        self,
    ) -> np.ndarray:
        pass

    @abc.abstractmethod
    def std_y(self) -> float:
        pass
