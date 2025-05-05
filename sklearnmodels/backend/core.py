from __future__ import annotations

import abc
import enum
from typing import Generator, Iterable

import numpy as np

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
    def x():
        pass

    @property
    @abc.abstractmethod
    def y():
        pass

    @property
    @abc.abstractmethod
    def n() -> int:
        pass

    @property
    @abc.abstractmethod
    def types() -> list[ColumnType]:
        pass

    @property
    @abc.abstractmethod
    def columns() -> list[str]:
        pass

    @abc.abstractmethod
    def drop(self, columns: list[str]) -> Dataset:
        pass

    @abc.abstractmethod
    def values(self, column: str) -> np.ndarray:
        pass

    @abc.abstractmethod
    def unique_values(self, column: str) -> np.ndarray:
        pass
