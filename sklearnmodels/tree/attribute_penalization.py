import abc
import numpy as np
import pandas as pd

from sklearnmodels.tree.conditions import Split
from sklearnmodels.tree.target_error import log
from sklearnmodels.tree.tree import Condition


class ColumnPenalization(abc.ABC):
    def __init__(self):
        super().__init__()

    abc.abstractmethod

    def penalize(self, x: pd.DataFrame, split: Split):
        pass


class NoPenalization(ColumnPenalization):
    def penalize(self, x: pd.DataFrame, split: Split):
        return 1


class GainRatioPenalization(ColumnPenalization):
    def penalize(self, x: pd.DataFrame, y: np.ndarray, split: Split):
        counts = np.array([len(x_i) for x_i, y_i in split.partition])
        counts /= counts.sum()
        return -np.sum(counts * log(counts, len(counts)))
