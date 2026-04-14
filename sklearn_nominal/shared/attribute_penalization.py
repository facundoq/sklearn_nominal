import abc

import numpy as np
import pandas as pd
from docutils.nodes import important

from sklearn_nominal.backend.core import Partition
from sklearn_nominal.backend.split import Split

from .target_error import log


class ColumnPenalization(abc.ABC):
    def __init__(self):
        super().__init__()

    abc.abstractmethod

    def penalize(self, partition: Partition):
        pass


class NoPenalization(ColumnPenalization):
    def penalize(self, partition: Partition):
        return 0


class DivisionInfoPenalization(ColumnPenalization):
    def __init__(self, importance: float = 1):
        super().__init__()
        self.importance = importance

    def penalize(self, partition: Partition):
        counts = np.array([d_i.n for d_i in partition], dtype="float64")
        p = counts / counts.sum()
        return -np.sum(p * log(p, 2)) * self.importance
