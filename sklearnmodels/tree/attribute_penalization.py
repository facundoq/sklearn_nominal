import abc

import numpy as np
import pandas as pd

from sklearnmodels.backend.core import Partition
from sklearnmodels.backend.split import Split
from sklearnmodels.tree.target_error import log


class ColumnPenalization(abc.ABC):
    def __init__(self):
        super().__init__()

    abc.abstractmethod

    def penalize(self, partition: Partition):
        pass


class NoPenalization(ColumnPenalization):
    def penalize(self, partition: Partition):
        return 1


class GainRatioPenalization(ColumnPenalization):
    def penalize(self, partition: Partition):
        counts = np.array([len(x_i) for x_i, y_i in partition])
        counts /= counts.sum()
        return -np.sum(counts * log(counts, len(counts)))
