import abc

import numpy as np
import pandas as pd

from sklearnmodels.backend.core import ColumnType, Dataset
from sklearnmodels.tree.attribute_penalization import ColumnPenalization

from .column_error import ColumnErrorResult
from .target_error import TargetError


# TODO simplify this
class GlobalErrorResult:
    def __init__(
        self,
        prediction: np.ndarray,
        error: float,
    ):
        self.prediction = prediction
        self.error = error


type ColumnErrors = dict[str, ColumnErrorResult]


class GlobalSplitter(abc.ABC):

    @abc.abstractmethod
    def global_error(self, x: pd.DataFrame, y: np.ndarray) -> GlobalErrorResult:
        pass

    @abc.abstractmethod
    def split_columns(self, x: pd.DataFrame, y: np.ndarray) -> ColumnErrors:
        pass


class DefaultSplitter(GlobalSplitter):

    def __init__(
        self,
        column_splitters: dict[ColumnType, GlobalSplitter],
        error_function: TargetError,
        column_penalization: ColumnPenalization,
    ):
        self.column_splitters = column_splitters
        self.target_error = error_function
        self.column_penalization = column_penalization

    def __repr__(self):
        return f"Error({self.target_error})"

    def global_error(self, d: Dataset):
        global_metric = self.target_error(d.y)
        global_prediction = self.target_error.prediction(d.y)
        return GlobalErrorResult(global_prediction, global_metric)

    def split_columns(self, d: Dataset) -> ColumnErrors:

        errors = {}
        for c, c_type in zip(d.columns, d.types):
            column_error = self.column_splitters[c_type]
            errors[c] = column_error.error(d, c, self.target_error)

        return errors
