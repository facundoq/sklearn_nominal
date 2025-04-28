import abc

import numpy as np
import pandas as pd

from sklearnmodels.tree.attribute_penalization import ColumnPenalization

from .column_error import SplitterResult
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


type ColumnErrors = dict[str, SplitterResult]


class GlobalSplitter(abc.ABC):

    @abc.abstractmethod
    def global_error(self, x: pd.DataFrame, y: np.ndarray) -> GlobalErrorResult:
        pass

    @abc.abstractmethod
    def split_columns(self, x: pd.DataFrame, y: np.ndarray) -> ColumnErrors:
        pass


class MixedSplitter(GlobalSplitter):

    def __init__(
        self,
        column_splitters: dict[str, GlobalSplitter],
        error_function: TargetError,
        column_penalization: ColumnPenalization,
    ):
        self.column_splitters = column_splitters
        self.target_error = error_function
        self.column_penalization = column_penalization

    def __repr__(self):
        return f"Error({self.target_error})"

    def global_error(self, x: pd.DataFrame, y: np.ndarray):
        global_metric = self.target_error(y)
        global_prediction = self.target_error.prediction(y)
        return GlobalErrorResult(global_prediction, global_metric)

    def split_columns(self, x: pd.DataFrame, y: np.ndarray) -> ColumnErrors:

        errors = {}
        for tipe, column_error in self.column_splitters.items():
            x_type = x.select_dtypes(include=tipe)
            for c in x_type.columns:
                errors[c] = column_error.error(x_type, y, c, self.target_error)

        return errors
