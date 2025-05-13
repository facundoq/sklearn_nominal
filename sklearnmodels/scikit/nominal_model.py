import abc

from sklearnmodels.backend.factory import make_dataset
from .. import shared
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data, _check_y
from sklearn.utils.multiclass import check_classification_targets
from ..backend.core import Dataset
from ..backend.pandas import PandasDataset

from sklearn.utils._tags import (
    ClassifierTags,
    RegressorTags,
    Tags,
    TargetTags,
    TransformerTags,
    get_tags,
)

from .. import tree


def atleast_2d(x):
    x = np.asanyarray(x)
    if x.ndim == 0:
        result = x.reshape(1, 1)
    elif x.ndim == 1:
        result = x[:, np.newaxis]
    else:
        result = x
    return result


class NominalModel(BaseEstimator, metaclass=abc.ABCMeta):
    check_parameters = {"dtype": None}

    def __init__(self, backend: str = "pandas", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = backend

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.single_output = False
        tags.target_tags.required = True
        tags.non_deterministic = False
        tags.input_tags.sparse = False
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True

        return tags

    def check_is_fitted(self) -> bool:
        return self.is_fitted_

    def validate_data_predict(self, x):
        self.check_is_fitted()
        _ = validate_data(
            self,
            x,
            accept_sparse=True,
            reset=False,
            dtype=None,
            ensure_all_finite=False,
        )
        n = len(x)
        if n == 0:
            raise ValueError(f"Input contains 0 samples.")
        return x

    def validate_data_fit_regression(self, x, y) -> Dataset:
        _, y = validate_data(
            self,
            x,
            y,
            reset=True,
            multi_output=True,
            y_numeric=True,
            ensure_all_finite=False,
            dtype=None,
        )
        y = _check_y(y, multi_output=True, y_numeric=True, estimator=self)
        self._y_original_shape = y.shape
        y = atleast_2d(y)
        return make_dataset(self.backend, x, y)

    def validate_data_fit_classification(self, x, y) -> Dataset:
        check_classification_targets(y)
        # print("fit", x.dtypes)
        x_, y = validate_data(
            self,
            x,
            y,
            reset=True,
            multi_output=True,
            y_numeric=False,
            ensure_all_finite=False,
            dtype=None,
        )
        y = _check_y(y, multi_output=True, y_numeric=False, estimator=self)
        self.classes_ = np.unique(y)
        # y = self._encode_y(y)
        # x_df = self.get_dataframe_from_x(x)
        if len(self.classes_) < 2:
            raise ValueError("Can't train classifier with one class.")
        return make_dataset(self.backend, x, y)

    def set_model(self, model):
        self.model_ = model
        self.is_fitted_ = True


class NominalClassifier(NominalModel):
    def __init__(self, class_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight = class_weight

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        tags.classifier_tags.multi_class = True

        return tags

    def build_error(self, criterion: str, classes: int):
        errors = {
            "entropy": shared.EntropyError(classes, self.class_weight),
            "gini": shared.GiniError(classes, self.class_weight),
            "gain_ratio": shared.EntropyError(classes, self.class_weight),
        }
        if criterion not in errors.keys():
            raise ValueError(f"Unknown error function {criterion}")
        return errors[criterion]

    def predict_proba(self, x: pd.DataFrame):
        x = self.validate_data_predict(x)
        return self.model_.predict(x)

    def predict(self, x: pd.DataFrame):
        p = self.predict_proba(x)
        c = p.argmax(axis=1)
        return c


class NominalRegressor(NominalModel):

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        return tags

    def build_error(self, criterion: str):
        errors = {
            "std": shared.DeviationError(),
        }
        if criterion not in errors.keys():
            raise ValueError(f"Unknown error function {criterion}")
        return errors[criterion]

    def predict(self, x: pd.DataFrame):
        x = self.validate_data_predict(x)
        y = self.model_.predict(x)
        if len(self._y_original_shape) == 1:
            y = y.squeeze()
        return y
