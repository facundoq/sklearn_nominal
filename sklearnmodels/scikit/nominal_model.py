import abc
from socket import has_dualstack_ipv6

from matplotlib.pylab import isinteractive
import numpy as np
import pandas as pd
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils._tags import (
    ClassifierTags,
    RegressorTags,
    Tags,
    TargetTags,
    TransformerTags,
    get_tags,
)
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_y, validate_data
from sklearn.utils import validation

from sklearnmodels.backend import Input, Output
from sklearnmodels.backend.factory import make_dataset
from sklearnmodels.shared.target_error import TargetError

from .. import shared, tree
from ..backend.core import Dataset
from ..backend.pandas import PandasDataset


def atleast_2d(x):
    x = np.asanyarray(x)
    if x.ndim == 0:
        result = x.reshape(1, 1)
    elif x.ndim == 1:
        result = x[:, np.newaxis]
    else:
        result = x
    return result


class NominalModel(metaclass=abc.ABCMeta):
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
        validation.check_is_fitted(self)
        if not self.is_fitted_:
            raise NotFittedError()

    def validate_data_predict(self, x):
        self.check_is_fitted()
        x = validate_data(
            self,
            x,
            reset=False,
            dtype=None,
            ensure_all_finite=False,
            accept_sparse=False,
        )
        n = len(x)
        if n == 0:
            raise ValueError(f"Input contains 0 samples.")
        return pd.DataFrame(x, columns=self.get_feature_names(), dtype=None)

    def validate_data_fit_regression(self, x, y) -> Dataset:
        if hasattr(x, "dtypes"):
            dtype = x.dtypes
        else:
            dtype = None

        x, y = validate_data(
            self,
            x,
            y,
            reset=True,
            multi_output=True,
            y_numeric=True,
            ensure_all_finite=False,
            dtype=None,
            accept_sparse=False,
        )
        y = _check_y(y, multi_output=True, y_numeric=True, estimator=self)
        self._y_original_shape = y.shape
        y = atleast_2d(y)

        return make_dataset(self.backend, x, y, self.get_feature_names(), dtype)

    def get_feature_names(self):
        if not hasattr(self, "feature_names_in_"):
            return [f"f{i}" for i in range(self.n_features_in_)]
        else:
            return self.feature_names_in_

    def set_dtypes(self, x):
        if isinstance(x, pd.DataFrame):
            self.dtypes_ = x.dtypes
        elif isinstance(x, np.ndarray) or scipy.sparse.issparse(x):
            assert len(x.shape) == 2, f"Expected 2d input, actual shape {x.shape}"
            self.dtypes_ = [x.dtype] * x.shape[1]
        else:
            raise ValueError(
                f"Only pd.Dataframe or np.ndarray supported, received: {x}"
            )

    def get_y(self, y):
        y = _check_y(y, multi_output=True, y_numeric=False, estimator=self)
        # TODO make pure numpy
        y_cat = pd.Series(data=y.squeeze()).astype("category")
        y = y_cat.cat.codes.to_numpy()
        return y

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
            accept_sparse=False,
        )

        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("Can't train classifier with one class.")
        y = self.get_y(y)
        # dtype = x_original.dtype
        return make_dataset(self.backend, x, y, self.get_feature_names(), None)

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

    def build_error(self, criterion: str, classes: int) -> TargetError:
        errors = {
            "entropy": shared.EntropyError(classes, self.class_weight),
            "gini": shared.GiniError(classes, self.class_weight),
            "gain_ratio": shared.EntropyError(classes, self.class_weight),
        }
        if criterion not in errors.keys():
            raise ValueError(f"Unknown error function {criterion}")
        return errors[criterion]

    def predict_proba(self, x: Input) -> Output:
        self.check_is_fitted()
        x = self.validate_data_predict(x)
        return self.model_.predict(x)

    def predict(self, x: Input) -> Output:
        p = self.predict_proba(x)
        c = p.argmax(axis=1)
        return c


class NominalRegressor(NominalModel):

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.regressor_tags = RegressorTags()
        tags.target_tags.required = True
        return tags

    def build_error(self, criterion: str):
        errors = {
            "std": shared.DeviationError(),
        }
        if criterion not in errors.keys():
            raise ValueError(f"Unknown error function {criterion}")
        return errors[criterion]

    def predict(self, x: Input):
        self.check_is_fitted()
        x = self.validate_data_predict(x)
        y = self.model_.predict(x)
        if len(self._y_original_shape) == 1:
            y = y.squeeze()
        return y
