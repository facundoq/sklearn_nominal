from scipy.odr import Output
from sklearn.base import BaseEstimator
from sklearn.utils import compute_class_weight
from sklearnmodels.backend import Input
from sklearnmodels.backend.core import Dataset
from sklearnmodels.backend.factory import DEFAULT_BACKEND
from sklearnmodels.rules.oner import OneR
from sklearnmodels.scikit.nominal_model import NominalClassifier, NominalRegressor


import numpy as np
import pandas as pd

from sklearnmodels.shared.target_error import TargetError


class OneRClassifier(NominalClassifier, BaseEstimator):

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags

    def __init__(self, criterion="entropy", backend=DEFAULT_BACKEND, class_weight=None):
        super().__init__(backend=backend, class_weight=class_weight)
        self.criterion = criterion

    def make_model(self, d: Dataset, class_weight: np.ndarray):
        error = self.build_error(self.criterion, class_weight)
        return OneR(error_function=error)


class OneRRegressor(NominalRegressor, BaseEstimator):

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        return tags

    def __init__(self, criterion="std", backend=DEFAULT_BACKEND):
        super().__init__(backend=backend)
        self.criterion = criterion

    def make_model(self, d: Dataset):
        error = self.build_error(self.criterion)
        return OneR(error_function=error)
