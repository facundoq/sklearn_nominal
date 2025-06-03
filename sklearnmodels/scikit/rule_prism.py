import sys

from scipy.odr import Output
from sklearn.base import BaseEstimator
from sklearnmodels.backend import Input
from sklearnmodels.backend.factory import DEFAULT_BACKEND
from sklearnmodels.rules.oner import OneR
from sklearnmodels.rules.prism import PRISM
from sklearnmodels.scikit.nominal_model import NominalClassifier

import numpy as np
import pandas as pd

eps = 1e-16


class PRISMClassifier(NominalClassifier, BaseEstimator):
    def __init__(
        self,
        max_rule_length: int = sys.maxsize,
        max_rules: int = sys.maxsize,
        min_rule_support=10,
        error_tolerance=eps,
        backend=DEFAULT_BACKEND,
        class_weight: np.ndarray | None = None,
    ):
        super().__init__(backend=backend, class_weight=class_weight)
        self.max_rule_length = max_rule_length
        self.max_rules = max_rules
        self.min_rule_support = min_rule_support
        self.error_tolerance = error_tolerance

    def fit(self, x: Input, y: Output):
        d = self.validate_data_fit_classification(x, y)
        nc = len(d.classes())
        if self.class_weight is not None:
            class_weight = self.class_weight
        else:
            class_weight = np.ones(nc) / nc

        trainer = PRISM(
            class_weight,
            self.max_rule_length,
            self.max_rules,
            self.min_rule_support,
            self.error_tolerance,
        )
        model = trainer.fit(d)
        self.set_model(model)
        return self
