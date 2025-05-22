import sys

from scipy.odr import Output
from sklearnmodels.backend import Input
from sklearnmodels.backend.factory import DEFAULT_BACKEND
from sklearnmodels.rules.oner import OneR
from sklearnmodels.rules.prism import PRISM
from sklearnmodels.scikit.nominal_model import NominalClassifier


import numpy as np
import pandas as pd

eps = 1e-16


class PRISMClassifier(NominalClassifier):
    def __init__(
        self,
        max_rule_length: int = sys.maxsize,
        max_rules: int = sys.maxsize,
        min_rule_support=10,
        error_tolerance=eps,
        criterion="entropy",
        backend=DEFAULT_BACKEND,
        class_weight=None,
    ):
        super().__init__(backend=backend, class_weight=class_weight)
        self.max_rule_length = max_rule_length
        self.max_rules = max_rules
        self.min_rule_support = min_rule_support
        self.error_tolerance = error_tolerance
        self.criterion = criterion

    def fit(self, x: Input, y: Output):
        d = self.validate_data_fit_classification(x, y)
        error = self.build_error(self.criterion, len(self.classes_))
        trainer = PRISM(
            error,
            self.max_rule_length,
            self.max_rules,
            self.min_rule_support,
            self.error_tolerance,
        )
        model = trainer.fit(d)
        self.set_model(model)
        return self
