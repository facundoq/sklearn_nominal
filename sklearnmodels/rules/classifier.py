import numpy as np
import pandas as pd

from sklearnmodels.backend.conditions import Condition

ClassificationRule = tuple[Condition, np.ndarray]


class RuleClassifier:
    def __init__(self, rules: list[ClassificationRule], class_names: list[str]):
        self.rules = rules
        self.class_names = class_names

    def predict_proba(self, x: pd.Series):

        for condition, p in self.rules:
            if condition(x):
                return p
        c = len(self.class_names)
        return np.ones(c) / c

    def predict(self, x: pd.Series):
        return self.predict_proba(x).argmax()
