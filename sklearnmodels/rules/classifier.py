import numpy as np
import pandas as pd

from sklearnmodels.backend.conditions import Condition


class Predicate(Condition):
    def __init__(self, conditions: list[Condition]):
        self.conditions = conditions

    def short_description(self):
        descriptions = [c.short_description() for c in self.conditions]
        return "Predicate: " + (",".join(descriptions))

    def __call__(self, x: pd.Series):
        for c in self.conditions:
            if not c(x):
                return False
        return True


class RuleClassifier:
    def __init__(self, rules: dict[Condition, np.ndarray], class_names: list[str]):
        self.rules = rules
        self.class_names = class_names

    def predict_proba(self, x: pd.Series):

        for condition, p in self.rules.items():
            if condition(x):
                return p
        c = len(self.class_names)
        return np.ones(c) / c

    def predict(self, x: pd.Series):
        return self.predict_proba(x).argmax()
