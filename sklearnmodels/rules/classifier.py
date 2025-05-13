import numpy as np
import pandas as pd

from sklearnmodels.backend.conditions import Condition

ClassificationRule = tuple[Condition, np.ndarray]


class RuleClassifier:
    def __init__(
        self,
        rules: list[ClassificationRule],
        class_names: list[str],
        default_prediction: np.ndarray = None,
    ):
        if default_prediction is None:
            c = len(class_names)
            default_prediction = np.ones(c) / c

        self.default_prediction = default_prediction
        self.rules = rules
        self.class_names = class_names

    def predict_proba(self, x: pd.Series):
        for condition, p in self.rules:
            if condition(x):
                return p
        return self.default_prediction

    def predict(self, x: pd.DataFrame):
        n = x.shape[0]

        predictions = np.zeros((n, len(self.default_prediction)))
        for i, (idx, row) in enumerate(x.iterrows()):
            predictions[i, :] = self.predict_sample(row)
        return predictions

    def predict_sample(self, x: pd.Series):
        return self.predict_proba(x)
