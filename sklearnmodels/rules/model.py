import numpy as np
import pandas as pd

from sklearnmodels.backend import Input, InputSample, Output
from sklearnmodels.backend.conditions import Condition

PredictionRule = tuple[Condition, Output]


class RuleModel:
    def __init__(
        self,
        rules: list[PredictionRule],
        default_prediction: Output,
    ):
        self.default_prediction = default_prediction
        self.rules = rules

    def predict_proba(self, x: InputSample):
        for condition, p in self.rules:
            if condition(x):
                return p
        return self.default_prediction

    def predict(self, x: Input):
        n = x.shape[0]

        predictions = np.zeros((n, len(self.default_prediction)))
        for i, (idx, row) in enumerate(x.iterrows()):
            predictions[i, :] = self.predict_sample(row)
        return predictions

    def predict_sample(self, x: InputSample):
        return self.predict_proba(x)

    def __repr__(self):
        return f"RuleModel(rules={len(self.rules)},p={self.default_prediction})"

    def description(self):
        rules = "\n".join([f"{c} => {p}" for c, p in self.rules])
        return f"{rules}\nDefault: {self.default_prediction}"
