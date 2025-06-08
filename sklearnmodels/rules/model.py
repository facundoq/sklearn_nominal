import numpy as np
import pandas as pd

from sklearnmodels.backend import Input, InputSample, Output
from sklearnmodels.backend.conditions import Condition
from sklearnmodels.backend.core import Model

PredictionRule = tuple[Condition, Output]


class RuleModel(Model):
    def __init__(
        self,
        rules: list[PredictionRule],
        default_prediction: Output,
    ):
        self.default_prediction = default_prediction
        self.rules = rules

    def output_size(self):
        return len(self.default_prediction)

    def predict_sample(self, x: InputSample):
        for condition, p in self.rules:
            if condition(x):
                return p
        return self.default_prediction

    def __repr__(self):
        return f"RuleModel(rules={len(self.rules)},p={self.default_prediction})"

    def complexity(self):
        return len(self.rules) + 1

    def pretty_print(self, class_names: list[str] = None):
        def format(p):
            if class_names is None:
                return str(p)
            else:
                return class_names[p.argmax()]

        rules = "\n".join([f"{c} => {format(p)}" for c, p in self.rules])
        rules += "\n" if rules != "" else ""
        return f"{rules}Default: {self.default_prediction}"
