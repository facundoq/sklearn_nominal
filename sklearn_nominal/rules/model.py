import numpy as np
import pandas as pd

from sklearn_nominal.backend import Input, InputSample, Output
from sklearn_nominal.backend.conditions import Condition
from sklearn_nominal.backend.core import Model

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
        df = pd.DataFrame([x])
        return self.predict(df)[0, :]

    def predict(self, x: pd.DataFrame):
        n = len(x)
        predictions = np.zeros((n, self.output_size()))
        remaining = np.ones(n, dtype=bool)
        for condition, p in self.rules:
            mask = condition(x) & remaining
            predictions[mask] = p
            remaining &= ~mask
        predictions[remaining] = self.default_prediction
        return predictions

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

        if len(self.rules) > 0:
            conditions_str = list(zip(*self.rules))[0]
            pad = max(map(len, map(str, conditions_str))) + 2
            rules = "\n".join([f"{str(c):{pad}} => {format(p)}" for c, p in self.rules])
            rules += "\n" if rules != "" else ""
        else:
            rules = ""
        return f"{rules}Default: {self.default_prediction}"

    def __eq__(self, x):
        if not isinstance(x, RuleModel):
            return False
        if self.default_prediction.shape != x.default_prediction.shape or (
            not np.allclose(self.default_prediction, x.default_prediction, atol=1e-8)
        ):
            return False
        if len(self.rules) != len(x.rules):
            return False
        if len(self.rules) == 0:
            return True
        conditions, predictions = zip(*self.rules)
        x_conditions, x_predictions = zip(*x.rules)
        return conditions == x_conditions and all([np.allclose(a, b) for a, b in zip(predictions, x_predictions)])
