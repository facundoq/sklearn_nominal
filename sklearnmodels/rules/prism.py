import sys

from h11 import Data
from .core import ColumnID
import numpy as np
from sklearnmodels.backend.conditions import (
    Condition,
    NotCondition,
    RangeCondition,
    TrueCondition,
    ValueCondition,
)
from sklearnmodels.backend.core import ColumnType, Dataset
from sklearnmodels.rules.model import PredictionRule, RuleModel
from sklearnmodels.shared.global_error import DefaultSplitter
from sklearnmodels.shared.target_error import TargetError


class PRISM:

    def __init__(
        self,
        target_error: TargetError,
        max_length: int,
        max_rules: int,
        min_rule_support: int,
        error_tolerance: float,
    ):
        self.max_length = max_length
        self.max_rules = max_rules
        self.target_error = target_error
        self.min_rule_support = min_rule_support
        self.splitter = DefaultSplitter(target_error)
        self.max_error = error_tolerance

    def fit(self, d: Dataset):
        rules = []
        classes = d.classes()
        for klass in classes:

            rules += self.fit_subset(d, klass, classes)
        return RuleModel(rules, self.target_error.prediction(d))

    def fit_subset(self, d: Dataset, klass: int, classes: list[int]):
        rules = []

        while d.filter_by_class(klass).n > self.min_rule_support:
            rule = self.generate_rule(d, klass, classes)
            if rule is None:
                break  # unable to generate rule; stop process
            rules.append(rule)
            condition, prediction = rule
            # keep samples that do not match the condition
            d = d.filter(NotCondition(condition))
        return rules

    def generate_rule(
        self, d: Dataset, klass: int, classes: list[int]
    ) -> None | PredictionRule:

        conditions = []
        error = np.inf
        while len(conditions) < self.max_length and error > self.max_error:
            condition, error, drop_column = self.improve_rule(d, klass, classes)
            if condition is None:
                break
            conditions.append(condition)
            d = d.filter(condition)
            if drop_column:
                d = d.drop(condition.column)

        if error >= self.max_error:
            return None

        prediction = np.zeros(len(classes))
        prediction[klass] = 1
        return (conditions, prediction)

    def generate_conditions(self, d: Dataset) -> list[tuple[Condition, bool]]:
        for column in d.columns:
            column_type = d.types(column)
            if column_type == ColumnType.Nominal:
                return [(ValueCondition(column, v), True) for v in d.values(column)]
            elif column_type == ColumnType.Numeric:
                v = d.mean(column)
                l = [False, True]
                return [([RangeCondition(column, v, less)], False) for less in l]
            else:
                raise ValueError(f"Invalid column type")

    def improve_rule(self, d: Dataset, klass: int, classes: list[int]):
        conditions_drops = self.generate_conditions(d)
        conditions, drops = zip(**conditions_drops)
        best_error = np.inf  # TODO make object with three fields
        best_condition = None
        best_drop = None
        for condition, drop, d_condition in zip(conditions, drops, d.split(conditions)):
            error = 1 - d_condition.count_class(klass) / d_condition.n
            if error < best_error:
                best_error = error
                best_condition = condition
                best_drop = drop
        return best_condition, best_error, best_drop
