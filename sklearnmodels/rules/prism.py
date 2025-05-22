from os import error
import sys

import numpy as np
from sklearnmodels.backend.conditions import NotCondition, TrueCondition
from sklearnmodels.backend.core import Dataset
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
        for c in d.classes():
            d_class = d.filter_by_class(c)
            rules += self.fit_subset(d_class)
        return RuleModel(rules, self.target_error.prediction(d))

    def fit_subset(self, d: Dataset):
        rules = []

        while d.n > self.min_rule_support:
            rule = self.generate_rule(d)
            if rule is None:
                break  # unable to generate rule; stop process
            rules.append(rule)
            condition, prediction = rule
            # keep samples that do not match the condition
            d = d.filter(NotCondition(condition))
        return rules

    def generate_rule(self, d: Dataset) -> None | PredictionRule:

        conditions = []
        error = np.inf
        while len(conditions) < self.max_length and error > self.max_error:
            proposal, error, drop_column = None
            if proposal is None:
                break
            d = d.filter(proposal)
            if drop_column:
                d = d.drop(proposal.column)

        if error > self.max_error:
            return None

        return (conditions, self.target_error.prediction(d))
