import sys

import numpy as np
from sklearnmodels.backend.conditions import TrueCondition
from sklearnmodels.backend.core import Dataset
from sklearnmodels.rules.classifier import ClassificationRule, RuleClassifier
from sklearnmodels.tree.global_error import DefaultSplitter
from sklearnmodels.tree.target_error import TargetError

eps = 1e-16


class PRISMTrainer:

    def __init__(
        self,
        target_error: TargetError,
        max_length: int = sys.maxsize,
        max_rules: int = sys.maxsize,
        min_rule_support=10,
        error_tolerance=eps,
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
        return RuleClassifier(rules, d.classes())

    def fit_subset(self, d: Dataset):
        rules = []

        while d.n > self.min_rule_support:
            rule = self.generate_rule(d)
            if rule is None:
                break
            rules.append(rule)
            d = d.without(rule[0])

    def generate_rule(self, d: Dataset) -> None | ClassificationRule:
        pass

    #     conditions = []
    #     error = np.inf
    #     while len(conditions)<self.max_length and error>self.max_error:

    # def propose_conditions(self,d:Dataset):


class ZeroR:

    def fit(self, d: Dataset):
        rules = [(TrueCondition(""), d.class_distribution())]
        return RuleClassifier(rules, d.classes())


class OneR:
    def __init__(self, error_function: TargetError):
        self.error_function = error_function
        self.splitter = DefaultSplitter(error_function)

    def fit(self, d: Dataset):
        e = self.splitter.split_columns(d)
        rules = [(e, p.class_distribution()) for e, p in zip(e.conditions, e.partition)]
        return RuleClassifier(rules, d.classes())
