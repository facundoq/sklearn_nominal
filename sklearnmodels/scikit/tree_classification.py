from scipy.odr import Output
from sklearn.base import BaseEstimator
from sklearnmodels.backend import Input
from .tree_base import SKLearnTree
from ..scikit.nominal_model import NominalClassifier
from sklearnmodels import tree, shared

import numpy as np
import pandas as pd


class SKLearnClassificationTree(NominalClassifier, SKLearnTree, BaseEstimator):
    def __init__(
        self,
        criterion="entropy",
        splitter="best",
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=5,
        min_error_decrease=0.0,
        class_weight=None,
        backend="pandas",
    ):
        super().__init__(
            class_weight=class_weight,
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_error_decrease=min_error_decrease,
            backend=backend,
        )

    def build_trainer(self, error: shared.TargetError):
        column_penalization = self.build_attribute_penalizer()

        scorers = self.build_splitter(error, column_penalization)

        scorer = shared.DefaultSplitter(error, scorers)
        prune_criteria = tree.pruning.PruneCriteria(
            max_height=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_error_decrease=self.min_error_decrease,
            min_samples_split=self.min_samples_split,
        )
        trainer = tree.BaseTreeTrainer(scorer, prune_criteria)
        return trainer

    def fit(self, x: Input, y: Output):

        d = self.validate_data_fit_classification(x, y)
        error = self.build_error(self.criterion, len(self.classes_))
        trainer = self.build_trainer(error)
        model = trainer.fit(d)
        self.set_model(model)
        return self
