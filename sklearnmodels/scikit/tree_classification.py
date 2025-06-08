from scipy.odr import Output
from sklearn.base import BaseEstimator
from sklearn.utils import compute_class_weight
from sklearnmodels.backend import Input
from sklearnmodels.backend.core import Dataset
from .tree_base import BaseTree
from ..scikit.nominal_model import NominalClassifier
from sklearnmodels import tree, shared

import numpy as np
import pandas as pd


class TreeClassifier(NominalClassifier, BaseTree, BaseEstimator):
    def __init__(
        self,
        criterion="entropy",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_error_decrease=1e-16,
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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags

    def make_model(self, d: Dataset, class_weight: np.ndarray):
        error = self.build_error(self.criterion, class_weight)
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
