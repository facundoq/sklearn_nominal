import numpy as np
import pandas as pd
from scipy.odr import Output
from sklearn.base import BaseEstimator

from sklearn_nominal import shared, tree
from sklearn_nominal.backend import Input
from sklearn_nominal.backend.core import Dataset

from ..scikit.nominal_model import NominalRegressor
from ..tree.pruning import PruneCriteria
from .tree_base import BaseTree


class TreeRegressor(NominalRegressor, BaseTree, BaseEstimator):
    def __init__(
        self,
        criterion="std",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_error_decrease=1e-16,
        backend="pandas",
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_error_decrease=min_error_decrease,
            backend=backend,
        )

    def make_model(self, d: Dataset):
        error = self.build_error(self.criterion)
        column_penalization = self.build_attribute_penalizer()
        scorers = self.build_splitter(error, column_penalization)
        scorer = shared.DefaultSplitter(error, scorers)
        prune_criteria = self.make_prune_criteria()
        trainer = tree.BaseTreeTrainer(scorer, prune_criteria)
        return trainer
