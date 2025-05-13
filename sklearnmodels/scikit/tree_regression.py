from .tree_scikit import SKLearnTree
from ..scikit.nominal_model import NominalRegressor
from ..tree.pruning import PruneCriteria
from sklearnmodels import tree, shared
import numpy as np
import pandas as pd


class SKLearnRegressionTree(NominalRegressor, SKLearnTree):
    def __init__(
        self,
        criterion="std",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_error_decrease=0.0,
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

    def build_trainer(self, error: shared.TargetError):
        column_penalization = self.build_attribute_penalizer()
        scorers = self.build_splitter(error, column_penalization)
        scorer = shared.DefaultSplitter(error, scorers)
        prune_criteria = PruneCriteria(
            max_height=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_error_decrease=self.min_error_decrease,
            min_samples_split=self.min_samples_split,
        )
        trainer = tree.BaseTreeTrainer(scorer, prune_criteria)
        return trainer

    def fit(self, x: pd.DataFrame, y: np.ndarray):
        d = self.validate_data_fit_regression(x, y)

        error = self.build_error(self.criterion)
        trainer = self.build_trainer(error)

        model = trainer.fit(d)
        self.set_model(model)
        return self
