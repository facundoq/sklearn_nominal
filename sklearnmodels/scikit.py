from multiprocessing import Value
import numpy as np
import pandas as pd

from . import tree

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, _fit_context
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import validate_data 
from sklearn.utils import InputTags

class SKLearnTree():
    def __init__(self):
        self.tree_: tree.Tree | None = None

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type="classifier"
        tags.target_tags.single_output = False
        tags.non_deterministic = False
        tags.input= InputTags(sparse=True)
        return tags
    
    def predict_base(self, x: pd.DataFrame):
        check_is_fitted(self)
        x = validate_data(x, accept_sparse=True, reset=False)
        n = len(x)
        assert n > 0
        predictions = np.zeros((n, len(self.tree_.prediction)))
        for i, row in x.iterrows():
            predictions[i, :] = self.tree_.predict(row)
        return predictions


class SKLearnClassificationTree(ClassifierMixin, BaseEstimator,SKLearnTree):
    def __init__(
        self,
        criterion="entropy",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        min_error_decrease=0.0,
        class_weight=None,
    ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_error_decrease = min_error_decrease
        self.class_weight = class_weight

    def build_trainer(self, error: tree.TargetError):
        if self.splitter =="best":
            max_evals = np.iinfo(np.int64).max 
        elif isinstance(self.splitter,int):
            max_evals=self.splitter
        else:
            raise ValueError(f"Invalid value '{self.splitter}' for splitter; expected integer or 'best'")
        scorers = {
            "number": tree.DiscretizingNumericColumnSplitter(
                tree.OptimizingDiscretizationStrategy(max_evals=max_evals)
            ),
            "object": tree.NominalColumnSplitter(),
            "category": tree.NominalColumnSplitter(),
        }

        scorer = tree.MixedGlobalError(scorers, error)
        prune_criteria = tree.PruneCriteria(
            max_height=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_error_decrease=self.min_error_decrease,
            min_samples_split=self.min_samples_split,
        )
        trainer = tree.BaseTreeTrainer(scorer, prune_criteria)
        return trainer
    
    
    def fit(self, x: pd.DataFrame, y: np.ndarray):
        
        x, y = validate_data(self, x, y, accept_sparse=False)
        self.classes_ = unique_labels(y)
        error = self.build_error(len(self.classes_))
        trainer = self.build_trainer(error)
        self.tree_ = trainer.fit(x, y)
        self.is_fitted_ = True
        return self

    def build_error(self, classes: int):
        errors = {
            "entropy": tree.EntropyMetric(classes, self.class_weight),
        }
        if self.criterion not in errors.keys():
            raise ValueError(f"Unknown error function {self.criterion}")
        return errors[self.criterion]

    

    def predict_proba(self, x: pd.DataFrame):
        return self.predict_base(x)

    def predict(self, x: pd.DataFrame):
        return self.predict_proba(x).argmax(axis=1)


class SKLearnRegressionTree(RegressorMixin, BaseEstimator, SKLearnTree):
    def predict(self, x: pd.DataFrame):
        return self.predict_base(x)
