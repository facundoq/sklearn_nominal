from multiprocessing import Value
import numpy as np
import pandas as pd

from . import tree

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, _fit_context
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import validate_data,check_X_y
from sklearn.utils import InputTags
import abc


def atleast_2d(x):
    x = np.asanyarray(x)
    if x.ndim == 0:
        result = x.reshape(1, 1)
    elif x.ndim == 1:
        result = x[:,np.newaxis]
    else:
        result = x
    return result

class SKLearnTree(BaseEstimator, metaclass=abc.ABCMeta):
    check_parameters={"dtype":None}

    @abc.abstractmethod
    def __init__(self, criterion="",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_error_decrease=0.0,):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_error_decrease = min_error_decrease
        
    def check_is_fitted(self):
        check_is_fitted(self,"tree_")
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type="classifier"
        tags.target_tags.single_output = False
        tags.non_deterministic = False
        tags.input_tags.sparse=True
        tags.input_tags.allow_nan=True
        return tags
    def get_dataframe_from_x(self,x:np.ndarray):
        if not hasattr(self, "feature_names_in_"):
            columns = list([f"feature{i}" for i in range(self.n_features_in_)])
        else:
            columns=self.feature_names_in_
        return pd.DataFrame(x,columns=columns)
    
    def build_splitter(self):
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
        return scorers
    
    def predict_base(self, x: pd.DataFrame):
        self.check_is_fitted()
        x = validate_data(self, x, accept_sparse=True, reset=False,dtype=None,ensure_all_finite=False)
        x_df = self.get_dataframe_from_x(x)
        n = len(x_df)
        assert n > 0
        predictions = np.zeros((n, len(self.tree_.prediction)))
        for i, row in x_df.iterrows():
            predictions[i, :] = self.tree_.predict(row)
        return predictions


class SKLearnClassificationTree(ClassifierMixin, SKLearnTree):
    def __init__(
        self,
        criterion="entropy",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_error_decrease=0.0,
        class_weight=None,
    ):
        super().__init__(criterion=criterion,splitter=splitter,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,min_error_decrease=min_error_decrease)
        self.class_weight = class_weight


    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type="classifier"
        return tags
    
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
        check_X_params = dict(dtype=None, accept_sparse=True, ensure_all_finite=False,ensure_2d=True)
        check_y_params = dict(ensure_2d=False)
        x, y = validate_data(self, x, y, reset=True,multi_output=False,y_numeric=True, validate_separately=(check_X_params, check_y_params))

        x_df =self.get_dataframe_from_x(x)
        self.classes_ = unique_labels(y)
        error = self.build_error(len(self.classes_))
        trainer = self.build_trainer(error)
        self.tree_ = trainer.fit(x_df, y)
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

class SKLearnRegressionTree(RegressorMixin, SKLearnTree):
    def __init__(
        self,
        criterion="std",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_error_decrease=0.0,
    ):
        super().__init__(criterion=criterion,splitter=splitter,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,min_error_decrease=min_error_decrease)

    
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type="regressor"
        return tags
    
    def build_trainer(self, error: tree.TargetError):
        
        scorers = self.build_splitter()
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
        check_X_params = dict(dtype=None, accept_sparse=True, ensure_all_finite=False,ensure_2d=True)
        check_y_params = dict(ensure_2d=False,ensure_all_finite=True)
        x, y = validate_data(self, x, y, reset=True,multi_output=True,y_numeric=True, validate_separately=(check_X_params, check_y_params))
        check_X_y(x,y,accept_sparse=False,dtype=None)
        y = atleast_2d(y)
        x_df = self.get_dataframe_from_x(x)
        error = self.build_error()
        trainer = self.build_trainer(error)
        self.tree_ = trainer.fit(x_df, y)
        self.is_fitted_ = True
        return self

    def build_error(self,):
        errors = {
            "std": tree.DeviationMetric(),
        }
        if self.criterion not in errors.keys():
            raise ValueError(f"Unknown error function {self.criterion}")
        return errors[self.criterion]
    
    def predict(self, x):
        return self.predict_base(x)

