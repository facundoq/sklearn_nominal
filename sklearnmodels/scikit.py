from multiprocessing import Value
import numpy as np
import pandas as pd

from . import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, _fit_context
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import validation 
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import validate_data,check_X_y,_check_y
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
        validation.check_is_fitted(self,"tree_")
        
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.single_output = False
        tags.non_deterministic = False
        tags.input_tags.sparse=False
        tags.input_tags.allow_nan=True
        tags.input_tags.string = True
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
        min_samples_split=5,
        min_samples_leaf=5,
        min_error_decrease=0.0,
        class_weight=None,
    ):
        super().__init__(criterion=criterion,splitter=splitter,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,min_error_decrease=min_error_decrease)
        self.class_weight = class_weight


    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type="classifier"
        tags.classifier_tags.multi_class=True
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
    
    def _encode_y(self,y:np.ndarray):
        le = LabelEncoder()
        encoded_y = le.fit_transform(y)
        self.classes_ = le.classes_
        if self.class_weight is not None:
            # get the classes in the same order as assigned by the transform
            ordered_classes = le.transform(le.inverse_transform(le.classes_))
            # print(self.class_weight)
            # print(le.classes_.dtype,ordered_classes.dtype,type(ordered_classes[0]))
            # use the ordered classes to obtain ordered class weights
            self._ordered_class_weights = np.array([self.class_weight[c] for c in ordered_classes])
        else:
            self._ordered_class_weights = None
        return encoded_y
    
    def fit(self, x: pd.DataFrame, y: np.ndarray):
        check_classification_targets(y)
        # check_X_params = dict(dtype=None, accept_sparse=False, ensure_all_finite=False,ensure_2d=True)
        # check_y_params = dict(ensure_2d=False,accept_sparse=False)
        # x, y = validate_data(self, x, y, reset=True,multi_output=False,y_numeric=True, validate_separately=(check_X_params, check_y_params))
        # check_X_y(x,y,accept_sparse=False,dtype=None,ensure_all_finite=False)
        x, y = validate_data(self, x, y, reset=True,multi_output=True,y_numeric=False, ensure_all_finite=False,dtype=None)
        y = _check_y(y,multi_output=True, y_numeric=False,estimator=self)
        y = self._encode_y(y)
        x_df =self.get_dataframe_from_x(x)
        if len(self.classes_)<2:
            raise ValueError("Can't train classifier with one class.")
        error = self.build_error(len(self.classes_))
        trainer = self.build_trainer(error)
        self.tree_ = trainer.fit(x_df, y)
        self.is_fitted_ = True
        return self

    def build_error(self, classes: int):
        errors = {
            "entropy": tree.EntropyMetric(classes, self._ordered_class_weights),
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
        # check_X_params = dict(dtype=None, accept_sparse=False, ensure_all_finite=False,ensure_2d=True)
        # check_y_params = dict(ensure_2d=False,ensure_all_finite=True)
        # x, y = validate_data(self, x, y, reset=True,multi_output=True,y_numeric=True, validate_separately=(check_X_params, check_y_params))
        # check_X_y(x,y,accept_sparse=False,dtype=None,ensure_all_finite=False)
        x, y = validate_data(self, x, y, reset=True,multi_output=True,y_numeric=True, ensure_all_finite=False,dtype=None)
        y = _check_y(y,multi_output=True, y_numeric=True,estimator=self)
        self._y_original_shape=y.shape
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
        y = self.predict_base(x)
        if len(self._y_original_shape)==1:
            y = y.squeeze()
        return y



class TemplateClassifier(ClassifierMixin, BaseEstimator):
    """An example classifier which implements a 1-NN algorithm.

    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.

    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.

    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from skltemplate import TemplateClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = TemplateClassifier().fit(X, y)
    >>> clf.predict(X)
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "demo_param": [str],
    }

    def __init__(self, demo_param="demo"):
        self.demo_param = demo_param

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # `_validate_data` is defined in the `BaseEstimator` class.
        # It allows to:
        # - run different checks on the input data;
        # - define some attributes associated to the input data: `n_features_in_` and
        #   `feature_names_in_`.
        X, y = validate_data(self,X, y)
        # We need to make sure that we have a classification task
        check_classification_targets(y)

        # classifier should always store the classes seen during `fit`
        self.classes_ = np.unique(y)

        # Store the training data to predict later
        self.X_ = X
        self.y_ = y

        # Return the classifier
        return self

    def predict(self, X):
        """A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check if fit had been called
        validation.check_is_fitted(self)

        # Input validation
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        X = self._validate_data(X, reset=False)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]
