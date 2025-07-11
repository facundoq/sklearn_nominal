import sys

import numpy as np
import pandas as pd
from scipy.odr import Output
from sklearn.base import BaseEstimator
from sklearn.utils import compute_class_weight

from sklearn_nominal.backend import Input
from sklearn_nominal.backend.core import Dataset
from sklearn_nominal.backend.factory import DEFAULT_BACKEND
from sklearn_nominal.rules.cn2 import CN2
from sklearn_nominal.rules.oner import OneR
from sklearn_nominal.rules.prism import PRISM
from sklearn_nominal.sklearn.nominal_model import NominalClassifier, NominalRegressor

eps = 1e-16


class CN2Classifier(NominalClassifier, BaseEstimator):
    """A rule-based classifier that performs sequential covering in a CN2 [1] style.

    [1]  Clark, P. and Niblett, T (1989) The CN2 induction algorithm. Machine Learning 3(4):261-283.

    Parameters
    ----------
    criterion : {"gini", "entropy", "log_loss"}, default="entropy"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "log_loss" and "entropy" both for the
        Shannon information gain.

    max_rule_length: int, default= sys.maxsize
        The maximum number of conditions in a rule. Analogous to the maximum height of a Tree model.

    max_rules: int, default=sys.maxsize
        Maximum number of rules for the model. Analogous to the maximum number of leaves in a Tree model.
    min_rule_support:int, default=10
        Minimum number of samples that satisfy the condition of a rule required to include that rule in the model. Analogous to the `min_samples_leaf` parameter for Tree models.
    max_error_per_rule:float, default=0.99
        Maximum (absolute) error that the rule can have. This value depends on the error (:param: criterion) used for the model.

    class_weight : dict, list of dict or "balanced", default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If None, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    n_classes_ : int or list of int
        The number of classes (for single output problems),

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    model_ : :class:`sklearn_nominal.rules.model.RuleModel` instance
        The underlying model object.

    See Also
    --------
    TreeRegressor : A decision tree regressor with nominal support.
    NaiveBayesClassifier: a NaiveBayesClassifier with nominal support.
    ZeroRClassifier: a ZeroR classifier with nominal support.
    OneRClassifier: a OneR classifier with nominal support.
    PRISMClassifier: a PRISM classifier with nominal support.

    Examples
    --------
    >>> from sklearn.datasets import fetch_openml
    >>> df = fetch_openml("credit-g",version=2).frame
    >>> x,y = df.iloc[:,0:-1], df.iloc[:,-1]
    >>>
    >>> from sklearn_nominal import CN2Classifier
    >>> model = CN2Classifier()
    >>> model.fit(x,y)
    >>>
    >>> from sklearn.metrics import accuracy_score
    >>> y_pred = model.predict(x)
    >>> print(accuracy_score(y,y_pred))
    ... 0.787
    """

    def __init__(
        self,
        criterion="entropy",
        max_rule_length: int = sys.maxsize,
        max_rules: int = sys.maxsize,
        min_rule_support=10,
        max_error_per_rule=0.99,
        backend=DEFAULT_BACKEND,
        class_weight: np.ndarray | None = None,
    ):
        super().__init__(backend=backend, class_weight=class_weight)
        self.max_rule_length = max_rule_length
        self.max_rules = max_rules
        self.min_rule_support = min_rule_support
        self.max_error_per_rule = max_error_per_rule
        self.criterion = criterion

    def make_model(self, d: Dataset, class_weight: np.ndarray):
        error = self.build_error(self.criterion, class_weight)
        return CN2(
            error,
            self.max_rule_length,
            self.max_rules,
            self.min_rule_support,
            self.max_error_per_rule,
        )


class CN2Regressor(NominalRegressor, BaseEstimator):
    def __init__(
        self,
        criterion="std",
        max_rule_length: int = sys.maxsize,
        max_rules: int = sys.maxsize,
        min_rule_support=10,
        max_error_per_rule=0.99,
        backend=DEFAULT_BACKEND,
    ):
        super().__init__(backend=backend)
        self.max_rule_length = max_rule_length
        self.max_rules = max_rules
        self.min_rule_support = min_rule_support
        self.max_error_per_rule = max_error_per_rule
        self.criterion = criterion

    def make_model(self, d: Dataset):
        error = self.build_error(self.criterion)
        return CN2(
            error,
            self.max_rule_length,
            self.max_rules,
            self.min_rule_support,
            self.max_error_per_rule,
        )
