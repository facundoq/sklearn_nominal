import sys

import numpy as np
import pandas as pd
from scipy.odr import Output
from sklearn.base import BaseEstimator
from sklearn.utils import compute_class_weight

from sklearn_nominal.backend import Input
from sklearn_nominal.backend.core import Dataset
from sklearn_nominal.backend.factory import DEFAULT_BACKEND
from sklearn_nominal.rules.oner import OneR
from sklearn_nominal.rules.prism import PRISM
from sklearn_nominal.sklearn.nominal_model import NominalClassifier

eps = 1e-16


class PRISMClassifier(NominalClassifier, BaseEstimator):
    """A PRISM classifier.

    [1] Chendrowska, J. (1987) PRISM: An Algorithm for Inducing Modular Rules. International Journal of Man-Machine Studies, vol 27, pp. 349-370.

    [2] Chendrowska, J. (1990) Knowledge Acquisition for Expert Systems: Inducing Modular Rules from Examples. PhD Thesis, The Open University.

    [3] Bramer, M. (2007) Principles of Data Mining, Springer Press.

    Parameters
    ----------
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
    CN2Classifier: a CN2Classifier classifier with nominal support.
    PRISMClassifier: a PRISM classifier with nominal support.
    OneRClassifier: a OneR classifier with nominal support.

    Examples
    --------
    >>> from sklearn.datasets import fetch_openml
    >>> df = fetch_openml("credit-g",version=2).frame
    >>> x,y = df.iloc[:,0:-1], df.iloc[:,-1]
    >>>
    >>> from sklearn_nominal import PRISMClassifier
    >>> model = PRISMClassifier()
    >>> model.fit(x,y)
    >>>
    >>> from sklearn.metrics import accuracy_score
    >>> y_pred = model.predict(x)
    >>> print(accuracy_score(y,y_pred))
    ... 0.787
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags

    def __init__(
        self,
        max_rule_length: int = sys.maxsize,
        max_rules_per_class: int = sys.maxsize,
        min_rule_support=10,
        max_error_per_rule=1,
        backend=DEFAULT_BACKEND,
        class_weight: np.ndarray | None = None,
    ):
        super().__init__(backend=backend, class_weight=class_weight)
        self.max_rule_length = max_rule_length
        self.max_rules_per_class = max_rules_per_class
        self.min_rule_support = min_rule_support
        self.max_error_per_rule = max_error_per_rule

    def make_model(self, d: Dataset, class_weight: np.ndarray):
        return PRISM(
            class_weight,
            self.max_rule_length,
            self.max_rules_per_class,
            self.min_rule_support,
            self.max_error_per_rule,
        )
