import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils import compute_class_weight

from sklearn_nominal.backend import Input, Output
from sklearn_nominal.backend.core import Dataset
from sklearn_nominal.backend.factory import DEFAULT_BACKEND
from sklearn_nominal.rules.zeror import ZeroR as ZeroR
from sklearn_nominal.sklearn.nominal_model import NominalClassifier, NominalRegressor


class ZeroRClassifier(NominalClassifier, BaseEstimator):
    """A Zero classifier, equivalent to a TreeClassifier with a depth of 0 (only root).

    [1] Holte, Robert C. "Very simple classification rules perform well on most commonly used datasets." Machine learning 11.1 (1993): 63-90.

    Parameters
    ----------
    criterion : {"gini", "entropy", "log_loss"}, default="entropy"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "log_loss" and "entropy" both for the
        Shannon information gain.

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
    >>> from sklearn_nominal import ZeroRClassifier
    >>> model = ZeroRClassifier()
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

    def __init__(self, criterion="entropy", backend=DEFAULT_BACKEND, class_weight=None):
        super().__init__(backend=backend, class_weight=class_weight)
        self.criterion = criterion

    def make_model(self, d: Dataset, class_weight: np.ndarray):
        error = self.build_error(self.criterion, class_weight)
        return ZeroR(error)


class ZeroRRegressor(NominalRegressor, BaseEstimator):
    """A ZeroR Regressor, equivalent to a TreeClassifier with a depth of 0 (only root).

    [1] Holte, Robert C. "Very simple classification rules perform well on most commonly used datasets." Machine learning 11.1 (1993): 63-90.

    Parameters
    ----------
    criterion : {"gini", "entropy", "log_loss"}, default="entropy"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "log_loss" and "entropy" both for the
        Shannon information gain.

    Attributes
    ----------

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
    CN2Regressor: a CN2Classifier regressor with nominal support.
    OneRRegressor: a OneR regressor with nominal support.

    Examples
    --------
    >>> from sklearn_nominal import ZeroRRegressor, read_golf_regression_dataset
    >>> x, y = read_golf_regression_dataset(url)
    >>> model = ZeroRRegressor()
    >>> from sklearn.metrics import mean_absolute_error
    >>> model.fit(x, y)
    >>> y_pred = model.predict(x)
    >>> print(f"{mean_absolute_error(y, y_pred):.2f}")
    0.07
    """

    def __init__(self, criterion="std", backend=DEFAULT_BACKEND):
        super().__init__(backend=backend)
        self.criterion = criterion

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        return tags

    def make_model(self, d: Dataset):
        error = self.build_error(self.criterion)
        return ZeroR(error_function=error)
