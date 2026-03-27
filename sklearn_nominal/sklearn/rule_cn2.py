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

    [1] Clark, P. and Niblett, T (1989) The CN2 induction algorithm. Machine Learning 3(4):261-283.

    Args:
        criterion (str, optional): The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity and "log_loss"
            and "entropy" both for the Shannon information gain. Defaults to "entropy".
        max_rule_length (int, optional): The maximum number of conditions in a rule.
            Analogous to the maximum height of a Tree model. Defaults to sys.maxsize.
        max_rules (int, optional): Maximum number of rules for the model.
            Analogous to the maximum number of leaves in a Tree model. Defaults to sys.maxsize.
        min_rule_support (int, optional): Minimum number of samples that satisfy
            the condition of a rule required to include that rule in the model.
            Analogous to the `min_samples_leaf` parameter for Tree models.
            Defaults to 10.
        max_error_per_rule (float, optional): Maximum (absolute) error that the
            rule can have. This value depends on the error (criterion) used for the model.
            Defaults to 0.99.
        backend (str, optional): The backend to use for computations. Defaults to DEFAULT_BACKEND.
        class_weight (dict or "balanced", optional): Weights associated with classes
            in the form ``{class_label: weight}``. If None, all classes are assumed
            to have weight one. Defaults to None.

    Attributes:
        classes_ (ndarray of shape (n_classes,)): The classes labels.
        n_classes_ (int): The number of classes.
        n_features_in_ (int): Number of features seen during :term:`fit`.
        feature_names_in_ (ndarray of shape (n_features_in_,)): Names of features
            seen during :term:`fit`. Defined only when `X` has feature names that
            are all strings.
        n_outputs_ (int): The number of outputs when ``fit`` is performed.
        model_ (RuleModel): The underlying model object.

    See Also:
        TreeRegressor: A decision tree regressor with nominal support.
        NaiveBayesClassifier: A NaiveBayesClassifier with nominal support.
        ZeroRClassifier: A ZeroR classifier with nominal support.
        OneRClassifier: A OneR classifier with nominal support.
        PRISMClassifier: A PRISM classifier with nominal support.

    Examples:
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
        """Initializes the CN2Classifier.

        Args:
            criterion (str): The function to measure the quality of a split.
                Supported criteria are "gini" for the Gini impurity and "log_loss"
                and "entropy" both for the Shannon information gain.
                Defaults to "entropy".
            max_rule_length (int): The maximum number of conditions in a rule.
                Defaults to sys.maxsize.
            max_rules (int): Maximum number of rules for the model.
                Defaults to sys.maxsize.
            min_rule_support (int): Minimum number of samples that satisfy
                the condition of a rule required to include that rule in the model.
                Defaults to 10.
            max_error_per_rule (float): Maximum (absolute) error that the
                rule can have. Defaults to 0.99.
            backend (str): The backend to use for computations.
                Defaults to DEFAULT_BACKEND.
            class_weight (dict or "balanced", optional): Weights associated with classes
                in the form ``{class_label: weight}``. If None, all classes are assumed
                to have weight one. Defaults to None.
        """
        super().__init__(backend=backend, class_weight=class_weight)
        self.max_rule_length = max_rule_length
        self.max_rules = max_rules
        self.min_rule_support = min_rule_support
        self.max_error_per_rule = max_error_per_rule
        self.criterion = criterion

    def make_model(self, d: Dataset, class_weight: np.ndarray):
        """Creates the CN2 trainer for the model.

        Args:
            d (Dataset): The dataset to train on.
            class_weight (np.ndarray): The weights for each class.

        Returns:
            CN2: The CN2 trainer instance.
        """
        error = self.build_error(self.criterion, class_weight)
        return CN2(
            error,
            self.max_rule_length,
            self.max_rules,
            self.min_rule_support,
            self.max_error_per_rule,
        )

    def fit(self, x, y):
        """Fit the CN2 model according to the given training data.

        The CN2 algorithm induces a set of classification rules using a
        sequential covering (or "separate-and-conquer") process. It repeatedly
        identifies a rule that covers a subset of the training data and
        removes the covered samples until a sufficient number of rules is
        found or all samples are covered.

        Args:
            x (pd.DataFrame or np.ndarray): The training input samples.
            y (np.ndarray): The target values (class labels) as integers or strings.

        Returns:
            self: Returns the instance itself.
        """
        return super().fit(x, y)

    def predict(self, x):
        """Perform classification on an array of test vectors X.

        For each input sample, the algorithm evaluates the learned rules in
        order. The first rule that matches the sample's features determines
        the predicted class. If no rules match, the default class (based on
        the distribution of uncovered training samples) is used.

        Args:
            x (pd.DataFrame or np.ndarray): The input samples.

        Returns:
            np.ndarray: Predicted target values for X.
        """
        return super().predict(x)

    def predict_proba(self, x):
        """Return probability estimates for the test data X.

        Probabilities are estimated from the class distribution of training
        samples covered by the first rule that matches the input sample. If no
        rules match, the probability distribution is based on the uncovered
        training samples (the default rule).

        Args:
            x (pd.DataFrame or np.ndarray): The input samples.

        Returns:
            np.ndarray: Returns the probability of the sample for each class
                in the model.
        """
        return super().predict_proba(x)


class CN2Regressor(NominalRegressor, BaseEstimator):
    """A rule-based regressor that performs sequential covering in a CN2 [1] style.

    [1] Clark, P. and Niblett, T (1989) The CN2 induction algorithm. Machine Learning 3(4):261-283.

    Args:
        criterion (str, optional): The function to measure the error of a split.
            Supported criteria are currently only "std", for standard deviation
            (equivalent to root MSE). Defaults to "std".
        max_rule_length (int, optional): The maximum number of conditions in a rule.
            Analogous to the maximum height of a Tree model. Defaults to sys.maxsize.
        max_rules (int, optional): Maximum number of rules for the model.
            Analogous to the maximum number of leaves in a Tree model. Defaults to sys.maxsize.
        min_rule_support (int, optional): Minimum number of samples that satisfy
            the condition of a rule required to include that rule in the model.
            Analogous to the `min_samples_leaf` parameter for Tree models.
            Defaults to 10.
        max_error_per_rule (float, optional): Maximum (absolute) error that the
            rule can have. This value depends on the error (criterion) used for the model.
            Defaults to 0.99.
        backend (str, optional): The backend to use for computations. Defaults to DEFAULT_BACKEND.

    Attributes:
        n_features_in_ (int): Number of features seen during :term:`fit`.
        feature_names_in_ (ndarray of shape (n_features_in_,)): Names of features
            seen during :term:`fit`. Defined only when `X` has feature names that
            are all strings.
        n_outputs_ (int): The number of outputs when ``fit`` is performed.
        model_ (RuleModel): The underlying model object.

    See Also:
        TreeRegressor: A decision tree regressor with nominal support.
        ZeroRRegressor: A ZeroR classifier regressor with nominal support.
        OneRRegressor: A OneR regressor with nominal support.

    Examples:
        >>> from sklearn_nominal import CN2Regressor, read_golf_regression_dataset
        >>> x, y = read_golf_regression_dataset(url)
        >>> model = CN2Regressor()
        >>> from sklearn.metrics import mean_absolute_error
        >>> model.fit(x, y)
        >>> y_pred = model.predict(x)
        >>> print(f"{mean_absolute_error(y, y_pred):.2f}")
        0.07
    """

    def __init__(
        self,
        criterion="std",
        max_rule_length: int = sys.maxsize,
        max_rules: int = sys.maxsize,
        min_rule_support=10,
        max_error_per_rule=0.99,
        backend=DEFAULT_BACKEND,
    ):
        """Initializes the CN2Regressor.

        Args:
            criterion (str): The function to measure the error of a split.
                Supported criteria are currently only "std", for standard deviation
                (equivalent to root MSE). Defaults to "std".
            max_rule_length (int): The maximum number of conditions in a rule.
                Defaults to sys.maxsize.
            max_rules (int): Maximum number of rules for the model.
                Defaults to sys.maxsize.
            min_rule_support (int): Minimum number of samples that satisfy
                the condition of a rule required to include that rule in the model.
                Defaults to 10.
            max_error_per_rule (float): Maximum (absolute) error that the
                rule can have. Defaults to 0.99.
            backend (str): The backend to use for computations.
                Defaults to DEFAULT_BACKEND.
        """
        super().__init__(backend=backend)
        self.max_rule_length = max_rule_length
        self.max_rules = max_rules
        self.min_rule_support = min_rule_support
        self.max_error_per_rule = max_error_per_rule
        self.criterion = criterion

    def make_model(self, d: Dataset):
        """Creates the CN2 trainer for the model.

        Args:
            d (Dataset): The dataset to train on.

        Returns:
            CN2: The CN2 trainer instance.
        """
        error = self.build_error(self.criterion)
        return CN2(
            error,
            self.max_rule_length,
            self.max_rules,
            self.min_rule_support,
            self.max_error_per_rule,
        )

    def fit(self, x, y):
        """Fit the CN2 model according to the given training data.

        The CN2 algorithm induces a set of regression rules using a
        sequential covering process. It iteratively identifies rules that
        minimize regression error for a subset of the training data and
        removes the covered samples.

        Args:
            x (pd.DataFrame or np.ndarray): The training input samples.
            y (np.ndarray): The target values (real numbers).

        Returns:
            self: Returns the instance itself.
        """
        return super().fit(x, y)

    def predict(self, x):
        """Predict regression value for X.

        For each input sample, the algorithm evaluates the learned rules in
        order. The first rule that matches the sample determines the
        predicted target value (usually the mean of samples covered by that
        rule during training).

        Args:
            x (pd.DataFrame or np.ndarray): The input samples.

        Returns:
            np.ndarray: Predicted target values for X.
        """
        return super().predict(x)
