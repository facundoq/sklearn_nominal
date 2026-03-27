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

    Args:
        max_rule_length (int, optional): The maximum number of conditions in a rule.
            Analogous to the maximum height of a Tree model. Defaults to sys.maxsize.
        max_rules_per_class (int, optional): Maximum number of rules for the model
            per class. Analogous to the maximum number of leaves in a Tree model.
            Defaults to sys.maxsize.
        min_rule_support (int, optional): Minimum number of samples that satisfy
            the condition of a rule required to include that rule in the model.
            Analogous to the `min_samples_leaf` parameter for Tree models.
            Defaults to 10.
        max_error_per_rule (float, optional): Maximum (absolute) error that the
            rule can have. Defaults to 1.
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
        CN2Classifier: A CN2Classifier classifier with nominal support.
        ZeroRClassifier: A ZeroR classifier with nominal support.
        OneRClassifier: A OneR classifier with nominal support.

    Examples:
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
        """Returns the scikit-learn tags for the estimator.

        Returns:
            Tags: The scikit-learn tags.
        """
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
        """Initializes the PRISMClassifier.

        Args:
            max_rule_length (int): The maximum number of conditions in a rule.
                Defaults to sys.maxsize.
            max_rules_per_class (int): Maximum number of rules for the model
                per class. Defaults to sys.maxsize.
            min_rule_support (int): Minimum number of samples that satisfy
                the condition of a rule required to include that rule in the model.
                Defaults to 10.
            max_error_per_rule (float): Maximum (absolute) error that the
                rule can have. Defaults to 1.
            backend (str): The backend to use for computations.
                Defaults to DEFAULT_BACKEND.
            class_weight (dict or "balanced", optional): Weights associated with classes
                in the form ``{class_label: weight}``. If None, all classes are assumed
                to have weight one. Defaults to None.
        """
        super().__init__(backend=backend, class_weight=class_weight)
        self.max_rule_length = max_rule_length
        self.max_rules_per_class = max_rules_per_class
        self.min_rule_support = min_rule_support
        self.max_error_per_rule = max_error_per_rule

    def make_model(self, d: Dataset, class_weight: np.ndarray):
        """Creates the PRISM trainer for the model.

        Args:
            d (Dataset): The dataset to train on.
            class_weight (np.ndarray): The weights for each class.

        Returns:
            PRISM: The PRISM trainer instance.
        """
        return PRISM(
            class_weight,
            self.max_rule_length,
            self.max_rules_per_class,
            self.min_rule_support,
            self.max_error_per_rule,
        )

    def fit(self, x, y):
        """Fit the PRISM model according to the given training data.

        The PRISM algorithm induces modular classification rules for each
        class independently. For each target class, it finds rules that cover
        as many samples of that class as possible while minimizing the
        inclusion of other classes.

        Args:
            x (pd.DataFrame or np.ndarray): The training input samples.
            y (np.ndarray): The target values (class labels) as integers or strings.

        Returns:
            self: Returns the instance itself.
        """
        return super().fit(x, y)

    def predict(self, x):
        """Perform classification on an array of test vectors X.

        For each input sample, the algorithm evaluates the modular rules. If
        multiple rules match, ties are resolved (often by choosing the class
        with the highest rule confidence). If no rules match, the default
        majority class from the entire training set is predicted.

        Args:
            x (pd.DataFrame or np.ndarray): The input samples.

        Returns:
            np.ndarray: Predicted target values for X.
        """
        return super().predict(x)

    def predict_proba(self, x):
        """Return probability estimates for the test data X.

        Probabilities are estimated based on the class distribution of all
        matching modular rules for the input sample. If no rules match,
        the global class distribution of the training data is used.

        Args:
            x (pd.DataFrame or np.ndarray): The input samples.

        Returns:
            np.ndarray: Returns the probability of the sample for each class
                in the model.
        """
        return super().predict_proba(x)
