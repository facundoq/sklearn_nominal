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
    """A ZeroR classifier, equivalent to a TreeClassifier with a depth of 0 (only root).

    [1] Holte, Robert C. "Very simple classification rules perform well on most commonly used datasets." Machine learning 11.1 (1993): 63-90.

    Args:
        criterion (str, optional): The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity and "log_loss"
            and "entropy" both for the Shannon information gain. Defaults to "entropy".
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
        NaiveBayesClassifier: A NaiveBayesClassifier with nominal support.
        CN2Classifier: A CN2Classifier classifier with nominal support.
        PRISMClassifier: A PRISM classifier with nominal support.
        OneRClassifier: A OneR classifier with nominal support.

    Examples:
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
        """Returns the scikit-learn tags for the estimator.

        Returns:
            Tags: The scikit-learn tags.
        """
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags

    def __init__(self, criterion="entropy", backend=DEFAULT_BACKEND, class_weight=None):
        """Initializes the ZeroRClassifier.

        Args:
            criterion (str): The function to measure the quality of a split.
                Supported criteria are "gini" for the Gini impurity and "log_loss"
                and "entropy" both for the Shannon information gain.
                Defaults to "entropy".
            backend (str): The backend to use for computations.
                Defaults to DEFAULT_BACKEND.
            class_weight (dict or "balanced", optional): Weights associated with classes
                in the form ``{class_label: weight}``. If None, all classes are assumed
                to have weight one. Defaults to None.
        """
        super().__init__(backend=backend, class_weight=class_weight)
        self.criterion = criterion

    def make_model(self, d: Dataset, class_weight: np.ndarray):
        """Creates the ZeroR trainer for the model.

        Args:
            d (Dataset): The dataset to train on.
            class_weight (np.ndarray): The weights for each class.

        Returns:
            ZeroR: The ZeroR trainer instance.
        """
        error = self.build_error(self.criterion, class_weight)
        return ZeroR(error)

    def fit(self, x, y):
        """Fit the ZeroR model according to the given training data.

        The ZeroR algorithm identifies the majority class (the mode) of the
        target values in the training data and uses it for all future
        predictions, ignoring all input features.

        Args:
            x (pd.DataFrame or np.ndarray): The training input samples.
                These are ignored by the ZeroR algorithm.
            y (np.ndarray): The target values (class labels) as integers or strings.

        Returns:
            self: Returns the instance itself.
        """
        return super().fit(x, y)

    def predict(self, x):
        """Perform classification on an array of test vectors X.

        Always predicts the majority class identified during :meth:`fit` for
        all input samples.

        Args:
            x (pd.DataFrame or np.ndarray): The input samples.

        Returns:
            np.ndarray: Predicted target values for X, all equal to the
                majority class.
        """
        return super().predict(x)

    def predict_proba(self, x):
        """Return probability estimates for the test data X.

        The probability estimates are constant for all samples and correspond
        to the class distributions (frequencies) observed in the training data.

        Args:
            x (pd.DataFrame or np.ndarray): The input samples.

        Returns:
            np.ndarray: Returns the probability of the sample for each class
                in the model, based on training set frequencies.
        """
        return super().predict_proba(x)


class ZeroRRegressor(NominalRegressor, BaseEstimator):
    """A ZeroR Regressor, equivalent to a TreeClassifier with a depth of 0 (only root).

    [1] Holte, Robert C. "Very simple classification rules perform well on most commonly used datasets." Machine learning 11.1 (1993): 63-90.

    Args:
        criterion (str, optional): The function to measure the error of a split.
            Supported criteria are currently only "std", for standard deviation
            (equivalent to root MSE). Defaults to "std".
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
        CN2Regressor: A CN2Classifier regressor with nominal support.
        OneRRegressor: A OneR regressor with nominal support.

    Examples:
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
        """Initializes the ZeroRRegressor.

        Args:
            criterion (str): The function to measure the error of a split.
                Supported criteria are currently only "std", for standard deviation
                (equivalent to root MSE). Defaults to "std".
            backend (str): The backend to use for computations.
                Defaults to DEFAULT_BACKEND.
        """
        super().__init__(backend=backend)
        self.criterion = criterion

    def __sklearn_tags__(self):
        """Returns the scikit-learn tags for the estimator.

        Returns:
            Tags: The scikit-learn tags.
        """
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        return tags

    def make_model(self, d: Dataset):
        """Creates the ZeroR trainer for the model.

        Args:
            d (Dataset): The dataset to train on.

        Returns:
            ZeroR: The ZeroR trainer instance.
        """
        error = self.build_error(self.criterion)
        return ZeroR(error_function=error)

    def fit(self, x, y):
        """Fit the ZeroR model according to the given training data.

        The ZeroR algorithm identifies the mean target value in the training
        data and uses it for all future predictions, ignoring all input
        features.

        Args:
            x (pd.DataFrame or np.ndarray): The training input samples.
                These are ignored by the ZeroR algorithm.
            y (np.ndarray): The target values (real numbers).

        Returns:
            self: Returns the instance itself.
        """
        return super().fit(x, y)

    def predict(self, x):
        """Predict regression value for X.

        Always predicts the mean target value identified during :meth:`fit` for
        all input samples.

        Args:
            x (pd.DataFrame or np.ndarray): The input samples.

        Returns:
            np.ndarray: Predicted target values for X, all equal to the
                training mean.
        """
        return super().predict(x)
