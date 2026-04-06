from sklearn_nominal.bayes.model import NaiveBayes
import numpy as np
from sklearn.base import BaseEstimator

from sklearn_nominal.backend.core import Dataset
from sklearn_nominal.backend.factory import DEFAULT_BACKEND
from sklearn_nominal.bayes.trainer import NaiveBayesTrainer
from sklearn_nominal.sklearn.nominal_model import NominalClassifier


class NaiveBayesClassifier(NominalClassifier, BaseEstimator):
    """A Naive Bayes classifier supporting nominal attributes.

    A NaiveBayesClassifier that mimics `scikit-learn`'s
    :class:`sklearn.tree.GaussianNB` but adds support for nominal
    attributes with categorical distributions.

    Args:
        smoothing (float, optional): The Laplace smoothing factor for categorical
            distributions. This value will be added to the count of each value to
            generate a smoothed categorical distribution. The default value, 0.0,
            indicates no smoothing.
        backend (str, optional): The backend to use for computations. Defaults to "pandas".
        class_weight (dict or "balanced", optional): Weights associated with classes
            in the form ``{class_label: weight}``. If None, all classes are assumed
            to have weight one. The "balanced" mode uses the values of y to
            automatically adjust weights inversely proportional to class frequencies.
            Defaults to None.

    Attributes:
        classes_ (ndarray of shape (n_classes,)): The classes labels.
        n_classes_ (int): The number of classes.
        n_features_in_ (int): Number of features seen during :term:`fit`.
        feature_names_in_ (ndarray of shape (n_features_in_,)): Names of features
            seen during :term:`fit`. Defined only when `X` has feature names that
            are all strings.
        n_outputs_ (int): The number of outputs when ``fit`` is performed.
        model_ (NaiveBayes): The underlying NaiveBayes that actually holds the
            distribution parameters and can perform inference.

    See Also:
        TreeClassifier: A decision tree classifier.

    Notes:
        The :meth:`predict` method operates using the :func:`numpy.argmax`
        function on the outputs of :meth:`predict_proba`. This means that in
        case the highest predicted probabilities are tied, the classifier will
        predict the tied class with the lowest index in :term:`classes_`.

    Examples:
        >>> from sklearn.datasets import fetch_openml
        >>> df = fetch_openml("credit-g",version=2).frame
        >>> x,y = df.iloc[:,0:-1], df.iloc[:,-1]
        >>>
        >>> from sklearn_nominal import NaiveBayesClassifier
        >>> model = NaiveBayesClassifier(smoothing = 0.01)
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

    def __init__(self, smoothing=0.0, backend=DEFAULT_BACKEND, class_weight=None):
        """Initializes the NaiveBayesClassifier.

        Args:
            smoothing (float): The Laplace smoothing factor for categorical
                distributions. This value will be added to the count of each value to
                generate a smoothed categorical distribution. The default value, 0.0,
                indicates no smoothing.
            backend (str): The backend to use for computations. Defaults to "pandas".
            class_weight (dict or "balanced", optional): Weights associated with classes
                in the form ``{class_label: weight}``. If None, all classes are assumed
                to have weight one. The "balanced" mode uses the values of y to
                automatically adjust weights inversely proportional to class frequencies.
                Defaults to None.
        """
        super().__init__(backend=backend, class_weight=class_weight)
        self.smoothing = smoothing

    def make_model(self, d: Dataset, class_weight: np.ndarray):
        """Creates the NaiveBayesTrainer for the model.

        Args:
            d (Dataset): The dataset to train on.
            class_weight (np.ndarray): The weights for each class.

        Returns:
            NaiveBayesTrainer: The trainer instance for Naive Bayes.
        """
        return NaiveBayesTrainer(class_weight, smoothing=self.smoothing)

    def fit(self, x, y):
        """Fit the Naive Bayes model according to the given training data.

        This algorithm calculates the prior probabilities of each class and
        the conditional probabilities of each feature given the class. For
        nominal features, categorical distributions are estimated (with
        Laplace smoothing if requested). For numeric features, Gaussian
        distributions are typically assumed.

        Args:
            x (pd.DataFrame or np.ndarray): The training input samples.
            y (np.ndarray): The target values (class labels) as integers or strings.

        Returns:
            self: Returns the instance itself.
        """
        return super().fit(x, y)

    def predict(self, x):
        """Perform classification on an array of test vectors X.

        Predictions are made using Bayes' Theorem by multiplying the prior
        class probability with the conditional probabilities of all feature
        values given the class, and selecting the class with the highest
        posterior probability. Ties are resolved by choosing the class with the
        lowest index in :term:`classes_`.

        Args:
            x (pd.DataFrame or np.ndarray): The input samples.

        Returns:
            np.ndarray: Predicted target values for X.
        """
        return super().predict(x)

    def predict_proba(self, x):
        """Return probability estimates for the test data X.

        Probabilities are calculated by normalizing the posterior
        probabilities obtained via Bayes' Theorem for each class across all
        samples.

        Args:
            x (pd.DataFrame or np.ndarray): The input samples.

        Returns:
            np.ndarray: Returns the probability of the sample for each class
                in the model.
        """
        return super().predict_proba(x)

    def plot_distributions(
        self, class_names: list[str] | None = None, feature_names: list[str] | None = None, n_cols=4
    ):
        self.check_is_fitted()
        model: NaiveBayes = self.model_
        model.plot_distributions(class_names, feature_names, n_cols)

    def explain(self, x, class_names: list[str] | None = None):
        self.check_is_fitted()
        return self.model_.explain(x, class_names)
