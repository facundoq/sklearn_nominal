import numpy as np
import pandas as pd
from scipy.odr import Output
from sklearn.base import BaseEstimator
from sklearn.utils import compute_class_weight

from sklearn_nominal import shared, tree
from sklearn_nominal.backend import Input
from sklearn_nominal.backend.core import Dataset

from ..sklearn.nominal_model import NominalClassifier
from .tree_base import BaseTree


class TreeClassifier(NominalClassifier, BaseTree, BaseEstimator):
    """A decision tree classifier with support for nominal attributes.

    A decision tree classifier that mimics `scikit-learn`'s
    :class:`sklearn.tree.DecisionTreeClassifier` but adds support for nominal
    attributes.

    Args:
        criterion (str, optional): The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity and "log_loss"
            and "entropy" both for the Shannon information gain. Defaults to "entropy".
        class_weight (dict or "balanced", optional): Weights associated with classes
            in the form ``{class_label: weight}``. If None, all classes are assumed
            to have weight one. Defaults to None.
        splitter (str or int, optional): The strategy used to choose the split at
            each numeric node. See :class:`BaseTree`. Defaults to "best".
        max_depth (int, optional): The maximum depth of the tree. See :class:`BaseTree`.
            Defaults to None.
        min_samples_split (int or float, optional): The minimum number of samples
            required to split an internal node. See :class:`BaseTree`. Defaults to 2.
        min_samples_leaf (int or float, optional): The minimum number of samples
            required to be at a leaf node. See :class:`BaseTree`. Defaults to 1.
        min_error_decrease (float, optional): Threshold for early stopping in tree
            growth. See :class:`BaseTree`. Defaults to 1e-16.
        nominal_split (str, optional): The strategy used to split nominal attributes.
            See :class:`BaseTree`. Defaults to "multi".
        backend (str, optional): The backend to use for computations. Defaults to "pandas".

    Attributes:
        classes_ (ndarray of shape (n_classes,)): The classes labels.
        n_classes_ (int): The number of classes.
        n_features_in_ (int): Number of features seen during :term:`fit`.
        feature_names_in_ (ndarray of shape (n_features_in_,)): Names of features
            seen during :term:`fit`. Defined only when `X` has feature names that
            are all strings.
        n_outputs_ (int): The number of outputs when ``fit`` is performed.
        tree_ (Tree): The underlying :class:`sklearn_nominal.tree.tree.Tree` object.

    See Also:
        BaseTree: Base class for TreeClassifier and TreeRegressor.
        TreeRegressor: A decision tree regressor with nominal support.
        NaiveBayesClassifier: A NaiveBayesClassifier with nominal support.

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
        >>> from sklearn_nominal import TreeClassifier
        >>> model = TreeClassifier(min_samples_leaf=0.01)
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
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_error_decrease=1e-16,
        class_weight=None,
        nominal_split="multi",
        backend="pandas",
    ):
        """Initializes the TreeClassifier.

        Args:
            criterion (str): The function to measure the quality of a split.
                Supported criteria are "gini" for the Gini impurity and "log_loss"
                and "entropy" both for the Shannon information gain. Defaults to "entropy".
            splitter (str or int): The strategy used to choose the split at
                each numeric node. See :class:`BaseTree`. Defaults to "best".
            max_depth (int, optional): The maximum depth of the tree. See :class:`BaseTree`.
                Defaults to None.
            min_samples_split (int or float): The minimum number of samples
                required to split an internal node. See :class:`BaseTree`. Defaults to 2.
            min_samples_leaf (int or float): The minimum number of samples
                required to be at a leaf node. See :class:`BaseTree`. Defaults to 1.
            min_error_decrease (float): Threshold for early stopping in tree
                growth. See :class:`BaseTree`. Defaults to 1e-16.
            class_weight (dict or "balanced", optional): Weights associated with classes
                in the form ``{class_label: weight}``. If None, all classes are assumed
                to have weight one. Defaults to None.
            nominal_split (str): The strategy used to split nominal attributes.
                See :class:`BaseTree`. Defaults to "multi".
            backend (str): The backend to use for computations. Defaults to "pandas".
        """
        super().__init__(
            class_weight=class_weight,
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_error_decrease=min_error_decrease,
            nominal_split=nominal_split,
            backend=backend,
        )

    def __sklearn_tags__(self):
        """Returns the scikit-learn tags for the estimator.

        Returns:
            Tags: The scikit-learn tags.
        """
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags

    def make_model(self, d: Dataset, class_weight: np.ndarray):
        """Creates the Tree trainer for the model.

        Args:
            d (Dataset): The dataset to train on.
            class_weight (np.ndarray): The weights for each class.

        Returns:
            BaseTreeTrainer: The tree trainer instance.
        """
        error = self.build_error(self.criterion, class_weight)
        column_penalization = self.build_attribute_penalizer()

        scorers = self.build_splitter(error, column_penalization)

        scorer = shared.DefaultSplitter(error, scorers)
        prune_criteria = self.build_prune_criteria(d)
        trainer = tree.BaseTreeTrainer(scorer, prune_criteria)
        return trainer

    def fit(self, x, y):
        """Fit the decision tree classifier according to the given training data.

        This algorithm builds a classification tree using recursive
        partitioning. At each node, it selects the feature and the split
        (numeric or nominal) that maximizes the chosen criterion (e.g.,
        Shannon information gain). For nominal attributes, it creates a multi-
        way split corresponding to the attribute's categories.

        Args:
            x (pd.DataFrame or np.ndarray): The training input samples.
            y (np.ndarray): The target values (class labels) as integers or strings.

        Returns:
            self: Returns the instance itself.
        """
        return super().fit(x, y)

    def predict(self, x):
        """Perform classification on an array of test vectors X.

        Predictions are made by traversing the decision tree from the root to
        a leaf node according to the feature values of each input sample.
        Ties in leaf node probabilities are resolved by choosing the class
        with the lowest index in :term:`classes_`.

        Args:
            x (pd.DataFrame or np.ndarray): The input samples.

        Returns:
            np.ndarray: Predicted target values for X.
        """
        return super().predict(x)

    def predict_proba(self, x):
        """Return probability estimates for the test data X.

        Probabilities are estimated as the class distribution observed at the
        leaf node reached by traversing the tree for each input sample.

        Args:
            x (pd.DataFrame or np.ndarray): The input samples.

        Returns:
            np.ndarray: Returns the probability of the sample for each class
                in the model.
        """
        return super().predict_proba(x)
