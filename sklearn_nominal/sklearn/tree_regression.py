import numpy as np
import pandas as pd
from scipy.odr import Output
from sklearn.base import BaseEstimator

from sklearn_nominal import shared, tree
from sklearn_nominal.backend import Input
from sklearn_nominal.backend.core import Dataset

from ..sklearn.nominal_model import NominalRegressor
from ..tree.pruning import PruneCriteria
from .tree_base import BaseTree


class TreeRegressor(NominalRegressor, BaseTree, BaseEstimator):
    """A decision tree regressor for nominal and numeric attributes.

    This estimator mimics scikit-learn's DecisionTreeRegressor but provides
    native support for nominal attributes without requiring pre-encoding.
    It builds a regression tree using a recursive partitioning approach.

    Args:
        criterion (str): The function to measure the quality of a split.
            Supported criteria is "std" for standard deviation. Defaults to "std".
        splitter (str): The strategy used to choose the split at each node.
            Supported strategies are "best" to choose the best split.
            Defaults to "best".
        max_depth (int, optional): The maximum depth of the tree. If None, then
            nodes are expanded until all leaves are pure or until all leaves
            contain less than min_samples_split samples. Defaults to None.
        min_samples_split (int): The minimum number of samples required to split
            an internal node. Defaults to 2.
        min_samples_leaf (int): The minimum number of samples required to be at
            a leaf node. A split point at any depth will only be considered if
            it leaves at least min_samples_leaf training samples in each of the
            left and right branches. Defaults to 1.
        min_error_decrease (float): A node will be split if this split induces
            a decrease of the error greater than or equal to this value.
            Defaults to 1e-16.
        nominal_split (str, optional): The strategy used to split nominal attributes.
            See :class:`BaseTree`. Defaults to "multi".
        backend (str): The backend used for data processing. Defaults to "pandas".

    Attributes:
        n_features_in_ (int): Number of features seen during fit.
        feature_names_in_ (ndarray): Names of features seen during fit.
            Defined only when X has feature names that are all strings.
        n_outputs_ (int): The number of outputs when fit is performed.
        tree_ (sklearn_nominal.tree.tree.Tree): The underlying Tree object.

    See Also:
        BaseTree: Base class for tree-based estimators.
        TreeClassifier: A decision tree classifier.

    Examples:
        >>> from sklearn_nominal import TreeRegressor, read_golf_regression_dataset
        >>> x, y = read_golf_regression_dataset(url)
        >>> model = TreeRegressor(criterion="std", max_depth=4)
        >>> model.fit(x, y)
        >>> y_pred = model.predict(x)
    """

    def __init__(
        self,
        criterion="std",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_error_decrease=1e-16,
        nominal_split="multi",
        backend="pandas",
    ):
        """Initializes the TreeRegressor.

        Args:
            criterion (str): The function to measure the quality of a split.
                Supported criteria is "std" for standard deviation. Defaults to "std".
            splitter (str): The strategy used to choose the split at each node.
                Supported strategies are "best" to choose the best split.
                Defaults to "best".
            max_depth (int, optional): The maximum depth of the tree. If None, then
                nodes are expanded until all leaves are pure or until all leaves
                contain less than min_samples_split samples. Defaults to None.
            min_samples_split (int): The minimum number of samples required to split
                an internal node. Defaults to 2.
            min_samples_leaf (int): The minimum number of samples required to be at
                a leaf node. A split point at any depth will only be considered if
                it leaves at least min_samples_leaf training samples in each of the
                left and right branches. Defaults to 1.
            min_error_decrease (float): A node will be split if this split induces
                a decrease of the error greater than or equal to this value.
                Defaults to 1e-16.
            nominal_split (str): The strategy used to split nominal attributes.
                See :class:`BaseTree`. Defaults to "multi".
            backend (str): The backend used for data processing. Defaults to "pandas".
        """
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_error_decrease=min_error_decrease,
            nominal_split=nominal_split,
            backend=backend,
        )

    def make_model(self, d: Dataset):
        """Creates the Tree trainer for the model.

        Args:
            d (Dataset): The dataset to train on.

        Returns:
            BaseTreeTrainer: The tree trainer instance.
        """
        error = self.build_error(self.criterion)
        column_penalization = self.build_attribute_penalizer()
        scorers = self.build_splitter(error, column_penalization)
        scorer = shared.DefaultSplitter(error, scorers)
        prune_criteria = self.build_prune_criteria(d)
        trainer = tree.BaseTreeTrainer(scorer, prune_criteria)
        return trainer

    def fit(self, x, y):
        """Fit the decision tree regressor according to the given training data.

        This algorithm builds a regression tree using recursive
        partitioning. At each node, it selects the feature and the split
        (numeric or nominal) that minimizes the regression error (e.g.,
        standard deviation). For nominal attributes, it creates a multi-
        way split corresponding to the attribute's categories.

        Args:
            x (pd.DataFrame or np.ndarray): The training input samples.
            y (np.ndarray): The target values (real numbers).

        Returns:
            self: Returns the instance itself.
        """
        return super().fit(x, y)

    def predict(self, x):
        """Predict regression value for X.

        The predicted regression value for each input sample is obtained by
        traversing the decision tree from the root to a leaf node according
        to the sample's feature values. The prediction is the mean of target
        values in that leaf node.

        Args:
            x (pd.DataFrame or np.ndarray): The input samples.

        Returns:
            np.ndarray: Predicted target values for X.
        """
        return super().predict(x)
