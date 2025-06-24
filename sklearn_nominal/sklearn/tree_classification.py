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
    """A decision tree classifier that mimics `scikit-learn`'s
    :class:`sklearn.tree.DecisionTreeClassifier` but adds support for nominal
    attributes.


    See also :class:`sklearn_nominal.sklearn.tree_base.BaseTree` for additional
    parameters

    Parameters
    ----------
    criterion : {"gini", "entropy", "log_loss"}, default="entropy"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "log_loss" and "entropy" both for the
        Shannon information gain.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the error
        greater than or equal to this value. For compatibility with
        `scikit-learn` this is called `min_impurity_decrease`, but a more proper
        name would be `min_error_decrease`.

        The weighted error decrease equation is the following::

            Δerror = error - Σ_i (N_i/N) * error_i


        where ``N`` is the total number of samples, ``N_i`` is the number of
        samples in the i-th branch, and `error_i` is the error at the i-th
        branch.

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

    tree_ : :class:`sklearn_nominal.tree.tree.Tree` instance
        The underlying :class:`sklearn_nominal.tree.tree.Tree` object.

    See Also
    --------
    TreeRegressor : A decision tree regressor.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The :meth:`predict` method operates using the :func:`numpy.argmax`
    function on the outputs of :meth:`predict_proba`. This means that in
    case the highest predicted probabilities are tied, the classifier will
    predict the tied class with the lowest index in :term:`classes_`.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
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
    >>> print(model.pretty_print(class_names=y.unique()))
    ... 🫚 Root
    ... |   🪵checking_status=>=200 =>
    ... |   |   🪵property_magnitude=real estate =>
    ... |   |   |   🪵installment_commitment <= 3 => ☘︎ bad
    ... ...........
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
        backend="pandas",
    ):
        super().__init__(
            class_weight=class_weight,
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_error_decrease=min_error_decrease,
            backend=backend,
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags

    def make_model(self, d: Dataset, class_weight: np.ndarray):
        error = self.build_error(self.criterion, class_weight)
        column_penalization = self.build_attribute_penalizer()

        scorers = self.build_splitter(error, column_penalization)

        scorer = shared.DefaultSplitter(error, scorers)
        prune_criteria = self.make_prune_criteria(d)
        trainer = tree.BaseTreeTrainer(scorer, prune_criteria)
        return trainer
