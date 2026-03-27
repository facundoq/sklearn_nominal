from math import floor

import numpy as np

from sklearn_nominal.backend.core import ColumnType, Dataset
from sklearn_nominal.tree.pruning import PruneCriteria

from .. import shared, tree


class BaseTree:
    """Base class for decision trees with native nominal attribute support.

    This class serves as a shared foundation for both classification and
    regression trees, managing the hyperparameters that control tree growth,
    node splitting, and pruning strategies.

    Architectural Context
    ---------------------
    `BaseTree` coordinates the high-level tree-building process. It acts as a
    configuration hub that translates scikit-learn compatible parameters into
    the specialized components required by the internal backend trainers.
    Specifically, it handles:
    1. **Splitter Construction**: Mapping numerical and nominal features to
       appropriate `ColumnError` scorers.
    2. **Pruning Configuration**: Translating user-defined constraints (like
       `max_depth` or `min_samples_leaf`) into a `PruneCriteria` object.
    3. **Attribute Penalization**: Implementing strategies like Gain Ratio to
       compensate for the bias of many-valued nominal attributes.

    Examples
    --------
    >>> from sklearn_nominal.sklearn.tree_base import BaseTree
    >>> class MyTree(BaseTree):
    ...     def __init__(self, **kwargs):
    ...         super().__init__(**kwargs)
    ...     def fit(self, X, y):
    ...         # implementation that uses self.build_splitter()
    ...         # and self.build_prune_criteria()
    ...         pass

    Parameters
    ----------
    criterion : str, default=""
        The function to measure the quality of a split. Supported criteria
        are "gini" for Gini impurity, and "entropy" or "gain_ratio" for
        Shannon information gain.
    splitter : str or int, default="best"
        The strategy used to choose the split at each numeric node.
        - "best": evaluates all possible split points.
        - int: limits the maximum number of splits to consider per node.
    max_depth : int, optional
        The maximum depth of the tree. If None, nodes are expanded until all
        leaves are pure or contain fewer than `min_samples_split` samples.
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node.
        - If int, treated as a constant count.
        - If float, treated as a fraction: `ceil(min_samples_split * n_samples)`.
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        - If int, treated as a constant count.
        - If float, treated as a fraction: `ceil(min_samples_leaf * n_samples)`.
    min_error_decrease : float, default=0.0
        A node will be split if the split induces a decrease of the error
        greater than or equal to this value.

    Notes
    -----
    The weighted error decrease is calculated as:
    ``Δerror = error - Σ_i (N_i / N) * error_i``
    where ``N`` is the total samples, ``N_i`` is the samples in branch ``i``,
    and ``error_i`` is the error in that branch.

    References
    ----------
    .. [1] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.
    .. [2] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.
    """

    def __init__(
        self,
        criterion="",
        splitter="best",
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf=1,
        min_error_decrease=0.0,
        nominal_split="multi",
    ):
        """Initializes the BaseTree with growth and splitting parameters.

        Parameters
        ----------
        criterion : str
            Splitting quality measure.
        splitter : str or int
            Numeric node splitting strategy.
        max_depth : int, optional
            Maximum tree depth.
        min_samples_split : int or float
            Minimum samples per internal node.
        min_samples_leaf : int or float
            Minimum samples per leaf node.
        min_error_decrease : float
            Minimum required error reduction for splitting.
        nominal_split : str, default="multi"
            The strategy used to split nominal attributes.
            - "multi": creates one branch for each unique value.
            - "binary": creates a binary (One-vs-Rest) split for each node.
        """
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_error_decrease = min_error_decrease
        self.nominal_split = nominal_split

    def build_attribute_penalizer(self):
        """Determines the penalization strategy for multi-valued attributes.

        This is used to implement "gain_ratio", which penalizes nominal
        attributes with many levels to prevent overfitting.

        Returns
        -------
        sklearn_nominal.shared.ColumnPenalization
            The penalization strategy (e.g., `GainRatioPenalization` or
            `NoPenalization`).
        """
        if self.criterion == "gain_ratio":
            return shared.GainRatioPenalization()
        else:
            return shared.NoPenalization()

    def build_splitter(self, e: shared.TargetError, p: shared.ColumnPenalization):
        """Constructs specialized split scorers for different column types.

        This method maps the general `splitter` and `criterion` parameters into
        concrete `ColumnError` implementations for both Numeric and Nominal
        features.

        Detailed Method Explanation
        ---------------------------
        For Numeric columns, it creates a `NumericColumnError` which can be
        limited by `max_evals` if the `splitter` parameter is an integer.
        For Nominal columns, it creates a `NominalColumnError` (multi-way split)
        or `BinaryNominalColumnError` (binary split) depending on the
        `nominal_split` parameter.

        Parameters
        ----------
        e : sklearn_nominal.shared.TargetError
            The target error function (e.g., Gini, Entropy, or MSE).
        p : sklearn_nominal.shared.ColumnPenalization
            The column-level penalization strategy.

        Returns
        -------
        dict
            A dictionary mapping `ColumnType` to its corresponding
            `ColumnError` scorer.

        Raises
        ------
        ValueError
            If the `splitter` value is neither "best" nor an integer, or if
            `nominal_split` is invalid.
        """
        if self.splitter == "best":
            max_evals = np.iinfo(np.int64).max
        elif isinstance(self.splitter, int):
            max_evals = self.splitter
        else:
            raise ValueError(f"Invalid value '{self.splitter}' for splitter; expected integer or 'best'")

        if self.nominal_split == "multi":
            nominal_scorer = shared.NominalColumnError(e, p)
        elif self.nominal_split == "binary":
            nominal_scorer = shared.BinaryNominalColumnError(e, p)
        else:
            raise ValueError(f"Invalid value '{self.nominal_split}' for nominal_split; expected 'multi' or 'binary'")

        scorers = {
            ColumnType.Numeric: shared.NumericColumnError(e, p, max_evals=max_evals),
            ColumnType.Nominal: nominal_scorer,
        }
        return scorers

    def build_prune_criteria(self, d: Dataset) -> PruneCriteria:
        """Translates tree constraints into internal pruning criteria.

        This method converts user-facing parameters (which can be counts or
        fractions) into the absolute integer values required by the tree
        builders.

        Detailed Method Explanation
        ---------------------------
        - **Fractional conversion**: If `min_samples_leaf` or
          `min_samples_split` are floats, they are multiplied by the total
          number of samples in dataset `d` and floored.
        - **Constraint packaging**: All constraints are encapsulated into a
          `PruneCriteria` object which is passed to the recursive tree
          trainer.

        Parameters
        ----------
        d : Dataset
            The dataset used to calculate relative sample counts.

        Returns
        -------
        PruneCriteria
            The consolidated criteria used for pruning.
        """
        min_samples_leaf = self.min_samples_leaf
        if isinstance(min_samples_leaf, float):
            min_samples_leaf = int(floor(d.n * min_samples_leaf))

        min_samples_split = self.min_samples_split
        if isinstance(min_samples_split, float):
            min_samples_split = int(floor(d.n * min_samples_split))

        return PruneCriteria(
            max_height=self.max_depth,
            min_samples_leaf=min_samples_leaf,
            min_error_decrease=self.min_error_decrease,
            min_samples_split=min_samples_split,
        )

    def pretty_print(self, class_names=None):
        """Returns a human-readable string representation of the tree.

        Parameters
        ----------
        class_names : list of str, optional
            The names of the classes for display.

        Returns
        -------
        str
            The pretty-printed tree structure.
        """
        return self.model_.pretty_print(class_names=class_names)

    def export_dot(self, class_names=None, title=""):
        """Exports the tree as a Graphviz dot string.

        Parameters
        ----------
        class_names : list of str, optional
            The names of the classes for display.
        title : str, default=""
            The title for the graph.

        Returns
        -------
        str
            The tree in Graphviz dot format.
        """
        return tree.export_dot(self.model_, title=title, class_names=class_names)

    def export_dot_file(self, filepath, class_names=None, title=""):
        """Exports the tree as a Graphviz dot file.

        Parameters
        ----------
        filepath : str
            The path to the file to save.
        class_names : list of str, optional
            The names of the classes for display.
        title : str, default=""
            The title for the graph.
        """
        tree.export_dot_file(self.model_, filepath, title=title, class_names=class_names)

    def export_image(self, filepath, class_names=None, title=""):
        """Exports the tree as an image file.

        This requires Graphviz to be installed on the system.

        Parameters
        ----------
        filepath : str
            The path to the image file (e.g., "tree.png").
        class_names : list of str, optional
            The names of the classes for display.
        title : str, default=""
            The title for the graph.
        """
        tree.export_image(self.model_, filepath, title=title, class_names=class_names)

    def display(self, class_names=None, title=""):
        """Displays the tree using the default system viewer or notebook output.

        Parameters
        ----------
        class_names : list of str, optional
            The names of the classes for display.
        title : str, default=""
            The title for the graph.

        Returns
        -------
        Any
            The image object for display in interactive environments.
        """
        return tree.display(self.model_, title=title, class_names=class_names)
