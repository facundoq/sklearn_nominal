from math import floor

import numpy as np

from sklearn_nominal.backend.core import ColumnType, Dataset
from sklearn_nominal.tree.pruning import PruneCriteria

from .. import shared, tree


class BaseTree:
    """A base class for decision trees

    Parameters
    ----------
    criterion : {"gini", "entropy", "gain_ratio"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "gain_ratio" and "entropy" both for the
        Shannon information gain with and without attribute penalization.

    splitter : {"best", int}, default="best"
        The strategy used to choose the split at each numeric node. Supported
        strategies are "best" to choose the best split, or a number to limit
        the maximum number of splits to consider.

      max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
    """

    def __init__(
        self,
        criterion="",
        splitter="best",
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf=1,
        min_error_decrease=0.0,
    ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_error_decrease = min_error_decrease

    def build_attribute_penalizer(self):
        if self.criterion == "gain_ratio":
            return shared.GainRatioPenalization()
        else:
            return shared.NoPenalization()

    def build_splitter(self, e: shared.TargetError, p: shared.ColumnPenalization):
        if self.splitter == "best":
            max_evals = np.iinfo(np.int64).max
        elif isinstance(self.splitter, int):
            max_evals = self.splitter
        else:
            raise ValueError(
                f"Invalid value '{self.splitter}' for splitter; expected integer or"
                " 'best'"
            )
        scorers = {
            ColumnType.Numeric: shared.NumericColumnError(e, p, max_evals=max_evals),
            ColumnType.Nominal: shared.NominalColumnError(e, p),
        }
        return scorers

    def make_prune_criteria(self, d: Dataset):
        min_samples_leaf = self.min_samples_leaf
        if isinstance(min_samples_leaf, float):
            min_samples_leaf = int(floor(d.n * min_samples_leaf))

        min_samples_split = self.min_samples_split
        if isinstance(min_samples_leaf, float):
            min_samples_split = int(floor(d.n * min_samples_split))

        return PruneCriteria(
            max_height=self.max_depth,
            min_samples_leaf=min_samples_leaf,
            min_error_decrease=self.min_error_decrease,
            min_samples_split=min_samples_split,
        )

    def pretty_print(self, class_names=None):
        return self.model_.pretty_print(class_names=class_names)

    def export_dot(self, class_names=None, title=""):
        return tree.export_dot(self.model_, title=title, class_names=class_names)

    def export_dot_file(self, filepath, class_names=None, title=""):
        tree.export_dot_file(
            self.model_, filepath, title=title, class_names=class_names
        )

    def export_image(self, filepath, class_names=None, title=""):
        tree.export_image(self.model_, filepath, title=title, class_names=class_names)

    def display(self, class_names=None, title=""):
        return tree.display(self.model_, title=title, class_names=class_names)
