import numpy as np
from sklearnmodels.backend.core import ColumnType
from sklearnmodels.tree.pruning import PruneCriteria


from .. import tree, shared


class BaseTree:

    def __init__(
        self,
        criterion="",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
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

    def make_prune_criteria(self):
        return PruneCriteria(
            max_height=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_error_decrease=self.min_error_decrease,
            min_samples_split=self.min_samples_split,
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
