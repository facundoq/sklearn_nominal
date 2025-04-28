import abc
from typing import Callable
import numpy as np
import pandas as pd

from .column_error import SplitterResult
from .tree import Condition, Tree
from .global_error import ColumnErrors, GlobalErrorResult, Splitter


class TreeTrainer(abc.ABC):

    @abc.abstractmethod
    def fit(self, x: pd.DataFrame, y: np.ndarray) -> Tree:
        pass


type TreeCreationCallback = Callable[
    [Tree, int, bool, ColumnErrors, SplitterResult], None
]
type TreeSplitCallback = Callable[[Tree, SplitterResult, Condition], None]


class PruneCriteria:
    def __init__(
        self,
        min_error_decrease: float = 0.00001,
        min_samples_leaf: int = 1,
        min_samples_split=1,
        max_height: int = None,
        error_tolerance: float = 1e-16,
    ):
        if max_height is not None:
            assert max_height > 0
        assert min_samples_leaf > 0
        assert min_error_decrease >= 0

        self.min_error_decrease = min_error_decrease
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_height = max_height
        self.error_tolerance = error_tolerance

    def params(self):
        return {
            "min_error_decrease": self.min_error_decrease,
            "min_samples_leaf": self.min_samples_leaf,
            "min_samples_split": self.min_samples_split,
            "max_height": self.max_height,
            "error_tolerance": self.error_tolerance,
        }

    def __repr__(self):
        params_str = ", ".join([f"{k}={v}" for k, v in self.params().items()])
        return f"Prune({params_str})"

    def pre_split_prune(self, x: pd.DataFrame, y: np.ndarray, height: int, tree: Tree):
        # BASE CASE: max_height reached
        if self.max_height is not None and height == self.max_height:
            return True
        # BASE CASE: not enough samples to split
        if len(y) < self.min_samples_split:
            return True

        # BASE CASE: no more columns to split
        if len(x.columns) == 0:
            return True

        # BASE CASE: the achieved error is within tolerance
        if tree.error <= self.error_tolerance:
            return True

        return False

    def post_split_prune(self, tree: Tree, best_column: SplitterResult):
        error_improvement = tree.error - best_column.error
        return error_improvement < self.min_error_decrease


class TreeTask:
    def __init__(
        self,
        parent: Tree,
        condition: Condition,
        x: pd.DataFrame,
        y: np.ndarray,
        height: int,
    ):
        self.parent = parent
        self.condition = condition
        self.x = x
        self.y = y
        self.height = height


class BaseTreeTrainer(TreeTrainer):

    def __init__(
        self,
        error: Splitter,
        prune: PruneCriteria,
        tree_creation_callbacks: list[TreeCreationCallback] = [],
        tree_split_callbacks: list[TreeSplitCallback] = [],
    ):
        self.prune = prune
        self.tree_creation_callbacks = tree_creation_callbacks
        self.tree_split_callbacks = tree_split_callbacks
        self.splitter = error

    def __repr__(self):
        return f"TreeTrainer({self.splitter},{self.prune})"

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> Tree:
        return self.build(x, y, 1)

    def build(self, x: pd.DataFrame, y: np.ndarray, height: int) -> Tree:
        tree, subtrees = self.make_tree(x, y, height)

        while len(subtrees) > 0:
            task = subtrees.pop()
            subtree, subtree_tasks = self.make_tree(task.x, task.y, task.height)
            task.parent.branches[task.condition] = subtree
            subtrees += subtree_tasks
        return tree

    def make_tree(
        self, x: pd.DataFrame, y: np.ndarray, height: int
    ) -> tuple[Tree, list[TreeTask]]:
        global_score = self.splitter.global_error(x, y)
        tree = Tree(global_score.prediction, global_score.error, y.shape[0])

        # BASE CASE: pre_split_prune
        if self.prune.pre_split_prune(x, y, height, tree):
            for callback in self.tree_creation_callbacks:
                callback(tree, height, True, None, None)
            return tree, []

        # COMPUTE SPLITS
        column_errors = self.splitter.split_columns(x, y)

        # BASE CASE: no viable columns to split found
        if len(column_errors) == 0:
            for callback in self.tree_creation_callbacks:
                callback(tree, height, True, None, None)
            return tree, []

        names, errors = zip(*[(k, s.error) for k, s in column_errors.items()])
        best_column_i = np.argmin(np.array(errors))
        best_column = column_errors[names[best_column_i]]

        # BASE CASE: best gain is not enough to split tree
        if self.prune.post_split_prune(tree, best_column):
            for callback in self.tree_creation_callbacks:
                callback(tree, height, True, column_errors, best_column)
            return tree, []

        for callback in self.tree_creation_callbacks:
            callback(tree, height, False, column_errors, best_column)

        # RECURSIVE CASE: use best attribute
        tree.column = best_column.column
        best_split = best_column.split
        subtrees = []
        for i, (x_branch, y_branch) in enumerate(best_split.partition):
            # avoid branches with low samples
            if len(y_branch) < self.prune.min_samples_leaf:
                continue
            # remove column from consideration
            if best_column.remove:
                x_branch = x_branch.drop(columns=[tree.column])
            condition = best_split.conditions[i]
            for callback in self.tree_split_callbacks:
                callback(tree, best_column, condition, x_branch, y_branch, height)
            subtrees.append(TreeTask(tree, condition, x_branch, y_branch, height + 1))

        return tree, subtrees
