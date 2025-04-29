import abc
from typing import Callable

import numpy as np
import pandas as pd

from .column_error import SplitterResult
from .global_error import ColumnErrors, GlobalSplitter
from .tree import Condition, Tree


class TreeTrainer(abc.ABC):

    @abc.abstractmethod
    def fit(self, x: pd.DataFrame, y: np.ndarray) -> Tree:
        pass


type TreeCreationCallbackResult = tuple[
    Tree, TreeTask, bool, ColumnErrors, SplitterResult
]
type TreeCreationCallback = Callable[[TreeCreationCallbackResult], None]

type TreeSplitCallbackResult = tuple[TreeTask, SplitterResult]
type TreeSplitCallback = Callable[[TreeSplitCallbackResult], None]


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
        error: GlobalSplitter,
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
        # ROOT
        global_score = self.splitter.global_error(x, y)
        root = Tree(global_score.prediction, global_score.error, y.shape[0])
        root_task = TreeTask(None, None, x, y, height)
        subtrees = self.make_tree(root, root_task)

        # OTHER NODES
        while len(subtrees) > 0:
            task = subtrees.pop()
            global_score = self.splitter.global_error(task.x, task.y)
            new_tree = Tree(global_score.prediction, global_score.error, y.shape[0])
            task.parent.branches[task.condition] = new_tree
            subtree_tasks = self.make_tree(new_tree, task)
            # bfs
            subtree_tasks.reverse()
            subtrees = subtrees + subtree_tasks
        return root

    def run_creation_callbacks(self, result: TreeCreationCallbackResult):

        for callback in self.tree_creation_callbacks:
            callback(result)

    def make_tree(self, tree: Tree, task: TreeTask) -> list[TreeTask]:
        # BASE CASE: pre_split_prune
        if self.prune.pre_split_prune(task.x, task.y, task.height, tree):
            self.run_creation_callbacks(
                (
                    tree,
                    task,
                    True,
                    None,
                    None,
                )
            )
            return []

        # COMPUTE SPLITS
        column_errors = self.splitter.split_columns(task.x, task.y)

        # BASE CASE: no viable columns to split found
        if len(column_errors) == 0:
            self.run_creation_callbacks((tree, task, True, None, None))
            return []

        names, errors = zip(*[(k, s.error) for k, s in column_errors.items()])

        best_column_i = np.argmin(np.array(errors))
        best_column = column_errors[names[best_column_i]]

        # BASE CASE: best gain is not enough to split tree
        if self.prune.post_split_prune(tree, best_column):
            self.run_creation_callbacks((tree, task, True, column_errors, best_column))
            return []

        self.run_creation_callbacks((tree, task, False, column_errors, best_column))

        # RECURSIVE CASE: use best attribute
        tree.column = best_column.column
        best_split = best_column.split
        subtrees = []

        for i, (x_branch, y_branch) in enumerate(best_split.partition):
            # avoid branches with low samples
            if len(y_branch) < self.prune.min_samples_leaf:
                continue
            condition = best_split.conditions[i]

            # remove column from consideration
            if best_column.remove:
                x_branch = x_branch.drop(columns=[best_column.column])
            subtask = TreeTask(tree, condition, x_branch, y_branch, task.height + 1)

            for callback in self.tree_split_callbacks:
                callback((subtask, best_column))
            subtrees.append(subtask)

        return subtrees
