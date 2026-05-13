import abc
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from sklearn_nominal.backend.core import Dataset
from sklearn_nominal.tree.pruning import PruneCriteria

from ..shared.column_error import ColumnErrorResult
from ..shared.global_error import Splitter
from .tree import Condition, Tree


class TreeTrainer(abc.ABC):
    @abc.abstractmethod
    def fit(self, d: Dataset) -> Tree:
        pass


class TreeTask:
    def __init__(
        self,
        parent: Tree,
        condition: Condition,
        d: Dataset,
        height: int,
    ):
        self.parent = parent
        self.condition = condition
        self.d = d
        self.height = height


@dataclass
class TreeCreationCallbackResult:
    tree: Tree
    task: TreeTask
    prune: bool = False
    best_column: ColumnErrorResult | None = None


TreeCreationCallback = Callable[[TreeCreationCallbackResult], None]


class BaseTreeTrainer(TreeTrainer):
    def __init__(
        self,
        error: Splitter,
        prune: PruneCriteria,
        tree_creation_callback: TreeCreationCallback | None = None,
    ):
        self.prune = prune
        self.tree_creation_callback = tree_creation_callback
        self.splitter = error

    def __repr__(self):
        return f"{self.__class__.__name__}({self.splitter},{self.prune})"

    def fit(self, d: Dataset) -> Tree:
        return self.build(d, 1)

    def do_creation_callback(self, r: TreeCreationCallbackResult):
        if self.tree_creation_callback is not None:
            self.tree_creation_callback(r)

    def build(self, d: Dataset, height: int) -> Tree:
        """
        Builds a decision tree from the given dataset.

        Parameters
        ----------
        d : Dataset
            The dataset to build the tree from.
        height : int
            The initial height of the tree (usually 1).

        Returns
        -------
        Tree
            The root of the constructed decision tree.
        """
        global_score = self.splitter.global_error(d)
        root = Tree(global_score.prediction, global_score.error, d.n)

        # Stack stores (current_tree, current_dataset, current_height)
        stack = [(root, d, height)]

        while stack:
            tree_node, dataset, node_height = stack.pop()

            # 1. Pre-split pruning
            if self.prune.pre_split_prune(dataset.x, dataset.y, node_height, tree_node):
                self.do_creation_callback(TreeCreationCallbackResult(tree_node, None, True))
                continue

            # 2. Find best split
            best_split = self.splitter.split_columns(dataset)

            # 3. Post-split pruning / Base cases
            if best_split is None or self.prune.post_split_prune(tree_node, best_split):
                self.do_creation_callback(TreeCreationCallbackResult(tree_node, None, best_column=best_split))
                continue

            self.do_creation_callback(TreeCreationCallbackResult(tree_node, None, best_column=best_split))

            # 4. Create child nodes
            for d_branch, condition in zip(best_split.partition, best_split.conditions):
                if d_branch.n < self.prune.min_samples_leaf:
                    continue

                if best_split.remove:
                    d_branch = d_branch.drop(columns=[best_split.column])

                branch_score = self.splitter.global_error(d_branch)
                child_tree = Tree(branch_score.prediction, branch_score.error, d_branch.n)
                tree_node.branches[condition] = child_tree

                stack.append((child_tree, d_branch, node_height + 1))

        return root
