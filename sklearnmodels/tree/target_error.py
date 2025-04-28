import abc
import numpy as np
import pandas as pd

from .conditions import Condition, Partition, Split


class TargetError(abc.ABC):
    @abc.abstractmethod
    def __call__(self, y: np.ndarray) -> float:
        pass

    def average_split(self, partition: Partition):
        error = 0.0
        n = 0
        for x_branch, y_branch in partition:
            branch_error = self(y_branch)
            n_branch = len(y_branch)
            error += n_branch * branch_error
            n += n_branch
        return error / n

    @abc.abstractmethod
    def prediction(self, y: np.ndarray):
        pass

    def __repr__(self):
        return self.__class__.__name__


eps = 1e-32


def log(x, base):
    x[x < eps] = eps
    if base == 2:
        return np.log2(x)
    elif base == 0:
        return np.log(x)
    elif base == 10:
        return np.log10(x)
    else:
        lb = 1 / np.log(base)
        return np.log(x) * lb


class ClassificationError(TargetError):
    def __init__(self, classes: int, class_weight: np.ndarray = None):
        self.classes = classes

        if class_weight is None:
            self.class_weight = np.ones(classes)
        else:
            self.class_weight = class_weight

    def prediction(self, y: np.ndarray):
        if len(y) == 0:
            result = np.ones(self.classes) / self.classes
        else:
            if np.issubdtype(y.dtype, object):
                # string classes
                y_cat = pd.Series(data=y.squeeze()).astype("category")
                y = y_cat.cat.codes.to_numpy().reshape(-1, 1)
            # numeric index classes classes
            counts = np.bincount(y, minlength=self.classes)
            result = counts / counts.sum()
            # print(result.dtype,self.class_weight.dtype)
        result *= self.class_weight
        result /= result.sum()
        return result

    def __repr__(self):
        return f"{super().__repr__()}(classes={self.classes})"


class EntropyError(ClassificationError):
    def __init__(self, classes: int, class_weight: np.ndarray, base=2):
        super().__init__(classes, class_weight)
        self.base = base

    def __call__(self, y: np.ndarray):
        p = self.prediction(y)
        # largest_value = log(np.array([self.classes]),self.base)[0]

        return -np.sum(p * log(p, self.classes))


class GiniError(ClassificationError):
    def __init__(self, classes: int, class_weight: np.ndarray, base=2):
        super().__init__(classes, class_weight)
        self.base = base

    def __call__(self, y: np.ndarray):
        p = self.prediction(y)
        # largest_value = log(np.array([self.classes]),self.base)[0]
        return 1 - np.sum(p**2)


class RegressionError(TargetError):
    def prediction(self, y: np.ndarray):
        return np.mean(y, axis=0)


class DeviationError(RegressionError):
    def __call__(self, y: np.ndarray):
        if y.shape[0] == 0:
            return np.inf
        return np.sum(np.std(y, axis=0))
