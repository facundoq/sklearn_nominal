import numpy as np
from sklearn.base import BaseEstimator

from sklearn_nominal.backend.core import Dataset
from sklearn_nominal.backend.factory import DEFAULT_BACKEND
from sklearn_nominal.bayes.trainer import NaiveBayesTrainer
from sklearn_nominal.scikit.nominal_model import NominalClassifier


class NaiveBayesClassifier(NominalClassifier, BaseEstimator):

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags

    def __init__(self, smoothing=0, backend=DEFAULT_BACKEND, class_weight=None):
        super().__init__(backend=backend, class_weight=class_weight)
        self.smoothing = smoothing

    def make_model(self, d: Dataset, class_weight: np.ndarray):
        return NaiveBayesTrainer(class_weight, smoothing=self.smoothing)
