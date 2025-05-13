from sklearnmodels.rules.prism import ZeroRClassifier
from sklearnmodels.scikit.nominal_model import NominalClassifier, NominalModel
import sklearnmodels.tree.pruning
from sklearnmodels import tree
from sklearnmodels.backend.pandas import PandasDataset
from sklearnmodels.scikit.tree_scikit import SKLearnTree


import numpy as np
import pandas as pd


class ZeroRClassifier(NominalClassifier):

    def fit(self, x: pd.DataFrame, y: np.ndarray):
        d = self.validate_data_fit_classification(x, y)
        trainer = ZeroRClassifier()
        model = trainer.fit(d, self.class_weight)
        self.set_model(model)
        return self
