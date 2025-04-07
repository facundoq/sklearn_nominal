import numpy as np
import pandas as pd

from . import Tree
from . import TreeTrainer

from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels

class SKLearnTree(ClassifierMixin,BaseEstimator):
    def __init__(self):
        self.trainer=trainer
        self.tree_: Tree|None = None

    def fit(self,x:pd.DataFrame,y:np.ndarray):
        x, y = self._validate_data(x, y, accept_sparse=True)
        self.classes_ = unique_labels(y)

        self.tree_ = self.trainer.fit(x,y)
        self.is_fitted_ = True
        return self


    def predict_base(self,x:pd.DataFrame):
        assert self.tree_ is not None, "Must `fit` tree before calling `predict`-like methods."   # noqa: F631
        check_is_fitted(self)
        x = self._validate_data(x, accept_sparse=True, reset=False)
        n = len(x)
        assert(n>0)
        predictions = np.zeros((n,len(self.tree_.prediction)))
        for i,row in x.iterrows():
            predictions[i,:]=self.tree_.predict(row)
        return predictions
        
class SKLearnClassificationTree(SKLearnTree,ClassifierMixin):
    def __init__(self, criterion='entropy', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,min_impurity_decrease=0.0, class_weight=None ):
        
    def predict_proba(self,x:pd.DataFrame):
        return self.predict_base(x)
    def predict(self,x:pd.DataFrame):
        return self.predict_proba(x).argmax(axis=1)
    
class SKLearnRegressionTree(SKLearnTree):
    def predict(self,x:pd.DataFrame):
        return self.predict_base(x)
        
