import abc
from typing import Callable
import numpy as np
import pandas as pd

from . import Tree
from . import TreeTrainer

class SKLearnTree:
    def __init__(self,trainer:TreeTrainer):
        self.trainer=trainer
        self.tree: Tree|None = None

    def fit(self,x:pd.DataFrame,y:np.ndarray):
        self.tree = self.trainer.fit(x,y)

    def predict_base(self,x:pd.DataFrame):
        assert self.tree is not None, "Must `fit` tree before calling `predict`-like methods."   # noqa: F631
        n = len(x)
        assert(n>0)
        predictions = np.zeros((n,len(self.tree.prediction)))
        for i,row in x.iterrows():
            predictions[i,:]=self.tree.predict(row)
        return predictions
        
class SKLearnClassificationTree(SKLearnTree):

    def predict_proba(self,x:pd.DataFrame):
        return self.predict_base(x)
    def predict(self,x:pd.DataFrame):
        return self.predict_proba(x).argmax(axis=1)
    
class SKLearnRegressionTree(SKLearnTree):
    def predict(self,x:pd.DataFrame):
        return self.predict_base(x)
        
