import abc
import numpy as np
import pandas as pd

from sklearnmodels.tree.target_error import log
from sklearnmodels.tree.tree import Condition, split_by_conditions

class AttributePenalization(abc.ABC):
    
    
    def __init__(self):
        super().__init__()
    abc.abstractmethod
    def penalize(self,x:pd.DataFrame,column:str,conditions:list[Condition]):
        pass
        
    
    
class NoPenalization(AttributePenalization):
    
    def penalize(self, x:pd.DataFrame, column:str,conditions:list[Condition]):
        return 1
    

    
class GainRatioPenalization(AttributePenalization):
    
    def penalize(self, x:pd.DataFrame, column:str,conditions:list[Condition]):
        counts = np.array([len(s) for s in split_by_conditions(x,conditions)])
        counts /=counts.sum()
        return -np.sum(counts*log(counts,len(counts)))
            
            