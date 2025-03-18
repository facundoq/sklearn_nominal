import abc
from typing import Callable

from numpy import isnan
from .tree import Condition, split_by_conditions
import numpy as np
import pandas as pd

from .target_error import TargetError


class ValueCondition(Condition):
    def __init__(self,column:str,value):
        self.column=column
        self.value=value
    def __call__(self,x:pd.Series):
        return x[self.column]==self.value
    def __repr__(self):
        return f"{self.column}={self.value}"
    def short_description(self):
        return f"{self.value}"
    

class RangeCondition(Condition):
    def __init__(self,column:str,value:float,less:bool):
        self.column=column
        self.value=value
        self.less=less
    def __call__(self,x:pd.Series):
        if self.less:
            return x[self.column]<=self.value
        else:
            return x[self.column]>self.value
    def __repr__(self):
        op = "<=" if self.less else ">"
        return f"{self.column} {op} {self.value:.4g}"
    def short_description(self):
        op = "<=" if self.less else ">"
        return f"{op} {self.value:.4g}"



class ColumnSplitterResult:
    def __init__(self,error:float,conditions:list[Condition],column:str,remove:bool):
        self.error=error
        self.conditions=conditions
        self.column=column
        self.remove = remove
    def __repr__(self):
        return f"Score({self.column},{self.error},{len(self.conditions)} branches)"
    
class ColumnSplitter(abc.ABC):

    @abc.abstractmethod
    def error(self,x:pd.DataFrame,y:np.ndarray,column:str,metric:TargetError)->ColumnSplitterResult:
        pass

    def __repr__(self):
        return self.__class__.__name__
    

class DiscretizationStrategy:

    @abc.abstractmethod
    def __call__(self,x:pd.Series,y:np.ndarray,column:str,metric:TargetError)->list[Condition]:
        pass

class MeanDiscretizationStrategy(DiscretizationStrategy):

    def __call__(self, x:pd.Series,y:np.ndarray,column:str,metric:TargetError)->list[Condition]:
        mu = x.mean()
        return [RangeCondition(column,mu,True),RangeCondition(column,mu,False)]

type ConditionEvaluationCallback = Callable[[str,np.ndarray,np.ndarray],None]

class OptimizingDiscretizationStrategy(DiscretizationStrategy):

    def __init__(self,max_evals:int=np.iinfo(np.int64).max,callbacks:list[ConditionEvaluationCallback]=[]):
        assert max_evals>0
        self.max_evals=max_evals
        self.callbacks = callbacks
    def __call__(self, x:pd.Series,y:np.ndarray,column:str,metric:TargetError)->list[Condition]:
        
        values = x.unique()
        n = len(values)
        if self.max_evals is not None:
            if n>self.max_evals:
                #subsample
                step = n//self.max_evals
                values=values[::step]
                n = len(values)
        values.sort()

        if n>1:
            values = values[:-1]
            n-=1

        # find best split value based on unique values of column
        errors = np.zeros(n)
        for i,v in enumerate(values):
            low,high = y[x<=v],y[x>v]
            n_low,n_high = low.shape[0],high.shape[0]
            errors[i] = (metric(low)*n_low+metric(high)*n_high)/n
        for callback in self.callbacks:
            callback(column,values,errors)
        best_i = np.argmin(errors)
        best_v = values[best_i]
        return [RangeCondition(column,best_v,True),RangeCondition(column,best_v,False)]

class DiscretizingNumericColumnSplitter(ColumnSplitter):
    def __init__(self,strategy:DiscretizationStrategy):
        super().__init__()
        self.strategy = strategy

    def error(self,x:pd.DataFrame,y:np.ndarray,column:str,metric:TargetError,)->ColumnSplitterResult:

        conditions = self.strategy(x[column],y,column,metric)
        n = len(y)
        error = 0.0
        for x_branch,y_branch,condition in split_by_conditions(x,y,conditions):
            p = len(y_branch) /n
            error += p* metric(y_branch)
        return ColumnSplitterResult(error,conditions,column,False)
    

class NominalColumnSplitter(ColumnSplitter):

    def error(self,x:pd.DataFrame,y:np.ndarray,column:str,metric:TargetError)->ColumnSplitterResult:
        conditions = [ValueCondition(column,v) for v in x[column].unique()]
        n = len(y)
        error = 0.0
        for x_branch,y_branch,condition in split_by_conditions(x,y,conditions):
            p = len(y_branch) /n
            error += p* metric(y_branch)
        return ColumnSplitterResult(error,conditions,column,True)