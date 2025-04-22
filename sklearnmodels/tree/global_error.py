import abc
import numpy as np
import pandas as pd

from sklearnmodels.tree.attribute_penalization import AttributePenalization, NoPenalization

from .target_error import TargetError

from .column_error import ColumnSplitter, ColumnSplitterResult

class GlobalErrorResult:
    def __init__(self,prediction:np.ndarray, error:float, ):
        self.prediction=prediction
        self.error=error

type ColumnErrors =  dict[str,ColumnSplitterResult]

class GlobalError(abc.ABC):
    @abc.abstractmethod
    def global_error(self, x:pd.DataFrame, y:np.ndarray)->GlobalErrorResult:
        pass

    @abc.abstractmethod
    def column_error(self, x:pd.DataFrame, y:np.ndarray)->ColumnErrors:
        pass

class MixedGlobalError(GlobalError):

    def __init__(self,column_splitters:dict[str,ColumnSplitter],error_function:TargetError,attribute_penalization:AttributePenalization):
        self.column_splitters = column_splitters
        self.target_error = error_function
        self.attribute_penalization=attribute_penalization
    def __repr__(self):
        return f"Error({self.target_error})"

    def global_error(self, x:pd.DataFrame, y:np.ndarray):
        global_metric = self.target_error(y)
        global_prediction = self.target_error.prediction(y)
        return GlobalErrorResult(global_prediction,global_metric)
    
    def column_error(self, x:pd.DataFrame, y:np.ndarray)->ColumnErrors:
        
        errors = {}
        for tipe, column_error in self.column_splitters.items():
            x_type = x.select_dtypes(include=tipe)
            for c in x_type.columns:
                error = column_error.error(x_type,y,c,self.target_error)
                penalization = self.attribute_penalization.penalize(x,error.column,error.conditions)
                errors[c]=error / penalization
        return errors
        