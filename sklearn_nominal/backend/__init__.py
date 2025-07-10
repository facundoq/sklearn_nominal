import numpy as np
import pandas as pd

Output = np.ndarray
Input = pd.DataFrame | np.ndarray
InputSample = pd.Series
ColumnID = int

from .conditions import Condition, RangeCondition, ValueCondition

from .core import ColumnType, Dataset
from .pandas import PandasDataset


from .factory import make_dataset
