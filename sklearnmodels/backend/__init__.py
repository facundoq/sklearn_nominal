import numpy as np
import pandas as pd

Output = np.ndarray
Input = pd.DataFrame
InputSample = pd.Series


from .conditions import Condition, RangeCondition, ValueCondition

from .core import ColumnType, Dataset
from .split import RangeCondition, ValueCondition
from .pandas import PandasDataset


from .factory import make_dataset
