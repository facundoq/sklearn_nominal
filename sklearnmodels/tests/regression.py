import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,mean_absolute_error
from sklearnmodels import tree

import pytest