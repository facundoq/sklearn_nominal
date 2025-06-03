import numpy as np
import pandas as pd
from sklearnmodels.backend.core import Dataset
from sklearnmodels.backend.pandas import PandasDataset

DEFAULT_BACKEND = "pandas"


def pyarrow_backed_pandas(x: pd.DataFrame) -> pd.DataFrame:
    import pyarrow as pa

    pa_table = pa.Table.from_pydict(x)
    df = pa_table.to_pandas(types_mapper=pd.ArrowDtype)
    return df


def make_dataset(backend: str, x, y, columns: list[str], dtype) -> Dataset:

    if backend == "pandas":
        x = pd.DataFrame(x, columns=columns, dtype=dtype)
        assert isinstance(y, np.ndarray)
        return PandasDataset(x, y)
    if backend == "pandas_pyarrow":
        x = pd.DataFrame(x, columns=columns, dtype=dtype)
        assert isinstance(y, np.ndarray)
        x = pyarrow_backed_pandas(x)
        return PandasDataset(x, y)
    else:
        raise ValueError(f"Backend {backend} not supported")
