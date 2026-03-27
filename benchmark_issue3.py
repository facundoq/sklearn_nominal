import time
import numpy as np
import pandas as pd
from sklearn_nominal.backend.pandas import PandasDataset
from sklearn_nominal.shared.column_error import NumericColumnError
from sklearn_nominal.shared.target_error import GiniError


def benchmark():
    n_samples = 10000
    n_features = 1
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    df = pd.DataFrame(X, columns=["col0"])
    ds = PandasDataset(df, y)

    metric = GiniError(2, np.array([1.0, 1.0]))
    ce = NumericColumnError(metric)

    start = time.time()
    res = ce.error(ds, "col0")
    end = time.time()

    print(f"Time taken for {n_samples} samples: {end - start:.4f}s")
    print(f"Best split: {res.conditions[0].value if res else 'None'}")


if __name__ == "__main__":
    benchmark()
