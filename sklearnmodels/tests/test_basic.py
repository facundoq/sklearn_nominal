"""This file will just show how to write tests for the template classes."""
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.utils._testing import assert_allclose, assert_array_equal

from sklearnmodels import SKLearnClassificationTree

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_classification_tree(data):
    """Check the internals and behaviour of `TemplateEstimator`."""
    est = SKLearnClassificationTree()
    est.fit(*data)
    assert hasattr(est, "is_fitted_")
    assert hasattr(est, "classes_")
    assert hasattr(est, "tree_")

    X = data[0]
    y_pred = est.predict(X)
    assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.int64))
    assert y_pred.shape == (X.shape[0],)