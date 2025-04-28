"""This file will just show how to write tests for the template classes."""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_diabetes
from sklearn.utils._testing import assert_allclose, assert_array_equal

from sklearnmodels import SKLearnClassificationTree
from sklearnmodels.scikit import SKLearnRegressionTree

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause


@pytest.fixture
def classification_data():
    return load_iris(return_X_y=True, as_frame=True)


def test_classification_tree(classification_data):
    est = SKLearnClassificationTree()
    est.fit(*classification_data)
    assert hasattr(est, "is_fitted_")
    assert hasattr(est, "classes_")
    assert hasattr(est, "tree_")

    X = classification_data[0]
    y_pred = est.predict(X)
    assert y_pred.shape == (X.shape[0],)


@pytest.fixture
def regression_data():
    return load_diabetes(return_X_y=True, as_frame=True)


def test_regression_tree(regression_data):
    x, y = regression_data
    print(y.shape)
    model = SKLearnRegressionTree()
    model.fit(x, y)
    print(y.shape)
    assert hasattr(model, "is_fitted_")
    assert hasattr(model, "tree_")
    y_pred = model.predict(x)
    assert y_pred.shape == (x.shape[0],)
