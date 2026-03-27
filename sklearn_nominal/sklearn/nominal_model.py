import abc

import numpy as np
import pandas as pd
import scipy
from pandas import DataFrame
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.calibration import LabelEncoder
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.utils import compute_class_weight, validation
from sklearn.utils._tags import (
    ClassifierTags,  # ty:ignore[unresolved-import]
    RegressorTags,  # ty:ignore[unresolved-import]
    Tags,  # ty:ignore[unresolved-import]
    TargetTags,  # ty:ignore[unresolved-import]
    TransformerTags,  # ty:ignore[unresolved-import]
    get_tags,  # ty:ignore[unresolved-import]
)
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_y, validate_data  # ty:ignore[unresolved-import]

from .. import shared, tree
from ..backend import Input, Output
from ..backend.core import Dataset, Model
from ..backend.factory import make_dataset
from ..backend.pandas import PandasDataset
from ..shared.target_error import TargetError


def atleast_2d(x):
    """View input as array with at least two dimensions.

    Args:
        x (array_like): Input data.

    Returns:
        ndarray: An array with at least two dimensions.
    """
    x = np.asanyarray(x)  # ty:ignore[unresolved-attribute]
    if x.ndim == 0:
        result = x.reshape(1, 1)
    elif x.ndim == 1:
        result = x[:, np.newaxis]
    else:
        result = x
    return result


# This is a mixin that must be used with sklearn estimators
class NominalModel(metaclass=abc.ABCMeta):
    """Mixin class for all nominal models in sklearn_nominal.

    This mixin provides the foundational infrastructure for models that natively
    handle nominal (categorical) attributes. It abstracts the complexities of
    managing different computation backends and provides a bridge between the
    scikit-learn API and the library's internal core logic.

    Architectural Context
    ---------------------
    `NominalModel` serves as the primary interface for backend abstraction. It
    manages the `model_` attribute, which stores the internal representation of
    the fitted model (from `sklearn_nominal.backend.core.Model`). It also
    handles the preservation of data types (dtypes) which is crucial for
    maintaining the nominal nature of features throughout the pipeline.

    Examples
    --------
    >>> from sklearn_nominal.sklearn.nominal_model import NominalModel
    >>> from sklearn.base import BaseEstimator
    >>> class MyNominalModel(NominalModel, BaseEstimator):
    ...     def __init__(self, backend='pandas'):
    ...         super().__init__(backend=backend)
    ...     def fit(self, X, y):
    ...         # implementation here
    ...         return self

    Attributes
    ----------
    backend : str
        The backend to use for computations (e.g., "pandas").
    model_ : sklearn_nominal.backend.core.Model
        The underlying fitted model object from the backend.
    is_fitted_ : bool
        Indicates whether the model has been successfully fitted.
    dtypes_ : pd.Series or list
        The data types of the features as observed during `fit`.
    """

    check_parameters = {"dtype": None}

    def __init__(self, backend: str = "pandas", *args, **kwargs):
        """Initializes the nominal model.

        Args:
            backend (str): The backend to use for computations. Defaults to "pandas".
            *args: Additional positional arguments for the parent class.
            **kwargs: Additional keyword arguments for the parent class.
        """
        super().__init__(*args, **kwargs)
        self.backend = backend

    def complexity(self):
        """Returns the complexity of the fitted model.

        The definition of complexity is backend and model dependent. For trees,
        it typically represents the number of nodes.

        Returns
        -------
        int or float
            The complexity metric of the model.

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        """
        self.check_is_fitted()
        return self.model_.complexity()

    def set_sklearn_tags(self, tags):
        """Sets scikit-learn tags for the nominal model.

        Configures the estimator tags to accurately reflect its capabilities,
        specifically its ability to handle string inputs and missing values
        natively.

        Parameters
        ----------
        tags : Tags
            The scikit-learn tags object to be modified in-place.
        """
        tags.non_deterministic = False
        tags.input_tags.sparse = False
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True

    def pretty_print(self, class_names: list[str] | None = None):
        """Returns a string representation of the fitted model.

        Delegates the visualization logic to the underlying backend model.

        Parameters
        ----------
        class_names : list of str, optional
            Names of the classes to use in the output. If None, default
            identifiers are used.

        Returns
        -------
        str
            A human-readable representation of the model.

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        """
        self.check_is_fitted()
        return self.model_.pretty_print(class_names=class_names)

    def check_is_fitted(self):
        """Checks if the model has been fitted.

        Raises
        ------
        NotFittedError
            If the `is_fitted_` attribute is not set or is False.
        """
        if not hasattr(self, "is_fitted_") or not self.is_fitted_:
            raise NotFittedError()

    def get_dtypes(self, x):
        """Extracts and maps data types from the input.

        This method identifies the data types of the input features to ensure
        they are correctly handled by the backend.

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        dict or None
            A dictionary mapping column names to data types if `x` is a
            DataFrame, otherwise None.
        """
        if isinstance(x, pd.DataFrame):
            dtypes = x.dtypes.to_dict()
        else:
            dtypes = None
        return dtypes

    def get_feature_names(self):
        """Returns the names of the features seen during fit.

        Returns
        -------
        ndarray of str or None
            The feature names, or None if they were not available during fit
            (e.g., if input was a numpy array).
        """
        if not hasattr(self, "feature_names_in_"):
            return None
        else:
            return self.feature_names_in_

    def set_dtypes(self, x):
        """Sets and persists the data types based on the input.

        This is called during `fit` to ensure that subsequent calls to
        `predict` can cast the input data to the same types, preserving
        nominal/numeric distinctions.

        Parameters
        ----------
        x : {pd.DataFrame, np.ndarray, sparse matrix}
            The input data to extract types from.

        Raises
        ------
        ValueError
            If the input type is not supported or if the input is not 2D.
        """
        if isinstance(x, pd.DataFrame):
            self.dtypes_ = x.dtypes
        elif isinstance(x, np.ndarray) or scipy.sparse.issparse(x):
            if len(x.shape) != 2:
                raise ValueError(f"Expected 2d input, actual shape {x.shape}")
            self.dtypes_ = [x.dtype] * x.shape[1]
        else:
            raise ValueError(f"Only pd.Dataframe or np.ndarray supported, received: {x}")

    def set_model(self, model):
        """Sets the underlying backend model and marks it as fitted.

        Parameters
        ----------
        model : sklearn_nominal.backend.core.Model
            The trained model instance from the backend.
        """
        self.model_: Model = model
        self.is_fitted_ = True


class NominalUnsupervisedModel(NominalModel):
    """Base class for unsupervised nominal models.

    Extends `NominalModel` to configure tags for unsupervised tasks.
    """

    def set_sklearn_tags(self, tags):
        """Sets scikit-learn tags for the unsupervised nominal model.

        Parameters
        ----------
        tags : Tags
            The scikit-learn tags object to be modified.

        Returns
        -------
        Tags
            The modified tags object.
        """
        super().set_sklearn_tags(tags)
        tags.target_tags.single_output = False
        tags.target_tags.required = False
        return tags


class NominalSupervisedModel(NominalModel):
    """Base class for supervised nominal models.

    Extends `NominalModel` to provide shared validation logic for supervised
    learning tasks.
    """

    def set_sklearn_tags(self, tags):
        """Sets scikit-learn tags for the supervised nominal model.

        Parameters
        ----------
        tags : Tags
            The scikit-learn tags object to be modified.
        """
        super().set_sklearn_tags(tags)
        tags.target_tags.single_output = False
        tags.target_tags.required = True

    def validate_data_predict(self, x):
        """Validates and prepares input data for prediction.

        This method ensures the input features match the structure seen during
        training, handles feature name alignment, and restores data types.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The input data to validate.

        Returns
        -------
        pd.DataFrame
            The validated data as a pandas DataFrame, with dtypes restored to
            match those observed during training.

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        ValueError
            If the input contains no samples or has inconsistent features.
        """
        dtypes = self.get_dtypes(x)
        self.check_is_fitted()
        x = validate_data(
            self,
            x,
            reset=False,
            dtype=None,
            ensure_all_finite=False,
            accept_sparse=False,
        )
        n = len(x)
        if n == 0:
            raise ValueError(f"Input contains 0 samples.")
        df = pd.DataFrame(x, columns=self.get_feature_names())
        if dtypes is not None:
            df = df.astype(dtypes)
        return df


class NominalClassifier(NominalSupervisedModel, ClassifierMixin):
    """Base class for nominal classifiers.

    This class coordinates the end-to-end classification workflow, including
    target encoding, class weight computation, and delegation to backend
    trainers.

    Architectural Context
    ---------------------
    `NominalClassifier` implements the standard scikit-learn `fit`/`predict`
    cycle while delegating the actual training logic to an internal "trainer"
    (created via `make_model`). It handles the transformation of target
    labels using `LabelEncoder` to ensure the backend receives integer-encoded
    classes, while providing inverse transformations for user-facing output.

    Examples
    --------
    >>> from sklearn_nominal.sklearn.nominal_model import NominalClassifier
    >>> class MyClassifier(NominalClassifier):
    ...     def make_model(self, d, class_weight):
    ...         # Return a backend-specific trainer
    ...         pass

    Attributes
    ----------
    class_weight : dict, list of dicts or "balanced", default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.
    classes_ : ndarray of shape (n_classes,)
        The unique class labels observed during `fit`.
    le_ : sklearn.preprocessing.LabelEncoder
        The encoder used to map class labels to internal integer indices.
    """

    def __init__(self, class_weight=None, *args, **kwargs):
        """Initializes the nominal classifier.

        Parameters
        ----------
        class_weight : dict, list of dicts or "balanced", optional
            Weights associated with classes. Defaults to None.
        *args : list
            Additional positional arguments for the parent class.
        **kwargs : dict
            Additional keyword arguments for the parent class.
        """
        super().__init__(*args, **kwargs)

        self.class_weight = class_weight

    def __sklearn_tags__(self):
        """Returns the scikit-learn tags for the classifier.

        Returns
        -------
        Tags
            The scikit-learn tags configured for a multi-class classifier.
        """
        tags = super().__sklearn_tags__()  # ty:ignore[unresolved-attribute]
        self.set_sklearn_tags(tags)
        tags.classifier_tags.multi_class = True
        return tags

    def validate_data_fit_classification(self, x, y) -> tuple[Dataset, np.ndarray]:
        """Validates and transforms data for classification fitting.

        This method performs the following transformations:
        1. Validates `x` and `y` using scikit-learn's `validate_data`.
        2. Determines the unique classes and stores them in `classes_`.
        3. Encodes `y` using `LabelEncoder`.
        4. Calculates class weights based on the `class_weight` parameter.
        5. Packages features and the encoded target into a backend-specific
           `Dataset` (e.g., `PandasDataset`).

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The input features.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target labels.

        Returns
        -------
        tuple
            A tuple containing:
            - Dataset : The backend-specific dataset object.
            - np.ndarray : The computed class weights for each class in `classes_`.

        Raises
        ------
        ValueError
            If `y` contains only one unique class.
        """
        check_classification_targets(y)
        dtypes = self.get_dtypes(x)
        x, y = validate_data(
            self,
            x,
            y,
            reset=True,
            multi_output=True,
            y_numeric=False,
            ensure_all_finite=False,
            dtype=None,
            accept_sparse=False,
        )
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("Can't train classifier with one class.")
        # dtype = x_original.dtype
        class_weight = self.get_class_weights(y)
        dataset = make_dataset(self.backend, x, self.get_y(y), self.get_feature_names(), dtypes)
        return dataset, class_weight

    def get_y(self, y):
        """Validates and encodes the target labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            The target labels to encode.

        Returns
        -------
        ndarray
            The integer-encoded target labels.
        """
        y = _check_y(y, multi_output=True, y_numeric=False, estimator=self)
        # TODO make pure numpy
        self.le_ = LabelEncoder()
        y = self.le_.fit_transform(y)
        return y

    @abc.abstractmethod
    def make_model(self, d: Dataset, class_weight: np.ndarray):
        """Abstract method to create the model trainer.

        Implementation Guidance
        -----------------------
        Subclasses must implement this method to return a trainer object that
        follows the internal library API (specifically, it should have a
        `.fit(dataset)` method).

        For example, a decision tree classifier would return a
        `sklearn_nominal.tree.trainer.TreeTrainer` instance configured with
        the appropriate split criteria and pruning parameters.

        Parameters
        ----------
        d : Dataset
            The training dataset prepared by `validate_data_fit_classification`.
        class_weight : np.ndarray
            The class weights computed during validation.

        Returns
        -------
        trainer : object
            A trainer instance capable of fitting the provided dataset.
        """
        pass

    def fit(self, x: Input, y: Output):
        """Fits the nominal classifier.

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        d, class_weight = self.validate_data_fit_classification(x, y)

        trainer = self.make_model(d, class_weight)
        model = trainer.fit(d)
        self.set_model(model)
        return self

    def get_class_weights(self, y):
        """Computes the class weights based on the input target.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            The target labels.

        Returns
        -------
        np.ndarray
            The computed weights for each class, aligned with `self.classes_`.
        """
        return compute_class_weight(class_weight=self.class_weight, classes=self.classes_, y=y)

    def build_error(self, criterion: str, class_weight: np.ndarray) -> TargetError:
        """Builds the error function for the given criterion.

        Parameters
        ----------
        criterion : str
            The error criterion to use (e.g., "entropy", "gini", "gain_ratio").
        class_weight : np.ndarray
            The class weights to be used by the error function.

        Returns
        -------
        TargetError
            An instance of the requested error function from `sklearn_nominal.shared`.

        Raises
        ------
        ValueError
            If the criterion is not recognized.
        """
        classes = len(class_weight)
        errors = {
            "entropy": shared.EntropyError(classes, class_weight),
            "gini": shared.GiniError(classes, class_weight),
            "gain_ratio": shared.EntropyError(classes, class_weight),
        }
        if criterion not in errors.keys():
            raise ValueError(f"Unknown error function {criterion}")
        return errors[criterion]

    def predict_proba(self, x: Input) -> Output:
        """Predicts class probabilities for input samples.

        This method first validates the input data to ensure compatibility
        with the fitted model, then delegates the prediction to the backend
        `model_`.

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        self.check_is_fitted()
        x = self.validate_data_predict(x)
        y = self.model_.predict(x)
        return y

    def predict(self, x: Input) -> Output:
        """Predicts class labels for input samples.

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        ndarray of shape (n_samples,)
            The predicted class labels.
        """
        p = self.predict_proba(x)
        c = p.argmax(axis=1)
        y = self.le_.inverse_transform(c)
        return y


class NominalRegressor(NominalSupervisedModel, RegressorMixin):
    """Base class for nominal regressors.

    This class coordinates the regression workflow for models that handle
    nominal features natively.

    Architectural Context
    ---------------------
    `NominalRegressor` mirrors the structure of `NominalClassifier` but
    specializes in continuous target variables. It manages the conversion
    of targets to at least 2D arrays to ensure consistent handling by the
    backend and provides error function building logic tailored for
    regression (e.g., standard deviation reduction).

    Examples
    --------
    >>> from sklearn_nominal.sklearn.nominal_model import NominalRegressor
    >>> class MyRegressor(NominalRegressor):
    ...     def make_model(self, d):
    ...         # Return a backend-specific trainer
    ...         pass
    """

    def __sklearn_tags__(self):
        """Returns the scikit-learn tags for the regressor.

        Returns
        -------
        Tags
            The scikit-learn tags configured for a regressor.
        """
        tags = super().__sklearn_tags__()  # ty:ignore[unresolved-attribute]
        self.set_sklearn_tags(tags)
        return tags

    def validate_data_fit_regression(self, x, y) -> Dataset:
        """Validates and prepares data for regression fitting.

        This method ensures `x` and `y` are compatible, extracts data types,
        and packages them into a backend `Dataset`. It also ensures the target
        `y` is at least 2D for backend consistency.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The input features.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values.

        Returns
        -------
        Dataset
            The backend-specific dataset object.
        """
        dtypes = self.get_dtypes(x)
        x, y = validate_data(
            self,
            x,
            y,
            reset=True,
            multi_output=True,
            y_numeric=True,
            ensure_all_finite=False,
            dtype=None,
            accept_sparse=False,
        )
        y = _check_y(y, multi_output=True, y_numeric=True, estimator=self)
        self._y_original_shape = y.shape
        y = atleast_2d(y)
        return make_dataset(self.backend, x, y, self.get_feature_names(), dtypes)

    def build_error(self, criterion: str):
        """Builds the regression error function for the given criterion.

        Parameters
        ----------
        criterion : str
            The error criterion to use (e.g., "std" for standard deviation).

        Returns
        -------
        TargetError
            An instance of the requested error function.

        Raises
        ------
        ValueError
            If the criterion is not recognized.
        """
        errors = {
            "std": shared.DeviationError(),
        }
        if criterion not in errors.keys():
            raise ValueError(f"Unknown error function {criterion}")
        return errors[criterion]

    def predict(self, x: Input):
        """Predicts target values for input samples.

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted target values.
        """
        self.check_is_fitted()
        x = self.validate_data_predict(x)
        y = self.model_.predict(x)
        if len(self._y_original_shape) == 1:
            y = y.squeeze()
        return y

    @abc.abstractmethod
    def make_model(self, d: Dataset):
        """Abstract method to create the model trainer.

        Implementation Guidance
        -----------------------
        Subclasses must implement this method to return a trainer object that
        follows the internal library API. For example, a decision tree
        regressor would return a `sklearn_nominal.tree.trainer.TreeTrainer`
        instance configured with regression-specific criteria.

        Parameters
        ----------
        d : Dataset
            The training dataset prepared by `validate_data_fit_regression`.

        Returns
        -------
        trainer : object
            A trainer instance capable of fitting the provided dataset.
        """
        pass

    def fit(self, x: Input, y: Output):
        """Fits the nominal regressor.

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        d = self.validate_data_fit_regression(x, y)
        trainer = self.make_model(d)
        model = trainer.fit(d)
        self.set_model(model)
        return self
