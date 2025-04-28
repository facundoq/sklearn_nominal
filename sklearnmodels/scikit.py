from multiprocessing import Value
import numpy as np
import pandas as pd

from sklearnmodels.tree.attribute_penalization import NoPenalization

from . import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, _fit_context
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import validation
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import validate_data, check_X_y, _check_y
import abc


def atleast_2d(x):
    x = np.asanyarray(x)
    if x.ndim == 0:
        result = x.reshape(1, 1)
    elif x.ndim == 1:
        result = x[:, np.newaxis]
    else:
        result = x
    return result


def pyarrow_backed_pandas(x: pd.DataFrame) -> pd.DataFrame:
    import pyarrow as pa

    pa_table = pa.Table.from_pydict(x)
    df = pa_table.to_pandas(types_mapper=pd.ArrowDtype)
    return df


class SKLearnTree(BaseEstimator, metaclass=abc.ABCMeta):
    check_parameters = {"dtype": None}

    @abc.abstractmethod
    def __init__(
        self,
        criterion="",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_error_decrease=0.0,
    ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_error_decrease = min_error_decrease

    def check_is_fitted(self):
        validation.check_is_fitted(self, "tree_")

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.single_output = False
        tags.non_deterministic = False
        tags.input_tags.sparse = False
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        return tags

    def get_dataframe_from_x(self, x: np.ndarray):
        if not hasattr(self, "feature_names_in_"):
            columns = list([f"feature{i}" for i in range(self.n_features_in_)])
        else:
            columns = self.feature_names_in_
        df = pd.DataFrame(x, columns=columns)
        return pyarrow_backed_pandas(df)

    def build_attribute_penalizer(self):
        if self.criterion == "gain_ratio":
            return tree.GainRatioPenalization()
        else:
            return tree.NoPenalization()

    def build_splitter(self):
        if self.splitter == "best":
            max_evals = np.iinfo(np.int64).max
        elif isinstance(self.splitter, int):
            max_evals = self.splitter
        else:
            raise ValueError(
                f"Invalid value '{self.splitter}' for splitter; expected integer or"
                " 'best'"
            )
        scorers = {
            "number": tree.NumericSplitter(max_evals=max_evals),
            "object": tree.NominalSplitter(),
            "category": tree.NominalSplitter(),
        }
        return scorers

    def predict_base(self, x: pd.DataFrame):
        self.check_is_fitted()
        x = validate_data(
            self,
            x,
            accept_sparse=True,
            reset=False,
            dtype=None,
            ensure_all_finite=False,
        )
        x_df = self.get_dataframe_from_x(x)
        n = len(x_df)
        assert n > 0
        predictions = np.zeros((n, len(self.tree_.prediction)))
        for i, row in x_df.iterrows():
            predictions[i, :] = self.tree_.predict(row)
        return predictions


class SKLearnClassificationTree(ClassifierMixin, SKLearnTree):
    def __init__(
        self,
        criterion="entropy",
        splitter="best",
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=5,
        min_error_decrease=0.0,
        class_weight=None,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_error_decrease=min_error_decrease,
        )
        self.class_weight = class_weight

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags.multi_class = True
        return tags

    def build_trainer(self, error: tree.TargetError):
        attribute_penalization = self.build_attribute_penalizer()

        scorers = self.build_splitter()

        scorer = tree.MixedSplitter(scorers, error, attribute_penalization)
        prune_criteria = tree.PruneCriteria(
            max_height=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_error_decrease=self.min_error_decrease,
            min_samples_split=self.min_samples_split,
        )
        trainer = tree.BaseTreeTrainer(scorer, prune_criteria)
        return trainer

    # def _decode_y(self,y:np.ndarray):
    #     return self.le_.inverse_transform(y)

    # def _encode_y(self,y:np.ndarray):
    #     self.le_ = LabelEncoder()
    #     encoded_y = self.le_.fit_transform(y)
    #     self.classes_ = self.le_.classes_
    #     if self.class_weight is not None:
    #         # get the classes in the same order as assigned by the transform
    #         ordered_classes = self.le_le.transform(self.le_.inverse_transform(self.classes_))
    #         # use the ordered classes to obtain ordered class weights
    #         self._ordered_class_weights = np.array([self.class_weight[c] for c in ordered_classes])
    #     else:
    #         self._ordered_class_weights = None
    #     return encoded_y

    def fit(self, x: pd.DataFrame, y: np.ndarray):
        check_classification_targets(y)
        # check_X_params = dict(dtype=None, accept_sparse=False, ensure_all_finite=False,ensure_2d=True)
        # check_y_params = dict(ensure_2d=False,accept_sparse=False)
        # x, y = validate_data(self, x, y, reset=True,multi_output=False,y_numeric=True, validate_separately=(check_X_params, check_y_params))
        # check_X_y(x,y,accept_sparse=False,dtype=None,ensure_all_finite=False)
        x, y = validate_data(
            self,
            x,
            y,
            reset=True,
            multi_output=True,
            y_numeric=False,
            ensure_all_finite=False,
            dtype=None,
        )
        y = _check_y(y, multi_output=True, y_numeric=False, estimator=self)
        self.classes_ = np.unique(y)
        # y = self._encode_y(y)
        x_df = self.get_dataframe_from_x(x)
        if len(self.classes_) < 2:
            raise ValueError("Can't train classifier with one class.")
        error = self.build_error(len(self.classes_))
        trainer = self.build_trainer(error)
        self.tree_ = trainer.fit(x_df, y)
        self.is_fitted_ = True
        return self

    def build_error(self, classes: int):
        errors = {
            "entropy": tree.EntropyError(classes, self.class_weight),
            "gini": tree.GiniError(classes, self.class_weight),
            "gain_ratio": tree.EntropyError(classes, self.class_weight),
        }
        if self.criterion not in errors.keys():
            raise ValueError(f"Unknown error function {self.criterion}")
        return errors[self.criterion]

    def predict_proba(self, x: pd.DataFrame):
        return self.predict_base(x)

    def predict(self, x: pd.DataFrame):
        p = self.predict_proba(x)
        c = p.argmax(axis=1)
        return c

    def export_dot(self, filepath, class_names=None):
        tree.export_dot_file(self.tree_, filepath, class_names=class_names)

    def export_image(self, filepath, class_names=None):
        tree.export_image(self.tree_, filepath, class_names=class_names)


class SKLearnRegressionTree(RegressorMixin, SKLearnTree):
    def __init__(
        self,
        criterion="std",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_error_decrease=0.0,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_error_decrease=min_error_decrease,
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        return tags

    def build_trainer(self, error: tree.TargetError):
        scorers = self.build_splitter()
        attribute_penalization = self.build_attribute_penalizer()
        scorer = tree.MixedSplitter(scorers, error, attribute_penalization)
        prune_criteria = tree.PruneCriteria(
            max_height=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_error_decrease=self.min_error_decrease,
            min_samples_split=self.min_samples_split,
        )
        trainer = tree.BaseTreeTrainer(scorer, prune_criteria)
        return trainer

    def fit(self, x: pd.DataFrame, y: np.ndarray):
        # check_X_params = dict(dtype=None, accept_sparse=False, ensure_all_finite=False,ensure_2d=True)
        # check_y_params = dict(ensure_2d=False,ensure_all_finite=True)
        # x, y = validate_data(self, x, y, reset=True,multi_output=True,y_numeric=True, validate_separately=(check_X_params, check_y_params))
        # check_X_y(x,y,accept_sparse=False,dtype=None,ensure_all_finite=False)
        x, y = validate_data(
            self,
            x,
            y,
            reset=True,
            multi_output=True,
            y_numeric=True,
            ensure_all_finite=False,
            dtype=None,
        )
        y = _check_y(y, multi_output=True, y_numeric=True, estimator=self)
        self._y_original_shape = y.shape
        y = atleast_2d(y)
        x_df = self.get_dataframe_from_x(x)
        error = self.build_error()
        trainer = self.build_trainer(error)
        self.tree_ = trainer.fit(x_df, y)
        self.is_fitted_ = True
        return self

    def build_error(
        self,
    ):
        errors = {
            "std": tree.DeviationError(),
        }
        if self.criterion not in errors.keys():
            raise ValueError(f"Unknown error function {self.criterion}")
        return errors[self.criterion]

    def predict(self, x):
        y = self.predict_base(x)
        if len(self._y_original_shape) == 1:
            y = y.squeeze()
        return y

    def export_dot(self, filepath):
        tree.export_dot_file(self.tree_, filepath)

    def export_image(self, filepath):
        tree.export_image(self.tree_, filepath)
