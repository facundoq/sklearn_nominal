from sklearnmodels.backend.core import Dataset
from sklearnmodels.rules.model import RuleModel
from sklearnmodels.shared.target_error import TargetError


class ZeroR:

    def __init__(self, error_function: TargetError):
        self.error_function = error_function

    def fit(self, d: Dataset):
        return RuleModel([], default_prediction=self.error_function.prediction(d))
