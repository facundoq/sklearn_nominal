from abc import ABC
import abc

from scipy.stats import norm
from sklearnmodels.backend import Input, InputSample
from sklearnmodels.backend.core import Model


import numpy as np
import pandas as pd


class Variable(ABC):

    @abc.abstractmethod
    def predict(x: pd.Series) -> np.ndarray:
        pass

    @abc.abstractmethod
    def complexity(self) -> int:
        pass


class GaussianVariable(Variable):

    def __init__(self, mu: float, std: float, smoothing: float = 0) -> None:
        self.mu = mu
        self.std = std
        self.normal = norm(mu, std + smoothing)

    def predict(self, x: pd.Series) -> np.ndarray:
        result = self.normal.pdf(x.values)
        result[np.isnan(x.values)] = 1
        return result

    def __repr__(self) -> str:
        return f"N({self.mu:.4g},{self.std:.4g})"

    def complexity(self):
        return 1


class CategoricalVariable(Variable):

    def __init__(self, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities

    def p(self, x: str, default=1.0) -> float:
        if x in self.probabilities:
            return self.probabilities[x]
        else:
            return default

    def predict(self, x: pd.Series) -> np.ndarray:
        return np.array(list(map(self.p, x.values)))

    def __repr__(self) -> str:
        variables = ", ".join([f"{k}={v:.4g}" for k, v in self.probabilities.items()])
        return f"C({variables})"

    def complexity(self):
        return len(self.probabilities)


class NaiveBayesSingleClass:

    def __init__(self, variables: dict[str, Variable]):
        self.variables = variables

    def predict(self, x: pd.DataFrame):
        p = 1
        for name, var in self.variables.items():
            value = x[name]
            pi = var.predict(value)
            p *= pi
        return p

    def pretty_print(self) -> str:
        max_name_length = max(map(len, self.variables.keys())) + 2
        variables = "\n".join(
            [f"    {k:{max_name_length}} ~ {v}" for k, v in self.variables.items()]
        )
        return f"{variables}"

    def complexity(self) -> int:
        return max([v.complexity() for v in self.variables.values()])


class NaiveBayes(Model):

    def __init__(
        self,
        class_names: list[str],
        class_models: list[NaiveBayesSingleClass],
        class_probabilities: CategoricalVariable,
    ):
        self.class_names = class_names
        self.class_models = class_models
        self.class_probabilities = class_probabilities

    def predict_sample(self, x: InputSample) -> int:
        df = pd.DataFrame([x])
        y = self.predict(df)
        return df.iloc[0, :]

    def predict(self, x: Input):
        n = len(x)
        classes = self.class_names
        results = np.zeros((n, len(classes)))
        for c in range(len(classes)):
            p_x = self.class_models[c].predict(x)
            name = self.class_names[c]
            p_class = self.class_probabilities.predict(pd.Series([name]))[0]
            results[:, c] = p_x * p_class
        #     if debug:
        #         name = self.class_names[c]
        #         details.append([f"Clase {name}","","","",""])
        #         for i in range(n):
        #             details.append([f"Ejemplo {i}", f"P(c={name})={p_class:.2f}", f"p(x \| c={name})={p_x[i]:.2e}",f"p(c={name} \| x)={results[i,c]:.2e}"])
        # if debug:
        #     return results, details
        # else:
        return results

    # def predict_classes(self,x:pd.DataFrame,debug=False):
    #     prob = self.predict(x,debug=debug)
    #     classes = prob.argmax(axis=1)
    #     pred = np.array([self.class_names[i] for i in classes])
    #     return pred

    def pretty_print(self, class_names: list[str] = None) -> str:

        def class_description(i: int, name: str):
            x = pd.Series([name])
            p_c = self.class_probabilities.predict(x)[0]
            return f"Class {name} (p={p_c:.3g}):\n{self.class_models[i].pretty_print()}"

        class_descriptions = [
            class_description(i, name) for i, name in enumerate(self.class_names)
        ]
        class_descriptions = "\n".join(class_descriptions)
        return f"{NaiveBayes.__name__}(classes={len(self.class_names)})\n{class_descriptions}"

    def table(self) -> str:
        rows = []
        for c, cm in enumerate(self.class_models):
            name = self.class_names[c]
            rows.append(
                [
                    f"Class {name}",
                    f"P(c={name}) = {self.class_probabilities.predict([name])[0]}",
                ]
            )
            for k, v in cm.variables.items():
                rows.append([f"{k}", f"{v}"])

        return rows

    def complexity(self) -> int:
        return max([m.complexity() for m in self.class_models])

    def output_size(self) -> int:
        return len(self.class_names)
