from sklearnmodels.backend.core import Model


import numpy as np
import pandas as pd


class Variable(ABC):
    @abstractmethod
    def predict(x, debug=False):
        pass


class GaussianVariable(Variable):

    def __init__(self, mu: float, std: float, smoothing: float = 0) -> None:
        self.mu = mu
        self.std = std
        self.norm = norm(mu, std + smoothing)

    def predict(self, x: pd.Series):
        return self.norm.pdf(x.values)

    def __repr__(self) -> str:
        return f"N~({self.mu},{self.std})"


class CategoricalVariable(Variable):

    def __init__(self, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities

    def predict(self, x: pd.Series):
        return np.array([self.probabilities[v] for v in x.values])

    def __repr__(self) -> str:
        return f"C~({self.probabilities})"


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
        variables = "\n\n".join([f"{k}: {v}" for k, v in self.variables.items()])
        return f"Distributions:\n\n {variables}"


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

    def predict(self, x: pd.DataFrame, callback=None):

        n = len(x)
        classes = self.class_names
        results = np.zeros((n, len(classes)))
        for c in range(len(classes)):
            p_x = self.class_models[c].predict(x)
            name = self.class_names[c]
            p_class = self.class_probabilities.predict([name])[0]
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

    def pretty_print(self) -> str:

        def class_description(i: int):
            name = self.class_names[i]
            p_c = self.class_probabilities.predict([name])[0]
            return f"Class {name}: (P(c={name})={p_c}):\n\n {self.class_models[i]}"

        class_descriptions = [class_description(i) for i in range(self.n_classes)]
        class_descriptions = "\n\n".join(class_descriptions)
        return f"{NaiveBayes.__name__}\n\n {class_descriptions}"

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
