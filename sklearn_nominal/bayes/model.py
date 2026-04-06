import operator
from functools import reduce
import abc
from abc import ABC
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn_nominal.backend import Input, InputSample
from sklearn_nominal.backend.core import Model

import matplotlib.pyplot as plt
from scipy.stats import norm


class Variable(ABC):
    @abc.abstractmethod
    def predict(self, x: pd.Series) -> np.ndarray:
        pass

    @abc.abstractmethod
    def complexity(self) -> int:
        pass


atol = 1e-02


class GaussianVariable(Variable):
    def __init__(self, mu: float, std: float, smoothing: float = 0) -> None:
        self.mu = mu
        self.std = std
        self.normal = norm(mu, std + smoothing)

    def predict(self, x: pd.Series) -> np.ndarray:
        result: np.ndarray = self.normal.pdf(x.values)
        result[np.isnan(x.values)] = 1
        return result

    def __repr__(self) -> str:
        return f"N({self.mu:.4g},{self.std:.4g})"

    def complexity(self):
        return 1

    def __eq__(self, x):
        if not isinstance(x, GaussianVariable):
            return False
        return np.allclose(self.mu, x.mu, atol=atol) and np.allclose(self.std, x.std, atol=atol)


def dict_allclose(x: dict[Any, float], y: dict[Any, float]):
    def keys(x: dict):
        return sorted(list(x.keys()))

    def values(x: dict):
        return np.array([x[k] for k in keys(x)])

    return len(x) == len(y) and keys(x) == keys(y) and np.allclose(values(x), values(y), atol=atol)


class CategoricalVariable(Variable):
    def __init__(self, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities

    def p(self, x: str, default: float = 1.0) -> float:
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

    def __eq__(self, x):
        if not isinstance(x, CategoricalVariable):
            return False
        return dict_allclose(self.probabilities, x.probabilities)


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
        variables = "\n".join([f"    {k:{max_name_length}} ~ {v}" for k, v in self.variables.items()])
        return f"{variables}"

    def complexity(self) -> int:
        return max([v.complexity() for v in self.variables.values()])

    def __eq__(self, x):
        if not isinstance(x, NaiveBayesSingleClass):
            return False

        return self.variables == x.variables


class NaiveBayes(Model):
    def __init__(
        self,
        class_models: list[NaiveBayesSingleClass],
        class_probabilities: np.ndarray,
    ):
        self.models = class_models
        self.pi = class_probabilities

    def predict_sample(self, x: InputSample) -> int:
        df = pd.DataFrame([x])
        y = self.predict(df)
        return y[0, :]

    def predict(self, x: Input):
        n = len(x)
        n_classes = len(self.models)
        results = np.zeros((n, n_classes))
        for c in range(n_classes):
            p_x = self.models[c].predict(x)
            p_class = self.pi[c]
            results[:, c] = p_x * p_class

        results /= results.sum(axis=1, keepdims=True)

        return results

    def explain(self, x: Input, class_names: list[str] | None = None) -> pd.DataFrame:
        n = len(x)
        if class_names is None:
            class_names = [f"class_{i}" for i in range(len(self.models))]

        n_classes = len(self.models)
        explanation = [{"sample": i} for i in range(n)]
        results = np.zeros((n, n_classes))
        for c in range(n_classes):
            p_x = 1
            for name, var in self.models[c].variables.items():
                p_var = var.predict(x[name])
                for i in range(n):
                    explanation[i][f"P({name}|{class_names[c]})"] = p_var[i]
                p_x *= p_var
            p_class = self.pi[c]
            for i in range(n):
                explanation[i][f"P({class_names[c]})"] = p_class
            results[:, c] = p_x * p_class

        results /= results.sum(axis=1, keepdims=True)
        for i in range(n):
            for c in range(n_classes):
                explanation[i][f"p({class_names[c]}|x)"] = results[i, c]
        # explanation = reduce(operator.ior, explanation, {})
        explanation = pd.DataFrame(explanation)
        return explanation

    def pretty_print(self, class_names: list[str] = None) -> str:
        n_classes = len(self.models)

        def class_description(i: int):
            name = f"{i}" if class_names is None else class_names[i]
            p_c = self.pi[i]
            return f"Class {name} (p={p_c:.3g}):\n{self.models[i].pretty_print()}"

        class_descriptions = [class_description(i) for i in range(n_classes)]
        class_descriptions = "\n".join(class_descriptions)
        return f"{NaiveBayes.__name__}(classes={n_classes})\n{class_descriptions}"

    def plot_distributions(
        self, class_names: list[str] | None = None, feature_names: list[str] | None = None, n_cols=4
    ):
        models = self.models
        if class_names is None:
            class_names = [f"{i}" for i in range(len(models))]
        if feature_names is None:
            feature_names = [v for v, d in models[0].variables.items()]
        n_rows = int(np.ceil(len(feature_names) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = np.array(axes).reshape(-1)
        for ax, feat in zip(axes, feature_names):
            var = models[0].variables[feat]
            if isinstance(var, GaussianVariable):
                for cls, m in zip(class_names, models):
                    dist = m.variables[feat]
                    mu, std = dist.mu, dist.std
                    x = np.linspace(mu - 4 * std, mu + 4 * std, 200)
                    y = dist.predict(pd.Series(x, name=feat))
                    ax.plot(x, y, label=cls)
            elif isinstance(var, CategoricalVariable):
                # plot heatmap of probabilities for each class vs value for variable feat
                probabilities = np.zeros((len(class_names), len(var.probabilities)))
                values = list(var.probabilities.keys())
                for i, cls in enumerate(class_names):
                    dist = models[i].variables[feat]
                    probabilities[i, :] = [dist.probabilities[v] for v in values]
                ax.imshow(probabilities, cmap="viridis", aspect="auto")
                ax.set_xticks(np.arange(len(values)))
                ax.set_yticks(np.arange(len(class_names)))
                ax.set_xticklabels(values)
                ax.set_yticklabels(class_names)
                ax.set_xlabel("Value")
                ax.set_ylabel("Class")
                ax.set_title(feat)
                ## add colorbar
                cbar = fig.colorbar(ax.imshow(probabilities, cmap="viridis", aspect="auto"), ax=ax)
                cbar.set_label("Probability")

        for ax in axes[len(feature_names) :]:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    def complexity(self) -> int:
        return max([m.complexity() for m in self.models])

    def output_size(self) -> int:
        return len(self.models)

    def __eq__(self, x):
        if not isinstance(x, NaiveBayes):
            return False
        return (
            self.pi.shape == x.pi.shape
            and np.allclose(self.pi, x.pi, atol=atol)
            and len(self.models) == len(x.models)
            and all([a == b for a, b in zip(self.models, x.models)])
        )
