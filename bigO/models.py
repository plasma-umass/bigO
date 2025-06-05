from dataclasses import dataclass
from typing import List, Literal, Tuple
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from bigO.outliers import remove_outliers
from statsmodels.stats.multitest import multipletests


system_name = "bigO"


def format_float(value):
    return f"{value:.2f}".rstrip("0").rstrip(".")


@dataclass
class FunctionCanonicalForm:
    """
    Represent any function f(n) = r^n * n^s * (log n)^t * (log log n)^u
    as the coefficients.  All coefficients are floats and >= 0.  Use
    "k" for s to indicate a variable power.
    """

    r: float
    s: float | Literal["k"]
    t: float
    u: float

    def __str__(self) -> str:
        def parameter(x):
            return f"{format_float(x)}" if x != "k" else "k"

        parts = []
        if self.r != 1 and self.r != 0:
            parts.append(f"{parameter(self.r)}**n")
        if self.s != 0:
            if self.s == 0.5:
                parts.append("sqrt(n)")
            elif self.s == 1:
                parts.append("n")
            else:
                parts.append(f"n**{parameter(self.s)}")
        if self.t != 0:
            if self.t == 1:
                parts.append("log(n)")
            else:
                parts.append(f"log(n)**{parameter(self.t)}")
        if self.u != 0:
            if self.u == 1:
                parts.append("log(log(n))")
            else:
                parts.append(f"log(log(n))**{parameter(self.u)}")
        if parts == []:
            return "1"
        return "*".join(parts)

    def as_lambda(self) -> str:
        if self.r == 0 and self.s == 0 and self.t == 0 and self.u == 0:
            return "lambda n, a: np.ones(np.shape(n)) * a"
        elif self.s == "k":
            return f"lambda n, a, b, k: a * ({self}) + b"
        else:
            return f"lambda n, a, b: a * ({self}) + b"

    def as_bigo(self) -> str:
        return f"O({self})"


def log(x):
    x = np.asarray(x)
    result = np.full_like(x, -np.inf, dtype=np.float64)  # Initialize with -inf
    np.log(x, where=(x > 0), out=result)
    return result


def sqrt(x):
    return np.sqrt(x)


class Model:
    def __init__(self, canonical_form: FunctionCanonicalForm):

        self.name = canonical_form.as_bigo()
        self.func = eval(canonical_form.as_lambda())
        self.param_count = self.func.__code__.co_argcount - 1  # Subtract 1 for 'n'
        self.canonical_form = canonical_form

    def __call__(self, n, *args):
        return self.func(n, *args)

    def __str__(self):
        return self.name


class FittedModel:
    def __init__(
        self,
        model: Model,
        params: np.ndarray,
        n: np.ndarray,
        y: np.ndarray,
    ):
        self.model = model
        self.params = params
        self.n = n
        self.y = y

    def predict(self, n: np.ndarray | None = None):
        if n is None:
            n = self.n
        return self.model.func(n, *self.params)

    def residuals(self, n: np.ndarray | None = None, y: np.ndarray | None = None):
        if n is None:
            n = self.n
        if y is None:
            y = self.y
        return y - self.predict(n)

    def mse(self):
        return np.mean(self.residuals() ** 2)

    def aic(self, n: np.ndarray | None = None, y: np.ndarray | None = None):
        k = len(self.params)  # Number of parameters
        residuals = self.residuals(n, y)
        rss = np.sum(residuals**2)  # Residual sum of squares
        if y is None:
            y = self.y
        n_points = len(y)  # Number of data points

        if n_points < 2 or rss < 0:
            return np.inf

        if rss == 0:
            return -np.inf

        aic = 2 * k + n_points * np.log(rss / n_points)
        return aic

    def replace_k(self, name):
        return name.replace("k", f"{self.params[-1]:.2f}")

    def __str__(self):
        name = self.model.name
        return f"{self.replace_k(name)}"

    def __repr__(self):
        return str(self)

    def actual(self):
        return f"{self.params[0]} * {str(self)} + {self.params[1]}"

    def __le__(self, other):
        def val_or_last_param(x):
            return x if x != "k" else self.params[-1]

        cf = self.model.canonical_form
        cfo = other.model.canonical_form

        c_params = [val_or_last_param(x) for x in (cf.r, cf.s, cf.t, cf.u)]
        co_params = [val_or_last_param(x) for x in (cfo.r, cfo.s, cfo.t, cfo.u)]

        for c, co in zip(c_params, co_params):
            if c < co:
                return True
            elif c > co:
                return False

        # coverages to a constant other than 0.
        return True

    def __lt__(self, other):
        def val_or_last_param(x):
            return x if x != "k" else self.params[-1]

        cf = self.model.canonical_form
        cfo = other.model.canonical_form

        c_params = [val_or_last_param(x) for x in (cf.r, cf.s, cf.t, cf.u)]
        co_params = [val_or_last_param(x) for x in (cfo.r, cfo.s, cfo.t, cfo.u)]

        for c, co in zip(c_params, co_params):
            if c < co:
                return True
            elif c > co:
                return False

        # coverages to a constant other than 0.
        return False


# Model definitions
model_constant = Model(FunctionCanonicalForm(0, 0, 0, 0))
model_log_log_n = Model(FunctionCanonicalForm(0, 0, 0, 1))
model_log_n = Model(FunctionCanonicalForm(0, 0, 1, 0))
model_sqrt_n = Model(FunctionCanonicalForm(0, 0.5, 0, 0))
model_linear_n = Model(FunctionCanonicalForm(0, 1, 0, 0))
model_n_log_n = Model(FunctionCanonicalForm(0, 1, 1, 0))
model_n_squared = Model(FunctionCanonicalForm(0, 2, 0, 0))
model_n_cubed = Model(FunctionCanonicalForm(0, 3, 0, 0))
model_n_power_k = Model(FunctionCanonicalForm(0, "k", 0, 0))
model_log_n_squared = Model(FunctionCanonicalForm(0, 0, 2, 0))
model_log_n_cubed = Model(FunctionCanonicalForm(0, 0, 3, 0))
model_n_exp = Model(FunctionCanonicalForm(2, 0, 0, 0))

# List of models
models = [
    model_constant,
    model_log_log_n,
    model_log_n,
    model_log_n_squared,
    model_log_n_cubed,
    model_sqrt_n,
    model_linear_n,
    model_n_log_n,
    model_n_squared,
    model_n_cubed,
    model_n_power_k,
    model_n_exp,
]


def get_model(name: str) -> Model | None:
    if not name.startswith("O(") or not name.endswith(")"):
        raise ValueError(f"Invalid model name: {name}.  Should be of the form O(...)")

    for model in models:
        if model.name == name:
            return model

    # Strip "O(" and ")"
    name = name[2:-1]

    # special case for model_n_power_k
    if name.startswith("n**"):
        k = float(name[3:])
        return Model(FunctionCanonicalForm(0, k, 0, 0))

    raise ValueError(f"Unknown model: {name}")


def fit_model(n, y, model) -> Tuple[FittedModel | None, List[str]]:
    try:
        with warnings.catch_warnings(record=True) as w:
            # Attempt to fit
            (params, _) = curve_fit(
                model.func,
                n,
                y,
                p0=[1.0] * model.param_count,
                maxfev=10000,
                full_output=False,
            )

            # If any relevant warning occurred, return None
            if w:
                messages = [f"fit_model {model.name}: {wm.message}" for wm in w]
                reported = []
                for message in messages:
                    if message not in reported:
                        reported += [message]
                return None, reported

        # Otherwise, build and return the FittedModel
        fitted = FittedModel(model=model, params=params, n=n, y=y)
        return fitted, []
    except Exception as e:
        return None, [f"fit_model {model.name}: {str(e)}"]


def fit_models(n, y) -> Tuple[List[FittedModel], List[str]]:
    """Fit models to data and order by increasing aic. Return list of (model, aic) tuples."""
    results = [fit_model(n, y, model) for model in models]
    fits = [f for f, _ in results if f is not None]
    warnings = [warning for _, warnings in results for warning in warnings]
    return sorted(fits, key=lambda x: x.aic()), warnings


@dataclass
class CheckBoundResult:
    declared_bound_fit: FittedModel
    better_models: pd.DataFrame
    warnings: List[str]


def check_bound(n: np.ndarray, y: np.ndarray, bound: Model) -> CheckBoundResult:
    lengths, y = remove_outliers(n, y)
    bound_model_fit, bound_warnings = fit_model(lengths, y, bound)

    if bound_model_fit is None:
        ws = "\n".join(bound_warnings)
        raise ValueError(
            f"Could not fit bound model {bound.name}.  Possible causes:\n{ws}"
        )

    fits, warnings = fit_models(lengths, y)
    warnings += bound_warnings

    fitted_models = pd.DataFrame(
        [
            (
                fitted,
                fitted.aic(lengths, y),
                pvalue_for_better_fit(fitted, bound_model_fit, lengths, y),
            )
            for fitted in fits
        ],
        columns=["model", "aic", "pvalue"],
    )

    # Multiple tests: Use Holm–Bonferroni to adjust for FDR.
    # Note: Tests are not independent.
    fitted_models = fitted_models.sort_values(by="pvalue", ascending=True)
    reject, p_adjusted, _, _ = multipletests(
        fitted_models["pvalue"], alpha=0.05, method="holm"
    )

    # Take only significantly different
    fitted_models = fitted_models.assign(p_adjusted=p_adjusted)
    fitted_models = fitted_models[reject]

    # Take only those with lower AIC and worse than the specified bound model
    fitted_models = fitted_models[fitted_models["aic"] < bound_model_fit.aic()]
    fitted_models = fitted_models[~(fitted_models["model"] <= bound_model_fit)]

    # sort by AIC
    better_models = fitted_models.sort_values(by="aic", ascending=True)

    return CheckBoundResult(bound_model_fit, better_models, warnings)


@dataclass
class InferBoundResult:
    models: List[FittedModel]
    warnings: List[str]


def infer_bound(n: np.ndarray, y: np.ndarray) -> InferBoundResult:
    lengths, y = remove_outliers(n, y)
    fits, warnings = fit_models(lengths, y)
    return InferBoundResult(fits, warnings)


def pvalue_for_better_fit(
    a: FittedModel, b: FittedModel, n, y, trials=1000  # low trials to start with
) -> float:
    """
    Return pvalue that model 'a' is a better fit than model 'b'.
    Technique from: https://aclanthology.org/D12-1091.pdf
    """
    delta = b.aic() - a.aic()

    bootstrap_indices = np.random.choice(len(n), size=(trials, len(n)), replace=True)

    data = np.column_stack((n, y))
    # a 3D array with shape (trials, n_samples, 2)
    resamples = data[bootstrap_indices]

    # compute the delta for each resample
    s = 0
    for resample in resamples:
        n2 = resample[:, 0]
        y2 = resample[:, 1]
        aic_a2 = a.aic(n2, y2)
        aic_b2 = b.aic(n2, y2)
        bootstrap_delta = aic_b2 - aic_a2
        # See paper for why 2
        if bootstrap_delta > 2 * delta:
            s += 1
    return s / trials


if __name__ == "__main__":

    print(", ".join([str(m) for m in models]))

    import unittest

    class TestFittedModelLE(unittest.TestCase):
        def setUp(self):
            # Dummy data for n and y.
            self.n_data = np.array([2, 3, 4])
            self.y_data = np.array([1, 2, 3])

        def create_fitted_model(self, model, params):
            return FittedModel(model, np.array(params), self.n_data, self.y_data)

        def test_constant_vs_log_log(self):
            # constant: (1, 0, 0, 0) vs log(log(n)): (1, 0, 0, 1)
            fm_const = self.create_fitted_model(model_constant, [5.0])
            fm_log_log = self.create_fitted_model(model_log_log_n, [5.0, 0.0])
            self.assertTrue(fm_const <= fm_log_log)
            self.assertFalse(fm_log_log <= fm_const)

        def test_log_vs_n_log(self):
            # log(n): (1, 0, 1, 0) vs n*log(n): (1, 1, 1, 0)
            fm_log = self.create_fitted_model(model_log_n, [5.0, 0.0])
            fm_n_log = self.create_fitted_model(model_n_log_n, [5.0, 0.0])
            self.assertTrue(fm_log <= fm_n_log)
            self.assertFalse(fm_n_log <= fm_log)

        def test_sqrt_vs_linear(self):
            # sqrt(n): (1, 0.5, 0, 0) vs n: (1, 1, 0, 0)
            fm_sqrt = self.create_fitted_model(model_sqrt_n, [5.0, 0.0])
            fm_linear = self.create_fitted_model(model_linear_n, [5.0, 0.0])
            self.assertTrue(fm_sqrt <= fm_linear)
            self.assertFalse(fm_linear <= fm_sqrt)

        def test_n_squared_vs_n_cubed(self):
            # n**2: (1, 2, 0, 0) vs n**3: (1, 3, 0, 0)
            fm_n_squared = self.create_fitted_model(model_n_squared, [5.0, 0.0])
            fm_n_cubed = self.create_fitted_model(model_n_cubed, [5.0, 0.0])
            self.assertTrue(fm_n_squared <= fm_n_cubed)
            self.assertFalse(fm_n_cubed <= fm_n_squared)

        def test_n_power_k_vs_n_squared_with_k_less(self):
            # n**k: (1, None, 0, 0) where None is substituted with k.
            # Use k = 1.5 which is less than 2 (from n**2)
            fm_n_power_k = self.create_fitted_model(model_n_power_k, [5.0, 0.0, 1.5])
            fm_n_squared = self.create_fitted_model(model_n_squared, [5.0, 0.0])
            # For fm_n_power_k, s becomes 1.5 vs 2.
            self.assertTrue(fm_n_power_k <= fm_n_squared)
            # In reverse, fm_n_squared <= fm_n_power_k compares 2 (its s) with
            # the substituted value from fm_n_power_k using fm_n_squared’s params[-1] (0.0), so False.
            self.assertFalse(fm_n_squared <= fm_n_power_k)

        def test_n_power_k_vs_n_squared_with_k_more(self):
            # Use k = 2.5 which is greater than 2.
            fm_n_power_k = self.create_fitted_model(model_n_power_k, [5.0, 0.0, 2.5])
            fm_n_squared = self.create_fitted_model(model_n_squared, [5.0, 0.0])
            self.assertFalse(fm_n_power_k <= fm_n_squared)

        def test_n_exp_vs_linear_more(self):
            # Use k = 1.2 (> 1)
            fm_n_exp = self.create_fitted_model(model_n_exp, [5.0, 0.0, 1.2])
            fm_linear = self.create_fitted_model(model_linear_n, [5.0, 0.0])
            self.assertFalse(fm_n_exp <= fm_linear)

        def test_log_n_squared_vs_log_n_cubed(self):
            # (log(n))**2: (1,0,2,0) vs (log(n))**3: (1,0,3,0)
            fm_log_n_squared = self.create_fitted_model(model_log_n_squared, [5.0, 0.0])
            fm_log_n_cubed = self.create_fitted_model(model_log_n_cubed, [5.0, 0.0])
            self.assertTrue(fm_log_n_squared <= fm_log_n_cubed)
            self.assertFalse(fm_log_n_cubed <= fm_log_n_squared)

        def test_n_log_n_vs_n_squared(self):
            # n*log(n): (1,1,1,0) vs n**2: (1,2,0,0)
            fm_n_log_n = self.create_fitted_model(model_n_log_n, [5.0, 0.0])
            fm_n_squared = self.create_fitted_model(model_n_squared, [5.0, 0.0])
            self.assertTrue(fm_n_log_n <= fm_n_squared)
            self.assertFalse(fm_n_squared <= fm_n_log_n)

    unittest.main()
