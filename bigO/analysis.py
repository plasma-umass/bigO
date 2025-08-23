import abc
from dataclasses import dataclass
from typing import List

from matplotlib import pyplot as plt
from matplotlib.figure import Figure, SubFigure
import pandas as pd
from bigO.abtest import non_parametric_fit, segmented_permutation_test
import bigO.models as models

from bigO.output import timer
import numpy as np
import seaborn as sns


@dataclass
class FunctionData:
    function_name: str
    file_name: str
    lengths: np.ndarray
    times: np.ndarray
    mems: np.ndarray


@dataclass
class Result:
    success: bool
    message: str
    details: str
    warnings: List[str]


class Analysis(abc.ABC):
    @abc.abstractmethod
    def title(self) -> str:
        pass

    @abc.abstractmethod
    def run(self) -> Result:
        pass

    @abc.abstractmethod
    def plot(self, fig: Figure | SubFigure | None = None):
        pass


class InferPerformance(Analysis):
    def __init__(self, function_data: FunctionData):
        self.function_data = function_data

    def title(self) -> str:
        return f"Infer performance for {self.function_data.function_name}"

    def run(self):
        with timer(self.title()):
            self.fitted_times = models.infer_bound(
                self.function_data.lengths, self.function_data.times
            )
            self.fitted_mems = models.infer_bound(
                self.function_data.lengths, self.function_data.mems
            )

        best_time_model = (
            self.fitted_times.models[0] if self.fitted_times.models else None
        )
        best_mem_model = self.fitted_mems.models[0] if self.fitted_mems.models else None
        return Result(
            success=True,
            message=(f"Time is {best_time_model}.  Memory is: {best_mem_model}."),
            details=(
                f"Inferred Bounds for {self.function_data.function_name}\n"
                f"Best time model: {best_time_model}\nBest memory model: {best_mem_model}"
            ),
            warnings=self.fitted_times.warnings + self.fitted_mems.warnings,
        )

    def plot_fits(self, ax, models: List[models.FittedModel], color, ylabel, title):
        best_fit = models[0]
        sns.scatterplot(
            x=best_fit.n,
            y=best_fit.y,
            ax=ax,
            color=color,
            alpha=0.7,
            label="Data (Outliers Removed)",
        )
        sns.lineplot(
            x=best_fit.n,
            y=best_fit.predict(best_fit.n),
            ax=ax,
            color=color,
            linewidth=2,
            label=f"Best fit: {best_fit}",
        )

        ax.set_xlabel("Input Size (n)")
        ax.set_ylabel(ylabel)

        ax.set_title(title, fontsize=12)
        ax.legend()

    def plot(self, fig: Figure | SubFigure | None = None):
        if fig is None:
            fig = plt.figure(constrained_layout=True, figsize=(12, 4))
        time_ax, mem_ax, extra = fig.subplots(1, 3)
        extra.axis("off")
        self.plot_fits(
            time_ax,
            self.fitted_times.models,
            color="C0",
            ylabel="Time (s)",
            title=f"{self.function_data.function_name} inferred time: {self.fitted_times.models[0]}",
        )
        self.plot_fits(
            mem_ax,
            self.fitted_mems.models,
            color="C1",
            ylabel="Memory",
            title=f"{self.function_data.function_name} inferred memory: {self.fitted_mems.models[0]}",
        )


class CheckBounds(Analysis):
    def __init__(
        self,
        function_data: FunctionData,
        time_bound: str | None = None,
        mem_bound: str | None = None,
    ):
        super().__init__()
        self.function_data = function_data
        self.time_bound = models.get_model(time_bound) if time_bound else None
        self.mem_bound = models.get_model(mem_bound) if mem_bound else None

    def title(self) -> str:
        return f"Check bounds for {self.function_data.function_name}."

    def run(self) -> Result:
        with timer(self.title()):
            if self.time_bound:
                with timer("Check time models"):
                    self.time_check = models.check_bound(
                        self.function_data.lengths,
                        self.function_data.times,
                        self.time_bound,
                    )
            else:
                self.time_check = None
            if self.mem_bound:
                with timer("Check memory models"):
                    self.mem_check = models.check_bound(
                        self.function_data.lengths,
                        self.function_data.mems,
                        self.mem_bound,
                    )
            else:
                self.mem_check = None

        success = True
        details = []
        message = []
        if self.time_check and len(self.time_check.better_models) > 0:
            success = success and len(self.time_check.better_models) == 0
            message += [
                f"Time bound is supposed to be {self.time_bound} but is actually {self.time_check.better_models['model'].iloc[0]}.",
            ]
            details += [
                f"Declared Time Bound for {self.function_data.function_name} is {self.time_bound}, ",
                "but these models with worse performance better fit the data: ",
                str(self.time_check.better_models.to_string(index=False)),
            ]
        if self.mem_check and len(self.mem_check.better_models) > 0:
            success = success and len(self.mem_check.better_models) == 0
            message = [
                f"Memory bound is supposed to be {self.mem_bound} but is actually {self.mem_check.better_models['model'].iloc[0]}.",
            ]
            details += [
                f"Declared Memory Bound for {self.function_data.function_name}: {self.mem_bound}, ",
                "but these models with worse performance better fit the data: ",
                str(
                    self.mem_check.better_models[["model", "pvalue"]].to_string(
                        index=False
                    )
                ),
            ]
        warnings = (self.time_check.warnings if self.time_check else []) + (
            self.mem_check.warnings if self.mem_check else []
        )
        return Result(
            success=success,
            message="\n".join(message),
            details="\n".join(details),
            warnings=warnings,
        )

    def plot_fits(
        self, ax, check_result: models.CheckBoundResult, color, ylabel, title
    ):
        declared_fit = check_result.declared_bound_fit
        sns.scatterplot(
            x=declared_fit.n,
            y=declared_fit.y,
            ax=ax,
            color=color,
            alpha=0.7,
            label="Data (Outliers Removed)",
        )
        sns.lineplot(
            x=declared_fit.n,
            y=declared_fit.predict(declared_fit.n),
            ax=ax,
            color=color,
            label=f"Declared: {declared_fit}",
        )
        best_fit = (
            check_result.better_models.iloc[0]
            if len(check_result.better_models) > 0
            else None
        )
        if best_fit is not None:
            best_model = best_fit["model"]
            pvalue = best_fit["pvalue"]
            sns.lineplot(
                x=best_model.n,
                y=best_model.predict(best_model.n),
                ax=ax,
                color="red",
                label=f"Best fit: {best_model} (p={pvalue:.3f})",
                linewidth=2,
            )

        ax.set_xlabel("Input Size (n)")
        ax.set_ylabel(ylabel)

        ax.set_title(title, fontsize=12)
        ax.legend()

    def plot(self, fig: Figure | SubFigure | None = None):
        if fig is None:
            fig = plt.figure(constrained_layout=True, figsize=(12, 4))
        time_ax, mem_ax, extra = fig.subplots(1, 3)
        extra.axis("off")
        if self.time_check:
            self.plot_fits(
                time_ax,
                self.time_check,
                color="C0",
                ylabel="Time (s)",
                title=f"{self.function_data.function_name} time bound",
            )
        else:
            time_ax.axis("off")
        if self.mem_check:
            self.plot_fits(
                mem_ax,
                self.mem_check,
                color="C1",
                ylabel="Memory",
                title=f"{self.function_data.function_name}: mem bound",
            )
        else:
            mem_ax.axis("off")


class CheckLimits(Analysis):
    def __init__(
        self,
        function_data: FunctionData,
        time_limit: float | None = None,
        mem_limit: float | None = None,
        length_limit: int | None = None,
    ):
        super().__init__()
        self.function_data = function_data
        self.time_limit = time_limit
        self.mem_limit = mem_limit
        self.length_limit = length_limit

    def title(self) -> str:
        return f"Check limits for {self.function_data.function_name}. "

    def run(self) -> Result:

        with timer(self.title()):
            success = True
            message = []
            details = []
            max_time = None
            max_mem = None
            max_length = None
            if self.time_limit:
                max_time = self.function_data.times.max()
                if max_time > self.time_limit:
                    message += [
                        f"Time of {max_time} exceeds limit of {self.time_limit}."
                    ]
                    details += [
                        f"Declared time limit for {self.function_data.function_name} is {self.time_limit}, ",
                        f"but the largest observed time is {max_time}.",
                    ]
                    success = False
            if self.mem_limit:
                max_mem = self.function_data.mems.max()
                if max_mem > self.mem_limit:
                    message += [
                        f"Memory of {max_mem} exceeds limit of {self.mem_limit}."
                    ]
                    details += [
                        f"Declared memory limit for {self.function_data.function_name} is {self.mem_limit}, ",
                        f"but the largest observed memory is {max_mem}.",
                    ]
                    success = False
            if self.length_limit:
                max_length = self.function_data.lengths.max()
                if max_length > self.length_limit:
                    message += [
                        f"Length of {max_length} exceeds limit of {self.length_limit}."
                    ]
                    details += [
                        f"Declared length limit for {self.function_data.function_name} is {self.length_limit}, ",
                        f"but the largest observed length is {max_length}.",
                    ]
                    success = False

            warnings = []
            return Result(
                success=success,
                message="\n".join(message),
                details="\n".join(details),
                warnings=warnings,
            )

    def plot(self, fig: Figure | SubFigure | None = None):
        if fig is None:
            fig = plt.figure(constrained_layout=True, figsize=(12, 4))
        time_ax, mem_ax, len_ax = fig.subplots(1, 3)
        if self.time_limit:
            sns.histplot(
                x=self.function_data.times,
                color="C0",
                ax=time_ax,
                label="Data",
            )
            time_ax.title.set_text(f"{self.function_data.function_name}")
            time_ax.set_xlabel("Time (s)")
            time_ax.axvline(
                self.time_limit, color="red", label=f"Declared limit: {self.time_limit}"
            )
            time_ax.legend()
        else:
            time_ax.axis("off")

        if self.mem_limit:
            sns.histplot(
                self.function_data.mems,
                color="C1",
                ax=mem_ax,
                label="Data",
            )
            mem_ax.title.set_text(f"{self.function_data.function_name}")
            mem_ax.set_xlabel("Memory")
            mem_ax.axvline(
                self.mem_limit, color="red", label=f"Declared limit: {self.mem_limit}"
            )
            mem_ax.legend()
        else:
            mem_ax.axis("off")

        if self.length_limit:
            sns.histplot(
                self.function_data.lengths,
                color="C2",
                ax=len_ax,
                label="Data",
            )
            len_ax.title.set_text(f"{self.function_data.function_name}")
            len_ax.set_xlabel("Length")
            len_ax.axvline(
                self.length_limit,
                color="red",
                label=f"Declared limit: {self.length_limit}",
            )
            len_ax.legend()
        else:
            len_ax.axis("off")


class ABTest(Analysis):
    def __init__(self, a: FunctionData, b: FunctionData, metric: str):
        self.a = a
        self.b = b
        self.metric = metric

    def title(self) -> str:
        return f"AB Test: {self.a.function_name} vs. {self.b.function_name}"

    def _summary(self) -> str:
        messages = []
        for full_result in self.ab_results.segments:
            result = full_result.result
            if full_result.null_rejected:
                messages += [
                    f"{result.faster} is significantly faster for n={result.n_common.min()} to {result.n_common.max()}"
                ]
        return "\n".join(messages)

    def run(self):
        with timer(self.title()):
            combined_labels = np.concatenate(
                [["A"] * len(self.a.lengths), ["B"] * len(self.b.lengths)]
            )
            combined_lengths = np.concatenate([self.a.lengths, self.b.lengths])
            combined_T = np.concatenate(
                [
                    self.a.times if self.metric == "time" else self.a.mems,
                    self.b.times if self.metric == "time" else self.b.mems,
                ]
            )
            self.combined_df = pd.DataFrame(
                {
                    "label": combined_labels,
                    "n": combined_lengths,
                    "T": combined_T,
                }
            )
            self.ab_results = segmented_permutation_test(
                self.combined_df,
                num_permutations=1000,
                num_points=100,
            )

        details = [
            f"AB Test for {self.metric}: A is {self.a.function_name}; B is {self.b.function_name}"
        ]
        for report in self.ab_results.segments:
            details += [f"  {x}" for x in str(report).splitlines()]

        return Result(
            success=True,
            message=self._summary(),
            details="\n".join(details),
            warnings=self.ab_results.warnings,
        )

    def plot(self, fig: Figure | SubFigure | None = None):
        if fig is None:
            fig = plt.figure(constrained_layout=True, figsize=(12, 4))

        # num_figs = min(1 + len(self.ab_results.segments), 4)
        axes = fig.subplots(1, 3)
        axes[1].axis("off")
        axes[2].axis("off")

        # ----------------------------------------
        # 4.1. Scatter Plots with Curves (First Visualization)
        # ----------------------------------------

        df_A = self.combined_df[self.combined_df.label == "A"]
        df_B = self.combined_df[self.combined_df.label == "B"]

        n_A, T_A = non_parametric_fit(df_A)
        n_B, T_B = non_parametric_fit(df_B)

        sns.scatterplot(
            data=df_A,
            x="n",
            y="T",
            label=f"A: {self.a.function_name}",
            alpha=0.7,
            ax=axes[0],
        )

        sns.scatterplot(
            data=df_B,
            x="n",
            y="T",
            label=f"B: {self.b.function_name}",
            alpha=0.7,
            ax=axes[0],
        )

        sns.lineplot(
            x=n_A,
            y=T_A,
            color="C0",
            ax=axes[0],
        )

        sns.lineplot(
            x=n_B,
            y=T_B,
            color="C1",
            ax=axes[0],
        )

        for index, full_result in enumerate(self.ab_results.segments):
            result = full_result.result
            if index > 0:
                axes[0].axvline(
                    result.n_common.min(),
                    color="black",
                    linestyle="--",
                    linewidth=1,
                )

            if index == 0:
                x_min = 0
            else:
                x_min = result.n_common.min()

            if index == len(self.ab_results.segments) - 1:
                x_max = result.n_common.max()
            else:
                x_max = self.ab_results.segments[index + 1].result.n_common.min()

            if full_result.null_rejected:
                axes[0].axvspan(
                    x_min,
                    x_max,
                    color="C0" if result.faster == "A" else "C1",
                    alpha=0.1,
                )

            # show the pvalue as text at the centered at midpoint near the top
            x_text = (x_min + x_max) / 2
            y_text = 0.9 * axes[0].get_ylim()[1]
            axes[0].text(
                x_text,
                y_text,
                f"p={result.p_value:.3f}\nadj_p={full_result.adjusted_pvalue:.3f}",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=10,
                color="black",
            )

        axes[0].set_xlabel("Input Size (n)")
        axes[0].set_ylabel(f"{self.metric.title()} (T)")
        axes[0].set_title(
            f"{self.a.function_name} vs. {self.b.function_name}: {self.metric.title()}"
        )
        axes[0].legend()
        axes[0].grid(True)
