from dataclasses import dataclass
from typing import Any, List, Literal, Tuple
import warnings
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

from bigO.output import log
from bigO.outliers import remove_outliers_df


def loess_fit(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits a LOESS curve to the provided DataFrame.

    Parameters:
    - df: DataFrame with 'n' and 'T' columns.

    Returns:
    - Sorted n values and the corresponding smoothed T values.
    """
    sorted_df = df.sort_values("n")
    smooth = lowess(sorted_df["T"], sorted_df["n"], frac=0.25, return_sorted=True)
    return smooth[:, 0], smooth[:, 1]


from statsmodels.nonparametric.kernel_regression import KernelReg


def fit_kernel_regression(df: pd.DataFrame):
    """
    Fits a kernel regression model to (n, T) data using statsmodels.

    Returns:
      (x_sorted, y_fitted)
        x_sorted: Sorted 'n' values (1D array).
        y_fitted: Fitted T values from the kernel regression.
    """
    df = df.sort_values("n")
    x = df["n"].values
    y = df["T"].values

    kr = KernelReg(endog=[y], exog=[x], var_type="c", bw="cv_ls")

    # Fit/predict on the same x (you could predict on a finer grid if you like)
    y_fit, y_std = kr.fit([x])  # y_std is the std error of predictions

    return x, y_fit


import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from typing import Tuple


def fit_spline(
    df: pd.DataFrame, s: float | None = None, k: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits a smoothing spline to the provided DataFrame.

    Parameters:
    - df: DataFrame with 'n' and 'T' columns.
    - s: Smoothing factor. If None, the spline will interpolate the data exactly.
         Increasing s leads to a smoother curve.
    - k: Degree of the spline. (Typically 2 for quadratic or 3 for cubic splines.)

    Returns:
    - x_new: A sorted array of 'n' values (on a dense grid).
    - y_new: The corresponding smoothed 'T' values.
    """
    # Ensure the data is sorted
    sorted_df = df.sort_values("n")
    x = sorted_df["n"].values
    y = sorted_df["T"].values

    if len(df) <= 3:
        k = 2

    # Create the smoothing spline; adjust s for more/less smoothing
    spline = UnivariateSpline(x, y, s=10000, k=k)

    # Evaluate the spline on a dense grid for a smooth curve
    x_new = np.linspace(x.min(), x.max(), len(x) * 5)  # type: ignore
    y_new = spline(x_new)

    assert isinstance(x_new, np.ndarray)
    assert isinstance(y_new, np.ndarray)

    return x_new, y_new


def non_parametric_fit(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    return fit_spline(df)


def interpolate_common_n(
    n_A: np.ndarray, T_A: np.ndarray, n_B: np.ndarray, T_B: np.ndarray, num_points=200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a common range of n and interpolates T_A and T_B.

    Parameters:
    - n_A, T_A: Arrays of n and T for group A.
    - n_B, T_B: Arrays of n and T for group B.
    - num_points: Number of points in the common n range.

    Returns:
    - n_common: Common n values.
    - T_A_interp: Interpolated T_A values at n_common.
    - T_B_interp: Interpolated T_B values at n_common.
    """
    n_min = min(n_A.min(), n_B.min())
    n_max = max(n_A.max(), n_B.max())
    n_common = np.linspace(n_min, n_max, num=num_points)

    T_A_interp = np.interp(n_common, n_A, T_A)  # type: ignore
    T_B_interp = np.interp(n_common, n_B, T_B)  # type: ignore

    return n_common, T_A_interp, T_B_interp


def compute_area_between_curves(
    n_common: np.ndarray, T_A: np.ndarray, T_B: np.ndarray
) -> float:
    """
    Computes the signed area between two curves using the trapezoidal rule.

    Parameters:
    - T_A, T_B: Arrays of interpolated T values for groups A and B.

    Returns:
    - Signed area between the two curves.
    """
    return np.trapezoid(T_A - T_B, n_common)  # type: ignore


# ----------------------------------------
# Define Permutation Test Function
# ----------------------------------------


@dataclass
class ABTestResult:
    observed_stat: float
    perm_stats: np.ndarray
    n_common: np.ndarray
    T_A_interp: np.ndarray
    T_B_interp: np.ndarray

    faster: Literal["A", "B"]
    p_value: np.floating[Any]

    def __str__(self):
        return f"{self.faster} is faster for {self.n_common.min():.3f} <= n <= {self.n_common.max():.3f} (p-value={self.p_value:.3f})"


def permutation_test(
    df: pd.DataFrame,
    num_permutations=1000,
    num_points=50,
    seed=None,
) -> ABTestResult:
    """
    Performs a permutation test to compare two groups.

    Parameters:
    - df: Combined DataFrame with 'n', 'T', and 'label' columns.
    - num_permutations: Number of permutations to perform.
    - num_points: Number of points in the common n range.
    - seed: Random seed for reproducibility.

    Returns:
    - observed_stat: Observed test statistic from original labels.
    - perm_stats: Array of test statistics from permutations.
    - n_common: Common n values.
    - T_A_interp: Interpolated T_A values at n_common.
    - T_B_interp: Interpolated T_B values at n_common.
    - n_interval: Confidence interval for n_common.
    - faster: Faster group ('A' or 'B').
    - p_value: P-value for the observed test statistic.

    """
    if seed is not None:
        np.random.seed(seed)

    # Smooth curves for original groups
    group_A = df[df["label"] == "A"]
    group_B = df[df["label"] == "B"]

    n_A, T_A = non_parametric_fit(group_A)
    n_B, T_B = non_parametric_fit(group_B)

    common_x = np.shape(n_A) == np.shape(n_B) and (n_A == n_B).all()

    if common_x:
        n_common, T_A_interp, T_B_interp = n_A, T_A, T_B
    else:
        n_common, T_A_interp, T_B_interp = interpolate_common_n(
            n_A, T_A, n_B, T_B, num_points=num_points
        )

    observed_stat = compute_area_between_curves(n_common, T_A_interp, T_B_interp)

    perm_stats = np.zeros(num_permutations)

    for i in range(num_permutations):
        # Shuffle the labels
        shuffled_labels = np.random.permutation(df["label"])
        df["shuffled_label"] = shuffled_labels

        # Split into permuted groups
        perm_group_A = df[df["shuffled_label"] == "A"]
        perm_group_B = df[df["shuffled_label"] == "B"]

        # Fit smoothed curves to permuted groups
        try:
            n_perm_A, T_perm_A = non_parametric_fit(perm_group_A)
            n_perm_B, T_perm_B = non_parametric_fit(perm_group_B)
        except ValueError:
            # In case a permutation results in empty group, assign a large statistic
            perm_stats[i] = np.inf
            continue

        if common_x:
            n_common_perm, T_A_interp_perm, T_B_interp_perm = (
                n_common,
                T_perm_A,
                T_perm_B,
            )
        else:
            # Interpolate to common n
            n_common_perm, T_A_interp_perm, T_B_interp_perm = interpolate_common_n(
                n_perm_A, T_perm_A, n_perm_B, T_perm_B, num_points=num_points
            )

        # Compute test statistic
        perm_stats[i] = compute_area_between_curves(
            n_common_perm, T_A_interp_perm, T_B_interp_perm
        )

    # Calculate p-value based on the direction of observed_stat
    if observed_stat < 0:
        # f_A is faster; calculate the proportion of permuted stats <= observed_stat
        p_value = np.mean(perm_stats <= observed_stat)
        faster = "A"
    else:
        # f_B is faster; calculate the proportion of permuted stats >= observed_stat
        p_value = np.mean(perm_stats >= observed_stat)
        faster = "B"

    return ABTestResult(
        observed_stat=observed_stat,
        perm_stats=perm_stats,
        n_common=n_common,
        T_A_interp=T_A_interp,
        T_B_interp=T_B_interp,
        faster=faster,
        p_value=p_value,
    )


def plot_abtest_results(
    df: pd.DataFrame, result: ABTestResult, sup_title: str | None = None
):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # ----------------------------------------
    # 4.1. Scatter Plots with Smoothed Curves (First Visualization)
    # ----------------------------------------

    df_A = df[df["label"] == "A"]
    df_B = df[df["label"] == "B"]

    sns.scatterplot(
        data=df_A,
        x="n",
        y="T",
        label="f_A Data",
        alpha=0.7,
        ax=axes[0],
    )

    sns.scatterplot(
        data=df_B,
        x="n",
        y="T",
        label="f_B Data",
        alpha=0.7,
        ax=axes[0],
    )

    # Plot smoothed curves for 'A' and 'B'
    sns.lineplot(
        x=result.n_common,
        y=result.T_A_interp,
        ax=axes[0],
    )

    sns.lineplot(
        x=result.n_common,
        y=result.T_B_interp,
        ax=axes[0],
    )

    axes[0].set_xlabel("Input Size (n)")
    axes[0].set_ylabel("Running Time (T)")
    axes[0].set_title("Running Times")
    axes[0].legend()
    axes[0].grid(True)

    # ----------------------------------------
    # 4.2. Difference in Running Times (Second Visualization)
    # ----------------------------------------

    diff = result.T_A_interp - result.T_B_interp
    sns.lineplot(
        x=result.n_common, y=diff, label="f_A - f_B", color="purple", ax=axes[1]
    )

    # Add a horizontal line at y=0
    axes[1].axhline(0, color="black", linestyle="--")

    axes[1].set_xlabel("Input Size (n)")
    axes[1].set_ylabel("Difference in Running Time (f_A - f_B)")
    axes[1].set_title("Difference in Running Times")
    axes[1].legend()
    axes[1].grid(True)

    # ----------------------------------------
    # 4.3. Permutation Test Distribution (Third Visualization)
    # ----------------------------------------

    valid_perm_stats = result.perm_stats[np.isfinite(result.perm_stats)]

    # Histogram of the permutation test distribution
    sns.histplot(
        valid_perm_stats,
        stat="percent",
        bins=50,
        color="C7",
        alpha=0.8,
        label="Permutation Distribution",
        ax=axes[2],
    )

    axes[2].axvline(
        result.observed_stat,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Observed Statistic",
    )

    axes[2].set_xlabel("Signed Area Between Smoothed Curves")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title(
        f"Permutation Test Distribution\n{result.faster} is faster (p-value={result.p_value:.3f})"
    )
    axes[2].legend()
    axes[2].grid(True)

    # Adjust layout and display the plot
    fig.suptitle(sup_title, fontsize=16) if sup_title else None
    plt.tight_layout()


def detect_crossover_points(
    n_common: np.ndarray, T_A_interp: np.ndarray, T_B_interp: np.ndarray
) -> List[float]:
    """
    Detects crossover points where f_A and f_B intersect.

    Parameters:
    - n_common: Array of common input sizes.
    - T_A_interp, T_B_interp: Interpolated running times for groups A and B.

    Returns:
    - List of crossover input sizes.
    """
    diff = T_A_interp - T_B_interp
    sign_diff = np.sign(diff)
    sign_changes = np.where(np.diff(sign_diff) != 0)[0]
    crossover_n = []

    sign_changes = sign_changes[~np.isnan(sign_changes)]

    for idx in sign_changes[1:]:
        n1, n2 = n_common[idx], n_common[idx + 1]
        t1, t2 = diff[idx], diff[idx + 1]
        if t2 - t1 != 0:
            # Linear interpolation to estimate crossover point
            x = n1 - t1 * (n2 - n1) / (t2 - t1)
            crossover_n.append(x)

    return crossover_n


def detect_dynamic_threshold_crossings(
    n_common: np.ndarray,
    T_A_interp: np.ndarray,
    T_B_interp: np.ndarray,
    threshold_proportion: float,
) -> List[Tuple[float, str]]:
    """
    Detects points where the absolute difference between T_A_interp and T_B_interp
    crosses a dynamic threshold. The threshold is computed as a proportion of the
    midpoint of the two values at each n.

    Specifically, we define for each n:

      f(n) = |T_A(n) - T_B(n)| - (threshold_proportion * ((T_A(n) + T_B(n)) / 2))

    A zero crossing of f(n) indicates that the difference between the curves
    equals the threshold. A change from positive to negative means the curves
    have come "closer" than the threshold, and a change from negative to positive
    means they have moved "further" apart.

    Parameters:
      n_common           : Array of common input sizes.
      T_A_interp, T_B_interp : Interpolated values for the two curves.
      threshold_proportion   : The proportion used to compute the threshold at each n.

    Returns:
      A list of tuples (n, event) where:
         n     : The interpolated input size at which the threshold crossing occurs.
         event : A string: "closer" if the curves come within the dynamic threshold,
                 or "further" if they separate beyond it.
    """
    diff = T_A_interp - T_B_interp
    abs_diff = np.abs(diff)

    # Compute the midpoint of the two curves at each n.
    midpoint = (T_A_interp + T_B_interp) / 2
    # Compute the dynamic threshold at each n.
    threshold_vals = threshold_proportion * midpoint

    # f(n) is positive when the difference is larger than the threshold, negative otherwise.
    f = abs_diff - threshold_vals

    sign_f = np.sign(f)
    # Find indices where the sign of f changes.
    threshold_indices = np.where(np.diff(sign_f) != 0)[0][1:]
    threshold_events = []

    for idx in threshold_indices:
        n1, n2 = n_common[idx], n_common[idx + 1]
        f1, f2 = f[idx], f[idx + 1]
        # Avoid division by zero.
        if f2 - f1 == 0:
            continue

        # Linear interpolation to estimate the n at which f(n) crosses zero.
        fraction = (0 - f1) / (f2 - f1)
        n_cross = n1 + fraction * (n2 - n1)

        # Determine the event type:
        # If f goes from positive to negative, |T_A-T_B| dropped below the threshold.
        if f1 > 0 and f2 < 0:
            event = "closer"
        else:
            event = "further"

        threshold_events.append((n_cross, event))

    return threshold_events


def define_segments(
    n_common: np.ndarray, crossover_n: List[float]
) -> List[Tuple[float, float]]:
    """
    Defines segments based on crossover points.

    Parameters:
    - n_common: Array of common input sizes (must be sorted in ascending order).
    - crossover_n: List of crossover input sizes.

    Returns:
    - List of tuples representing segment boundaries.
      Each tuple contains (start, end) defining the range of the segment.
    """
    segments = []
    sorted_crossovers = sorted(crossover_n)

    if not sorted_crossovers:
        # No crossover points; single segment covering all input sizes
        segments.append((n_common[0], n_common[-1]))
    else:
        # Initialize the start of the first segment
        start = n_common[0]

        for crossover in sorted_crossovers:
            # Ensure crossover is within the range of n_common
            if crossover < n_common[0]:
                # Crossover point is before the first n_common point
                start = crossover
                continue
            elif crossover > n_common[-1]:
                # Crossover point is after the last n_common point
                break

            # Find the closest n_common point to the crossover
            closest_n = n_common[np.argmin(np.abs(n_common - crossover))]

            # Define the current segment
            segments.append((start, closest_n))

            # Update the start for the next segment
            start = closest_n

        # Add the final segment from the last crossover to the end
        segments.append((start, n_common[-1]))

    return segments


@dataclass
class SegmentedABTestResult:
    result: ABTestResult
    adjusted_pvalue: float
    null_rejected: bool

    def __str__(self):
        return f"{self.result.faster} is faster for {self.result.n_common.min():.3f} <= n <= {self.result.n_common.max():.3f} (p-value={self.result.p_value:.3f}, adj-p-value={self.adjusted_pvalue:.3f}, null_rejected={self.null_rejected})"


@dataclass
class SegmentedPermutationTestResult:
    segments: List[SegmentedABTestResult]
    warnings: List[str]


def segmented_permutation_test(
    df: pd.DataFrame,
    num_permutations=1000,
    num_points=50,
    seed=None,
) -> SegmentedPermutationTestResult:
    with warnings.catch_warnings(record=True) as w:

        group_A = df[df["label"] == "A"]
        group_B = df[df["label"] == "B"]

        group_A = remove_outliers_df(group_A, "n", "T")
        group_B = remove_outliers_df(group_B, "n", "T")

        n_A, T_A = non_parametric_fit(group_A)
        n_B, T_B = non_parametric_fit(group_B)
        n_common, T_A_interp, T_B_interp = interpolate_common_n(
            n_A, T_A, n_B, T_B, num_points=num_points
        )
        crossover_n = [
            n
            for n, _ in detect_dynamic_threshold_crossings(
                n_common, T_A_interp, T_B_interp, 0.05
            )
        ]
        # crossover_n = detect_crossover_points(n_common, T_A_interp, T_B_interp)
        log(f"Detected Crossover Points at n = {crossover_n}\n")

        segments = define_segments(n_common, crossover_n)
        log("Defined Segments:")
        for idx, segment in enumerate(segments, 1):
            log(f"Segment {idx}: n = {segment[0]:.2f} to n = {segment[1]:.2f}")
        log()

        # Initialize list to store results
        results: List[ABTestResult] = []

        for idx, segment in enumerate(segments, 1):
            if segment[0] < segment[1]:
                log(
                    f"Performing Permutation Test for Segment {idx}: n = {segment[0]:.2f} to n = {segment[1]:.2f}"
                )
                df_segment = df[
                    (df["n"] >= segment[0]) & (df["n"] <= segment[1])
                ].copy()
                if len(df_segment) > 8:
                    result = permutation_test(
                        df_segment,
                        num_permutations,
                        num_points,
                        seed,
                    )
                    log(f"{result.faster} is faster (p-value={result.p_value:.3f})\n")
                    results += [result]

        # Use Holm–Bonferroni to adjust for FDR.
        # Note: Tests are not independent, as adjacent segments may be not
        # be entirely independent.
        reject, adjusted_pvalues, _, _ = multipletests(
            [x.p_value for x in results], alpha=0.05, method="holm"
        )
        full_results = []
        for idx, result in enumerate(results):
            full_results += [
                SegmentedABTestResult(
                    result=result,
                    adjusted_pvalue=adjusted_pvalues[idx],
                    null_rejected=reject[idx],
                )
            ]

        messages = [f"ab_test: {wm.message}" for wm in w]
        reported = []
        for message in messages:
            if message not in reported:
                reported += [message]

    return SegmentedPermutationTestResult(segments=full_results, warnings=reported)


def plot_segmented_abtest_results(
    df: pd.DataFrame, results: List[ABTestResult], sup_title: str | None = None
):
    num_figs = 1 + len(results)
    fig, axes = plt.subplots(1, num_figs, figsize=(num_figs * 4, 4))

    # ----------------------------------------
    # 4.1. Scatter Plots with Smoothed Curves (First Visualization)
    # ----------------------------------------

    df_A = df[df["label"] == "A"]
    df_B = df[df["label"] == "B"]

    n_A, T_A = non_parametric_fit(df_A)
    n_B, T_B = non_parametric_fit(df_B)

    sns.scatterplot(
        data=df_A,
        x="n",
        y="T",
        label="f_A",
        alpha=0.7,
        ax=axes[0],
    )

    sns.scatterplot(
        data=df_B,
        x="n",
        y="T",
        label="f_B",
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

    for index, result in enumerate(results):
        if index > 0:
            axes[0].axvline(
                result.n_common.min(),
                color="black",
                linestyle="--",
                linewidth=1,
            )

    axes[0].set_xlabel("Input Size (n)")
    axes[0].set_ylabel("Running Time (T)")
    axes[0].set_title("Running Times")
    axes[0].legend()
    axes[0].grid(True)

    # ----------------------------------------
    # 4.2. Difference in Running Times and Permutation Test by Segment
    # ----------------------------------------

    for segment_number, result in enumerate(results):
        index = segment_number

        valid_perm_stats = result.perm_stats[np.isfinite(result.perm_stats)]

        # Histogram of the permutation test distribution
        sns.histplot(
            valid_perm_stats,
            stat="percent",
            bins=50,
            color="C7",
            alpha=0.8,
            label="Permutation Distribution",
            ax=axes[index + 1],
        )

        axes[index + 1].axvline(
            result.observed_stat,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Observed Statistic",
        )

        axes[index + 1].set_xlabel("Signed Area Between Smoothed Curves")
        axes[index + 1].set_ylabel("Frequency")
        axes[index + 1].set_title(
            f"Permutation Test for {result.n_common.min():.2f} <= n <= {result.n_common.max():.2f}\n{result.faster} is faster (p-value={result.p_value:.3f})"
        )
        axes[index + 1].legend()
        axes[index + 1].grid(True)

    # Adjust layout and display the plot
    fig.suptitle(sup_title, fontsize=16) if sup_title else None
    plt.tight_layout()
