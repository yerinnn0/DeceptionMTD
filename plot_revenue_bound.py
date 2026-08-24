"""Plot 5x5-gridworld revenue and manuscript bounds for three deceptions.

Run this file after producing the consolidated optimization PKLs:

    python plot_revenue_bounds.py

The output uses the same visual convention as ``plot.ipynb``: actual revenue
is blue, nominal revenue is dashed gray, the theorem bound is dotted gray,
and the area between the bound and nominal revenue is lightly shaded.

Expected beta ranges (10 points each, including zero and the endpoint):
diversionary 0--0.07, targeted 0--0.15, and equivocal 0--0.05.
"""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


# ---------------------------------------------------------------------------
# EDIT 1: change only these file names when EXPERIMENT_NAME/file tags differ.
# ---------------------------------------------------------------------------
PKL_FILES = {
    "Nominal": "result_gridworld_5x5_1_lp.pkl",
    "Diversionary": "final_rev1_gridworld_5x5_1_div_opt.pkl",
    "Targeted": "final_rev1_gridworld_5x5_1_tar_opt.pkl",
    "Equivocal": "final_rev1_gridworld_5x5_1_equ_opt.pkl",
}

OUTPUT_FILE = "gridworld_5x5_revenue_bounds.png"

# Final experiment design.  These values only validate the loaded PKLs; the
# plotted x values always come from each file's saved beta_vec.
EXPECTED_BETA_MAX = {
    "Diversionary": 0.07,
    "Targeted": 0.15,
    "Equivocal": 0.05,
}
EXPECTED_BETA_COUNT = 11

# A beta=0 deceptive solve is mathematically the nominal problem.  OSQP may
# nevertheless stop early because the quadratic term vanishes at beta=0.
# True: plot the exact nominal value (100%) at beta=0 and print the raw value.
# False: show the raw solver output even when its status was inaccurate.
USE_EXACT_NOMINAL_AT_BETA_ZERO = True


def to_numpy(value):
    """Convert NumPy/Torch data to a floating-point NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=float)


def scalar(value):
    """Convert a NumPy/Torch scalar to a Python float."""
    return float(to_numpy(value).reshape(-1)[0])


def load_logger(path):
    with path.open("rb") as file:
        logger = pickle.load(file)
    if not isinstance(logger, dict) or "results" not in logger:
        raise ValueError(f"{path.name} is not a consolidated result PKL.")
    if not logger["results"]:
        raise ValueError(f"{path.name} contains no results.")
    return logger


def occupancy(result, key):
    array = to_numpy(result["occupancy_measure"][key]).squeeze()
    if array.ndim != 2:
        raise ValueError(f"{key} must have shape (n_states, n_actions), got {array.shape}.")
    return array


def result_settings(result):
    try:
        return result["config"]["MMDP_SETTINGS"]
    except KeyError as error:
        raise KeyError("Result PKL is missing config/MMDP_SETTINGS.") from error


def preferred_and_decoy_states(settings, n_states):
    """Recover the exact sets used by the gridworld equivocal objective."""
    shape = tuple(int(v) for v in settings["grid_shape"])
    rows, cols = shape
    if rows * cols != n_states:
        raise ValueError(f"grid_shape={shape} does not match {n_states} states.")

    terminal = int(settings["terminal_state"])
    # Task reachability uses P_T including T; equivocal deception deliberately
    # excludes the shared terminal and compares preferred versus decoy states.
    preferred = np.asarray(
        [int(state) for state in settings["task_states"] if int(state) != terminal],
        dtype=int,
    )
    decoy = np.asarray(
        [row * cols for row in range(1, rows)]
        + [(rows - 1) * cols + col for col in range(1, cols - 1)],
        dtype=int,
    )
    return preferred, decoy


def bound_coefficients(nominal_result, targeted_result):
    """Return theorem coefficients C such that revenue/R* >= 1 - C beta."""
    x_star = occupancy(nominal_result, "occupancy_measure")
    x_tar = occupancy(targeted_result, "target_occupancy_measure")
    if x_tar.shape != x_star.shape:
        raise ValueError(f"x_tar shape {x_tar.shape} differs from x* shape {x_star.shape}.")
    if np.min(x_tar) < -1e-9:
        raise ValueError("The gridworld targeted bound expects non-negative x_tar.")

    settings = result_settings(nominal_result)
    gamma = float(settings["gamma"])
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must satisfy 0 <= gamma < 1.")
    r_star = scalar(nominal_result["revenue"]["revenue"])
    if r_star <= 0.0:
        raise ValueError(f"The percentage bound requires positive nominal R*, got {r_star}.")

    one_minus_gamma_inv = 1.0 / (1.0 - gamma)
    x_star_square = float(np.sum(x_star**2))
    x_star_x_tar = float(np.sum(x_star * x_tar))
    max_x_tar = float(np.max(x_tar))
    state_action_count = x_star.size  # |S||A|

    # Theorem 1: diversionary-deception revenue-loss bound.
    diversionary = (
        x_star_square + one_minus_gamma_inv**2
    ) / r_star

    # -----------------------------------------------------------------------
    # EDIT 2 / DOUBLE-CHECK: Theorem 2 uses max(x_tar), not max(x_tar)-min(x_tar).
    # This is the manuscript formula for the non-negative gridworld occupancy
    # target.  Do not copy the legacy MTD notebook's signed-weight adjustment.
    # -----------------------------------------------------------------------
    targeted = (
        x_star_square
        - 2.0 * x_star_x_tar
        - one_minus_gamma_inv**2 / state_action_count
        + 2.0 * one_minus_gamma_inv * max_x_tar
    ) / r_star

    preferred, decoy = preferred_and_decoy_states(settings, x_star.shape[0])
    x_star_by_state = np.sum(x_star, axis=1)
    preferred_mass = float(np.sum(x_star_by_state[preferred]))
    decoy_mass = float(np.sum(x_star_by_state[decoy]))

    # -----------------------------------------------------------------------
    # EDIT 3 / DOUBLE-CHECK: match the implemented equivocal objective.
    # It compares P_T\{T} against P_D; the task constraint still uses P_T
    # including T.  The terminal is shared by both routes and is excluded here.
    # -----------------------------------------------------------------------
    equivocal = (preferred_mass - decoy_mass) ** 2 / r_star

    diagnostics = {
        "R_star": r_star,
        "gamma": gamma,
        "sum_x_star_squared": x_star_square,
        "sum_x_star_x_tar": x_star_x_tar,
        "max_x_tar": max_x_tar,
        "state_action_count": state_action_count,
        "preferred_mass_x_star": preferred_mass,
        "decoy_mass_x_star": decoy_mass,
    }
    coefficients = {
        "Diversionary": diversionary,
        "Targeted": targeted,
        "Equivocal": equivocal,
    }
    return coefficients, diagnostics


def revenue_curve(logger, r_star, label):
    betas = np.asarray(logger.get("beta_vec", []), dtype=float)
    results = logger["results"]
    if len(betas) != len(results):
        raise ValueError(f"{label}: {len(betas)} beta values but {len(results)} results.")

    revenues = np.asarray(
        [scalar(result["revenue"]["deceptive_revenue"]) for result in results],
        dtype=float,
    )
    percentages = 100.0 * revenues / r_star

    # -----------------------------------------------------------------------
    # EDIT 4 / DOUBLE-CHECK: normalize every method by the nominal LP R*.
    # Never normalize by the first deceptive result; an inaccurate beta=0 QP
    # can otherwise make later points and the 100% reference line misleading.
    # -----------------------------------------------------------------------
    zero_indices = np.flatnonzero(np.isclose(betas, 0.0, atol=1e-12))
    if USE_EXACT_NOMINAL_AT_BETA_ZERO:
        for index in zero_indices:
            raw = percentages[index]
            if not np.isclose(raw, 100.0, atol=1e-4):
                print(
                    f"WARNING: {label} beta=0 raw revenue is {raw:.6f}% of R*. "
                    "Plotting the exact nominal value 100%; check solver status/residuals."
                )
            percentages[index] = 100.0
    # return betas, percentages
    return betas, revenues, percentages


def format_value_list(values):
    """Format a numeric vector as a compact, copyable Python-style list."""
    return np.array2string(
        np.asarray(values, dtype=float),
        precision=6,
        separator=", ",
        suppress_small=False,
        max_line_width=200,
    )


def print_revenue_and_bound_values(
    label,
    betas,
    actual_revenue,
    revenue_percent,
    bound_revenue,
    bound_percent,
):
    """Print the exact arrays used to inspect and draw each revenue panel."""
    print(f"\n{label} revenue and bound values")
    print(f"  beta: {format_value_list(betas)}")
    print(f"  actual revenue: {format_value_list(actual_revenue)}")
    print(f"  actual revenue (% of nominal): {format_value_list(revenue_percent)}")
    print(f"  revenue bound: {format_value_list(bound_revenue)}")
    print(f"  revenue bound (% of nominal): {format_value_list(bound_percent)}")


def task_reach(result, deceptive):
    """Evaluate the implemented P_T transition-reachability left-hand side."""
    settings = result_settings(result)
    task_states = np.asarray(settings["task_states"], dtype=int)
    transition = to_numpy(result["transition_matrix"]).squeeze()
    key = "deceptive_occupancy_measure" if deceptive else "occupancy_measure"
    x = occupancy(result, key)
    if transition.shape != (x.shape[0], x.shape[1], x.shape[0]):
        raise ValueError(
            f"transition matrix shape {transition.shape} is incompatible with x shape {x.shape}."
        )
    return float(np.sum(x * np.sum(transition[:, :, task_states], axis=2)))


def print_double_check(coefficients, diagnostics):
    print("\nBound inputs (double-check against the manuscript)")
    for key, value in diagnostics.items():
        print(f"  {key}: {value:.10g}")
    print("Bound coefficient C in 100 * (1 - C * beta)")
    for label, value in coefficients.items():
        print(f"  {label}: {value:.10g}")


def main():
    base_dir = Path(__file__).resolve().parent
    loggers = {}
    for label, file_name in PKL_FILES.items():
        path = base_dir / file_name
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Update PKL_FILES at the top of this file."
            )
        loggers[label] = load_logger(path)

    nominal_result = loggers["Nominal"]["results"][0]
    targeted_result = loggers["Targeted"]["results"][0]
    coefficients, diagnostics = bound_coefficients(nominal_result, targeted_result)
    print_double_check(coefficients, diagnostics)
    r_star = diagnostics["R_star"]

    titles = {
        "Diversionary": "Revenue of diversionary deception",
        "Targeted": "Revenue of targeted deception",
        "Equivocal": "Revenue of equivocal deception",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 3), constrained_layout=True)

    for axis, label in zip(axes, ("Diversionary", "Targeted", "Equivocal")):

        betas, actual_revenue, revenue_percent = revenue_curve(
            loggers[label], r_star, label
        )
        expected_max = EXPECTED_BETA_MAX[label]
        if len(betas) != EXPECTED_BETA_COUNT or not np.isclose(
            np.max(betas), expected_max, atol=1e-10
        ):
            print(
                f"WARNING: {label} PKL has {len(betas)} beta values with "
                f"max={np.max(betas):.10g}; expected {EXPECTED_BETA_COUNT} values "
                f"from 0 to {expected_max}."
            )
        bound_percent = 100.0 * (1.0 - coefficients[label] * betas)
        bound_revenue = r_star * (1.0 - coefficients[label] * betas)
        print_revenue_and_bound_values(
            label,
            betas,
            actual_revenue,
            revenue_percent,
            bound_revenue,
            bound_percent,
        )
        reaches = np.asarray(
            [task_reach(result, deceptive=True) for result in loggers[label]["results"]]
        )
        v_reach = float(result_settings(loggers[label]["results"][0])["v_reach"])
        print(
            f"  {label} P_T transition reach: min={np.min(reaches):.10g}, "
            f"required={v_reach:.10g}"
        )
        if np.min(reaches) < v_reach - 1e-6:
            print(
                f"WARNING: {label} contains a saved solution that violates v_reach. "
                "Re-run it and inspect solver status/residuals."
            )

        axis.plot(betas, revenue_percent, color="blue", linewidth=1.8, label="Revenue")
        axis.axhline(
            100.0, color="gray", linestyle="--", linewidth=1.2, label="Optimal revenue"
        )
        axis.plot(
            betas,
            bound_percent,
            color="gray",
            linestyle=":",
            linewidth=1.3,
            label="Revenue bound",
        )
        axis.fill_between(betas, bound_percent, 100.0, color="C0", alpha=0.2)
        axis.set_title(titles[label])
        axis.set_xlabel(r"Deception parameter, $\beta$")
        axis.set_ylabel("Revenue (%)")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.legend(loc="lower left")

        min_margin = float(np.min(revenue_percent - bound_percent))
        print(f"  {label} minimum revenue-minus-bound margin: {min_margin:.6g} percentage points")
        if min_margin < -1e-3:
            print(
                f"WARNING: {label} falls below the theoretical bound. Check solver "
                "optimality/feasibility and confirm that its objective matches the theorem."
            )

    output_path = base_dir / OUTPUT_FILE
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()