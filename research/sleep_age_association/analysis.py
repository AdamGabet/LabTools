import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from body_system_loader.load_feature_df import load_columns_as_df, load_body_system_df

# Plotting Standards - Nature Medicine style
plt.style.use("seaborn-v0_8-ticks")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "font.family": "sans-serif",
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)
MALE_COLOR = "#3498DB"  # Blue
FEMALE_COLOR = "#E74C3C"  # Orange-Red


def calculate_percentiles_lowess(x, y, percentiles=[3, 10, 50, 90, 97]):
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]

    bins = np.linspace(x_sorted.min(), x_sorted.max(), 25)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    quantile_values = []

    for i in range(len(bins) - 1):
        mask = (x_sorted >= bins[i]) & (x_sorted < bins[i + 1])
        if np.any(mask):
            bin_data = y_sorted[mask]
            quantile_values.append(
                [np.quantile(bin_data, p / 100) for p in percentiles]
            )
        else:
            quantile_values.append([np.nan] * len(percentiles))

    res = np.array(quantile_values).T
    smoothed_lines = []
    for p_idx in range(len(percentiles)):
        y_vals = res[p_idx]
        mask = ~np.isnan(y_vals)
        if np.sum(mask) > 4:
            smoothed = lowess(bin_centers[mask], y_vals[mask], frac=0.4)
            smoothed_lines.append((bin_centers[mask], smoothed[:, 1]))
        else:
            smoothed_lines.append(None)
    return smoothed_lines


def create_joint_panel(
    fig, outer_ax, df, feature, label, is_log=False, severity_zones=None
):
    # Use the outer_ax as the main plot and add marginals relative to it
    ax_main = outer_ax
    pos = ax_main.get_position()

    # Add Top Axis (Age Hist)
    ax_top = fig.add_axes([pos.x0, pos.y1 + 0.01, pos.width, 0.05])
    # Add Right Axis (Feature Hist)
    ax_right = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.05, pos.height])

    df_m = df[df["gender"] == 1]
    df_f = df[df["gender"] == 2]

    # Main Scatter
    ax_main.scatter(
        df_m["age"], df_m[feature], color=MALE_COLOR, alpha=0.2, s=5, edgecolors="none"
    )
    ax_main.scatter(
        df_f["age"],
        df_f[feature],
        color=FEMALE_COLOR,
        alpha=0.2,
        s=5,
        edgecolors="none",
    )

    # Regression Lines
    for data, color, label_sex in [(df_m, MALE_COLOR, "M"), (df_f, FEMALE_COLOR, "F")]:
        mask = data[feature].notna()
        x, y = data.loc[mask, "age"], data.loc[mask, feature]
        if len(x) > 1:
            slope, intercept, r, p, se = stats.linregress(x, y)
            ax_main.plot(x, intercept + slope * x, color=color, linewidth=1.5)
            ax_main.text(
                0.05,
                0.95,
                f"{label_sex}: p={p:.3f}",
                transform=ax_main.transAxes,
                color=color,
                fontsize=8,
                verticalalignment="top",
            )

    # Population Percentiles
    mask = df[feature].notna()
    x_all, y_all = df.loc[mask, "age"].values, df.loc[mask, feature].values
    if len(x_all) > 10:
        lines = calculate_percentiles_lowess(x_all, y_all)
        for line in lines:
            if line:
                ax_main.plot(
                    line[0],
                    line[1],
                    color="black",
                    linestyle=":",
                    linewidth=0.8,
                    zorder=10,
                )

    # Marginal Histograms
    sns.histplot(
        df_m["age"],
        ax=ax_top,
        color=MALE_COLOR,
        alpha=0.4,
        stat="density",
        element="step",
        fill=False,
    )
    sns.histplot(
        df_f["age"],
        ax=ax_top,
        color=FEMALE_COLOR,
        alpha=0.4,
        stat="density",
        element="step",
        fill=False,
    )

    sns.histplot(
        df_m[feature],
        ax=ax_right,
        color=MALE_COLOR,
        alpha=0.4,
        stat="density",
        element="step",
        fill=False,
    )
    sns.histplot(
        df_f[feature],
        ax=ax_right,
        color=FEMALE_COLOR,
        alpha=0.4,
        stat="density",
        element="step",
        fill=False,
    )

    # Formatting
    if is_log:
        ax_main.set_yscale("log")
    ax_main.set_ylabel(label)
    ax_main.set_xlabel("Age")

    if severity_zones:
        for lower, upper, color in severity_zones:
            ax_main.axhspan(lower, upper, color=color, alpha=0.1)

    ax_top.set_axis_off()
    ax_right.set_axis_off()

    return ax_main


def main():
    print("Loading data...")
    demographics = load_columns_as_df(["age", "gender"])
    sleep_df = load_body_system_df("sleep")
    df = demographics.join(sleep_df, how="inner")
    df = df[(df["age"] >= 40) & (df["age"] <= 70)]

    features_config = [
        {
            "col": "ahi",
            "label": "pAHI (events/hr)",
            "is_log": True,
            "zones": [(0, 15, "green"), (15, 30, "yellow"), (30, 1000, "red")],
        },
        {
            "col": "desaturations_mean_nadir",
            "label": "Mean Nadir SpO2 (%)",
            "is_log": False,
            "zones": None,
        },
        {
            "col": "percent_of_light_sleep_time",
            "label": "Light Sleep (%)",
            "is_log": False,
            "zones": None,
        },
        {
            "col": "percent_of_deep_sleep_time",
            "label": "Deep Sleep (%)",
            "is_log": False,
            "zones": None,
        },
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for i, config in enumerate(features_config):
        create_joint_panel(
            fig,
            axes[i],
            df,
            config["col"],
            config["label"],
            is_log=config["is_log"],
            severity_zones=config["zones"],
        )

    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    import os

    os.makedirs("research/sleep_age_association/figures", exist_ok=True)
    plt.savefig(
        "research/sleep_age_association/figures/sleep_age_association.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(
        "Figure saved to research/sleep_age_association/figures/sleep_age_association.png"
    )


if __name__ == "__main__":
    main()
