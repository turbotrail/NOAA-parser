"""Exploratory Data Analysis for NOAA geomagnetic field readings exported from InfluxDB.

This script loads an InfluxDB CSV export, cleans the data, prints summary statistics,
computes time coverage and gaps, and writes diagnostics plots to an output directory.
"""

from __future__ import annotations

import argparse
import os
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    sns = None


DEFAULT_PLOT_DIR = "eda_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perform EDA on a NOAA geomagnetic InfluxDB CSV export."
    )
    parser.add_argument(
        "input_csv",
        help="Path to the exported InfluxDB CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_PLOT_DIR,
        help="Directory where summary and plot outputs will be written.",
    )
    parser.add_argument(
        "--field",
        help="Only analyze this _field value (for example: F, D, H, Z).",
    )
    parser.add_argument(
        "--observatory",
        help="Only analyze this observatory station code (for example: BOU).",
    )
    parser.add_argument(
        "--start-date",
        help="Filter data on or after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        help="Filter data on or before this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--resample",
        help="Resample the time series at this frequency string for plotting (for example: 1H, 1D).",
    )
    return parser.parse_args()


def load_influx_csv(csv_path: str) -> pd.DataFrame:
    """Load a CSV exported from InfluxDB and return a cleaned DataFrame."""
    df = pd.read_csv(
        csv_path,
        comment="#",
        na_values=["", "null", "NULL"],
    )

    if "_time" not in df.columns or "_value" not in df.columns:
        raise ValueError(
            "CSV does not contain required InfluxDB export columns '_time' and '_value'."
        )

    df["_time"] = pd.to_datetime(df["_time"], utc=True, errors="coerce")
    df["_value"] = pd.to_numeric(df["_value"], errors="coerce")

    df = df.dropna(subset=["_time", "_value"]).reset_index(drop=True)
    df["date"] = df["_time"].dt.date
    df["hour"] = df["_time"].dt.hour
    df["weekday"] = df["_time"].dt.day_name()

    return df


def filter_data(
    df: pd.DataFrame,
    field: str | None = None,
    observatory: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    if field:
        df = df[df["_field"].astype(str) == field]
    if observatory:
        obs_list = [o.strip() for o in observatory.split(",")]
        df = df[df["observatory"].astype(str).isin(obs_list)]
    if start_date:
        df = df[df["_time"] >= pd.to_datetime(start_date, utc=True)]
    if end_date:
        df = df[df["_time"] <= pd.to_datetime(end_date, utc=True)]
    return df.reset_index(drop=True)


def summarize_data(df: pd.DataFrame, output_dir: str) -> None:
    summary_path = os.path.join(output_dir, "eda_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("DataFrame shape: %s\n" % (df.shape,))
        file.write("\nColumn data types:\n")
        file.write(df.dtypes.to_string())
        file.write("\n\nMissing values by column:\n")
        file.write(df.isna().sum().to_string())
        file.write("\n\nUnique value counts:\n")

        for column in ["_field", "_measurement", "data_type", "observatory"]:
            if column in df.columns:
                file.write(f"\n\n{column}:\n")
                file.write(df[column].value_counts(dropna=False).to_string())

        file.write("\n\nNumeric description for _value:\n")
        file.write(df["_value"].describe().to_string())

        time_min = df["_time"].min()
        time_max = df["_time"].max()
        file.write(f"\n\nTime range: {time_min} to {time_max}\n")

        if len(df) > 1:
            diff = df.sort_values("_time")["_time"].diff().dropna()
            file.write(f"Median time delta: {diff.median()}\n")
            file.write(f"Minimum time delta: {diff.min()}\n")
            file.write(f"Maximum time delta: {diff.max()}\n")

    print(f"Saved EDA summary to {summary_path}")


def detect_gaps(df: pd.DataFrame, output_dir: str) -> None:
    gap_report_path = os.path.join(output_dir, "gap_report.txt")
    df_sorted = df.sort_values("_time").reset_index(drop=True)
    if len(df_sorted) < 2:
        print("Not enough data to detect gaps.")
        return

    time_diff = df_sorted["_time"].diff()
    median_spacing = time_diff.iloc[1:].median()
    threshold = median_spacing * 3
    gap_mask = time_diff > threshold
    large_gaps = df_sorted.loc[gap_mask, ["_time"]].copy()
    large_gaps["next_time"] = pd.to_datetime(
        df_sorted["_time"].shift(-1).loc[gap_mask], utc=True, errors="coerce"
    )
    large_gaps["_time"] = pd.to_datetime(large_gaps["_time"], utc=True, errors="coerce")
    large_gaps["gap_duration"] = time_diff.loc[gap_mask]

    with open(gap_report_path, "w", encoding="utf-8") as file:
        file.write(f"Median sample spacing: {median_spacing}\n")
        file.write(f"Threshold for large gap detection: {threshold}\n")
        file.write(f"Number of large gaps: {len(large_gaps)}\n\n")
        if not large_gaps.empty:
            file.write(large_gaps.to_string(index=False))
        else:
            file.write("No large gaps detected.\n")

    print(f"Saved gap report to {gap_report_path}")


def plot_time_series(
    df: pd.DataFrame,
    output_dir: str,
    resample_freq: str | None = None,
) -> None:
    pivot = None
    if "_field" in df.columns:
        if "observatory" in df.columns and df["observatory"].nunique() > 1:
            pivot = df.pivot_table(
                index="_time",
                columns=["observatory", "_field"],
                values="_value",
                aggfunc="mean",
            )
        elif df["_field"].nunique() <= 6:
            pivot = df.pivot_table(
                index="_time",
                columns="_field",
                values="_value",
                aggfunc="mean",
            )
        else:
            pivot = df.set_index("_time")["_value"]
    else:
        pivot = df.set_index("_time")["_value"]

    if resample_freq:
        pivot = pivot.resample(resample_freq).mean()
    elif len(pivot) > 1:
        time_diff = pd.Series(pivot.index).diff()
        threshold = time_diff.median() * 3
        large_gaps = time_diff > threshold
        if large_gaps.any():
            gap_times = pivot.index[large_gaps] - time_diff.median() / 2
            pivot = pivot.reindex(pivot.index.union(gap_times).sort_values())
    if isinstance(pivot, pd.DataFrame) and isinstance(pivot.columns, pd.MultiIndex) and pivot.columns.names[0] == "observatory":
        observatories = pivot.columns.levels[0]
        n_obs = len(observatories)
        fig, axes = plt.subplots(nrows=n_obs, figsize=(12, 4 * n_obs), sharex=True)
        if n_obs == 1:
            axes = [axes]
        for i, obs in enumerate(observatories):
            ax = axes[i]
            pivot[obs].plot(ax=ax, linewidth=1)
            ax.set_title(f"Observatory: {obs}")
            ax.set_ylabel("_value")
            ax.legend(title="_field", loc="upper left", bbox_to_anchor=(1, 1))
        axes[-1].set_xlabel("Time")
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        if isinstance(pivot, pd.DataFrame):
            pivot.plot(ax=ax, linewidth=1)
            ax.legend(title="_field")
        else:
            pivot.plot(ax=ax, color="tab:blue", linewidth=1)
        ax.set_title("Geomagnetic Value Time Series")
        ax.set_xlabel("Time")
        ax.set_ylabel("_value")
        fig.tight_layout()

    chart_path = os.path.join(output_dir, "time_series.png")
    fig.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved time series plot to {chart_path}")


def plot_distribution(df: pd.DataFrame, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["_value"].dropna(), bins=50, color="#3b8ad9", edgecolor="black")
    ax.set_title("Distribution of Geomagnetic Values")
    ax.set_xlabel("_value")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    path = os.path.join(output_dir, "value_distribution.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved distribution plot to {path}")

    if sns is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.kdeplot(df["_value"].dropna(), fill=True, ax=ax)
        ax.set_title("Kernel Density Estimate of _value")
        fig.tight_layout()
        path = os.path.join(output_dir, "value_kde.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Saved KDE plot to {path}")


def plot_boxplot(df: pd.DataFrame, output_dir: str) -> None:
    if "_field" not in df.columns or df["_field"].nunique() > 10:
        print("Skipping boxplot by field because there are too many unique _field values.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    df_sorted = df.sort_values("_field")
    ax.boxplot(
        [group["_value"].dropna().values for _, group in df_sorted.groupby("_field")],
        tick_labels=df_sorted["_field"].unique(),
        patch_artist=True,
    )
    ax.set_title("Value Distribution by _field")
    ax.set_xlabel("_field")
    ax.set_ylabel("_value")
    fig.tight_layout()
    path = os.path.join(output_dir, "value_boxplot_by_field.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved boxplot to {path}")


def plot_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    if "weekday" not in df.columns or "hour" not in df.columns:
        return

    counts = (
        df.groupby(["weekday", "hour"])["_value"]
        .count()
        .unstack(fill_value=0)
        .reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            fill_value=0,
        )
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    if sns is not None:
        sns.heatmap(counts, fmt="d", cmap="rocket", ax=ax)
    else:
        im = ax.imshow(counts, aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=ax)
        ax.set_yticks(range(len(counts.index)))
        ax.set_yticklabels(counts.index)
        ax.set_xticks(range(len(counts.columns)))
        ax.set_xticklabels(counts.columns)

    ax.set_title("Record Count by Weekday and Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Weekday")
    fig.tight_layout()
    path = os.path.join(output_dir, "weekday_hour_heatmap.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved heatmap to {path}")


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    df = load_influx_csv(args.input_csv)
    df = filter_data(
        df,
        field=args.field,
        observatory=args.observatory,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if df.empty:
        raise SystemExit("No rows remain after filtering. Check your input CSV and filter values.")

    summarize_data(df, args.output_dir)
    detect_gaps(df, args.output_dir)
    plot_time_series(df, args.output_dir, resample_freq=args.resample)
    plot_distribution(df, args.output_dir)
    plot_boxplot(df, args.output_dir)
    plot_heatmap(df, args.output_dir)

    print("EDA complete.")


if __name__ == "__main__":
    main()
