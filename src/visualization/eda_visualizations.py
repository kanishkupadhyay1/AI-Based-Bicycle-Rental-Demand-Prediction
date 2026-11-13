"""Generate Exploratory Data Analysis (EDA) visualizations from the real UCI dataset."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import CLEANED_HOUR_CSV, PLOTS_DIR, ensure_directories, get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", palette="deep")


def generate_eda_plots(
    cleaned_path: Path = CLEANED_HOUR_CSV,
    plots_dir: Path = PLOTS_DIR,
) -> list[Path]:
    """Generate and save EDA plots from the cleaned hourly dataset."""
    ensure_directories()
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_path}")

    df = pd.read_csv(cleaned_path)
    saved_paths: list[Path] = []

    # 1. Demand Distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["cnt"], bins=45, kde=True, color="#2563eb", ax=ax)
    ax.axvline(df["cnt"].mean(), color="#dc2626", linestyle="--", label=f"Mean = {df['cnt'].mean():.1f}")
    ax.axvline(df["cnt"].median(), color="#16a34a", linestyle="-.", label=f"Median = {df['cnt'].median():.1f}")
    ax.set_title("Hourly Bicycle Rental Demand Distribution (cnt)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Total Hourly Rentals (cnt)")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    p1 = plots_dir / "demand_distribution.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved_paths.append(p1)

    # 2. Hourly Demand Pattern (Working Day vs Non-Working Day)
    fig, ax = plt.subplots(figsize=(10, 5))
    hourly_work = df.groupby(["hr", "workingday"])["cnt"].mean().reset_index()
    hourly_work["Day Type"] = hourly_work["workingday"].map({0: "Weekend / Holiday", 1: "Working Day"})
    sns.lineplot(
        data=hourly_work,
        x="hr",
        y="cnt",
        hue="Day Type",
        marker="o",
        linewidth=2.2,
        ax=ax,
    )
    ax.set_title("Average Hourly Bicycle Demand: Working Day vs Weekend/Holiday", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour of Day (0-23)")
    ax.set_ylabel("Average Hourly Rentals (cnt)")
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    p2 = plots_dir / "hourly_demand.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved_paths.append(p2)

    # 3. Weather & Environmental Demand Analysis
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    weather_labels = {
        1: "1: Clear/Partly Cloudy",
        2: "2: Mist/Cloudy",
        3: "3: Light Snow/Rain",
        4: "4: Heavy Rain/Snow",
    }
    weather_df = df.copy()
    weather_df["Weather"] = weather_df["weathersit"].map(weather_labels)
    sns.boxplot(data=weather_df, x="Weather", y="cnt", hue="Weather", ax=axes[0, 0], palette="Blues", legend=False)
    axes[0, 0].set_title("Demand by Weather Situation")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Hourly Rentals (cnt)")
    axes[0, 0].tick_params(axis="x", rotation=15)

    # Temperature vs Demand (binned mean)
    temp_bins = pd.cut(df["temp"], bins=12)
    temp_agg = df.groupby(temp_bins, observed=False)["cnt"].mean().reset_index()
    temp_agg["temp_mid"] = temp_agg["temp"].apply(lambda x: x.mid)
    sns.lineplot(data=temp_agg, x="temp_mid", y="cnt", marker="o", color="#ea580c", ax=axes[0, 1])
    axes[0, 1].set_title("Average Demand vs Normalized Temperature")
    axes[0, 1].set_xlabel("Normalized Temperature (temp)")
    axes[0, 1].set_ylabel("Average Rentals (cnt)")

    # Humidity vs Demand (binned mean)
    hum_bins = pd.cut(df["hum"], bins=12)
    hum_agg = df.groupby(hum_bins, observed=False)["cnt"].mean().reset_index()
    hum_agg["hum_mid"] = hum_agg["hum"].apply(lambda x: x.mid)
    sns.lineplot(data=hum_agg, x="hum_mid", y="cnt", marker="o", color="#0284c7", ax=axes[1, 0])
    axes[1, 0].set_title("Average Demand vs Normalized Humidity")
    axes[1, 0].set_xlabel("Normalized Humidity (hum)")
    axes[1, 0].set_ylabel("Average Rentals (cnt)")

    # Windspeed vs Demand (binned mean)
    wind_bins = pd.cut(df["windspeed"], bins=10)
    wind_agg = df.groupby(wind_bins, observed=False)["cnt"].mean().reset_index()
    wind_agg["wind_mid"] = wind_agg["windspeed"].apply(lambda x: x.mid)
    sns.lineplot(data=wind_agg, x="wind_mid", y="cnt", marker="o", color="#0d9488", ax=axes[1, 1])
    axes[1, 1].set_title("Average Demand vs Normalized Windspeed")
    axes[1, 1].set_xlabel("Normalized Windspeed (windspeed)")
    axes[1, 1].set_ylabel("Average Rentals (cnt)")

    plt.suptitle("Weather Factors vs Bicycle Rental Demand", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p3 = plots_dir / "weather_demand.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    saved_paths.append(p3)

    # 4. Seasonal and Monthly Demand
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    season_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
    season_df = df.copy()
    season_df["Season"] = season_df["season"].map(season_map)
    sns.barplot(data=season_df, x="Season", y="cnt", hue="Season", ax=axes[0], palette="viridis", errorbar=None, legend=False)
    axes[0].set_title("Average Hourly Demand by Season")
    axes[0].set_ylabel("Average Rentals (cnt)")

    monthly = df.groupby("mnth")["cnt"].mean().reset_index()
    sns.lineplot(data=monthly, x="mnth", y="cnt", marker="o", color="#2563eb", ax=axes[1])
    axes[1].set_title("Average Hourly Demand by Month")
    axes[1].set_xlabel("Month (1-12)")
    axes[1].set_ylabel("Average Rentals (cnt)")
    axes[1].set_xticks(range(1, 13))
    plt.tight_layout()
    p4 = plots_dir / "seasonal_monthly_demand.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    saved_paths.append(p4)

    # 5. Correlation Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_cols = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "cnt"]
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p5 = plots_dir / "correlation_matrix.png"
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    saved_paths.append(p5)

    logger.info("Generated %d EDA plots in %s", len(saved_paths), plots_dir)
    return saved_paths


if __name__ == "__main__":
    generate_eda_plots()
