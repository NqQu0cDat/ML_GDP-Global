"""
02_eda_analysis.py

Exploratory Data Analysis for the cleaned country economic dataset.

Input file:
- data/processed/country_economic_cleaned.csv

Output folder:
- reports/figures/
"""

from pathlib import Path
import os

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "country_economic_cleaned.csv"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
MATPLOTLIB_CACHE_DIR = BASE_DIR / "reports" / ".matplotlib_cache"

MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import seaborn as sns


def print_overview(df):
    """Print basic dataset information for the first EDA check."""
    print("\n===== DATASET OVERVIEW =====")
    print(f"Shape: {df.shape}")

    print("\nColumns:")
    print(list(df.columns))

    print("\nInfo:")
    df.info()

    print("\nDescribe:")
    print(df.describe())


def save_current_plot(filename, saved_files):
    """Save the current matplotlib figure and remember the output path."""
    output_path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_files.append(output_path)


def plot_top_10_gdp(df, saved_files):
    """Create a bar chart for the top 10 countries by GDP per capita."""
    top_10 = df.sort_values("gdp_per_capita", ascending=False).head(10)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=top_10,
        x="gdp_per_capita",
        y="country",
        hue="country",
        palette="viridis",
        legend=False,
    )
    plt.title("Top 10 Countries by GDP per Capita")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Country")
    save_current_plot("top_10_countries_by_gdp_per_capita.png", saved_files)


def plot_scatter(df, x_column, y_column, title, xlabel, ylabel, filename, saved_files):
    """Create and save a scatter plot for two numeric columns."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_column, y=y_column, s=70, alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    save_current_plot(filename, saved_files)


def plot_correlation_heatmap(df, saved_files):
    """Create a correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
    )
    plt.title("Correlation Heatmap of Economic Indicators")
    plt.xlabel("Economic Indicators")
    plt.ylabel("Economic Indicators")
    save_current_plot("correlation_heatmap.png", saved_files)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing cleaned dataset: {DATA_FILE}\n"
            "Please run notebooks/01_data_cleaning.py first."
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    print_overview(df)

    saved_files = []

    plot_top_10_gdp(df, saved_files)

    plot_scatter(
        df=df,
        x_column="annual_income",
        y_column="gdp_per_capita",
        title="Annual Income vs GDP per Capita",
        xlabel="Annual Income",
        ylabel="GDP per Capita",
        filename="annual_income_vs_gdp_per_capita.png",
        saved_files=saved_files,
    )

    plot_scatter(
        df=df,
        x_column="corruption_index",
        y_column="gdp_per_capita",
        title="Corruption Index vs GDP per Capita",
        xlabel="Corruption Index",
        ylabel="GDP per Capita",
        filename="corruption_index_vs_gdp_per_capita.png",
        saved_files=saved_files,
    )

    plot_scatter(
        df=df,
        x_column="unemployment_rate",
        y_column="gdp_per_capita",
        title="Unemployment Rate vs GDP per Capita",
        xlabel="Unemployment Rate",
        ylabel="GDP per Capita",
        filename="unemployment_rate_vs_gdp_per_capita.png",
        saved_files=saved_files,
    )

    plot_scatter(
        df=df,
        x_column="cost_index",
        y_column="gdp_per_capita",
        title="Cost Index vs GDP per Capita",
        xlabel="Cost Index",
        ylabel="GDP per Capita",
        filename="cost_index_vs_gdp_per_capita.png",
        saved_files=saved_files,
    )

    plot_scatter(
        df=df,
        x_column="purchasing_power_index",
        y_column="gdp_per_capita",
        title="Purchasing Power Index vs GDP per Capita",
        xlabel="Purchasing Power Index",
        ylabel="GDP per Capita",
        filename="purchasing_power_index_vs_gdp_per_capita.png",
        saved_files=saved_files,
    )

    plot_correlation_heatmap(df, saved_files)

    print("\n===== SAVED FIGURES =====")
    for file_path in saved_files:
        print(file_path)


if __name__ == "__main__":
    main()
