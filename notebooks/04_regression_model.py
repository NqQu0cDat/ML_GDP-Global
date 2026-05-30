"""
04_regression_model.py

Train regression models to predict GDP per capita.

Input file:
- data/processed/country_economic_clustered.csv

Output files:
- reports/figures/linear_regression_actual_vs_predicted.png
- reports/figures/random_forest_actual_vs_predicted.png
- reports/figures/random_forest_feature_importance.png
- models/random_forest_gdp_model.pkl
- models/regression_features.pkl
"""

from pathlib import Path
import os
import warnings

# Avoid a macOS/joblib physical-core warning when Random Forest runs.
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="joblib.externals.loky.backend.context",
)

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "country_economic_clustered.csv"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
MODELS_DIR = BASE_DIR / "models"
RANDOM_FOREST_MODEL_FILE = MODELS_DIR / "random_forest_gdp_model.pkl"
REGRESSION_FEATURES_FILE = MODELS_DIR / "regression_features.pkl"
MATPLOTLIB_CACHE_DIR = BASE_DIR / "reports" / ".matplotlib_cache"

MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import seaborn as sns


TARGET = "gdp_per_capita"

FEATURE_CANDIDATES = [
    "annual_income",
    "corruption_index",
    "cost_index",
    "monthly_income",
    "purchasing_power_index",
    "unemployment_rate",
    "tourists_in_millions",
    "receipts_in_billions",
    "percentage_of_gdp",
    "cluster",
]


def get_available_features(df):
    """Use only regression features that exist in the dataframe."""
    available_features = [column for column in FEATURE_CANDIDATES if column in df.columns]
    missing_features = [column for column in FEATURE_CANDIDATES if column not in df.columns]

    print("\n===== FEATURE SELECTION =====")
    print("Features used for regression:")
    print(available_features)

    if missing_features:
        print("\nFeatures skipped because they are not in the dataset:")
        print(missing_features)

    if not available_features:
        raise ValueError("No valid features found for regression.")

    return available_features


def calculate_metrics(y_true, y_pred):
    """Calculate common regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2,
    }


def print_model_comparison(results):
    """Print model metrics side by side for easy comparison."""
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df[["MAE", "RMSE", "R2 Score"]]

    print("\n===== MODEL COMPARISON =====")
    print(comparison_df.round(4))


def plot_actual_vs_predicted(y_true, y_pred, title, filename):
    """Create an Actual vs Predicted scatter plot."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, s=70, alpha=0.75)

    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())
    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        color="red",
        linestyle="--",
        label="Perfect Prediction",
    )

    plt.title(title)
    plt.xlabel("Actual GDP per Capita")
    plt.ylabel("Predicted GDP per Capita")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_random_forest_feature_importance(model, features):
    """Create a bar chart showing which features Random Forest used most."""
    importance_df = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df,
        x="importance",
        y="feature",
        hue="feature",
        palette="viridis",
        legend=False,
    )
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "random_forest_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing clustered dataset: {DATA_FILE}\n"
            "Please run notebooks/03_clustering.py first."
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    print("\n===== DATASET LOADED =====")
    print(f"Shape: {df.shape}")

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' was not found in the dataset.")

    features = get_available_features(df)

    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("\n===== TRAIN TEST SPLIT =====")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    linear_regression = LinearRegression()
    random_forest = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=None,
    )

    linear_regression.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)

    linear_predictions = linear_regression.predict(X_test)
    random_forest_predictions = random_forest.predict(X_test)

    results = {
        "Linear Regression": calculate_metrics(y_test, linear_predictions),
        "Random Forest": calculate_metrics(y_test, random_forest_predictions),
    }
    print_model_comparison(results)

    plot_actual_vs_predicted(
        y_true=y_test,
        y_pred=linear_predictions,
        title="Linear Regression: Actual vs Predicted GDP per Capita",
        filename="linear_regression_actual_vs_predicted.png",
    )

    plot_actual_vs_predicted(
        y_true=y_test,
        y_pred=random_forest_predictions,
        title="Random Forest: Actual vs Predicted GDP per Capita",
        filename="random_forest_actual_vs_predicted.png",
    )

    plot_random_forest_feature_importance(random_forest, features)

    joblib.dump(random_forest, RANDOM_FOREST_MODEL_FILE)
    joblib.dump(features, REGRESSION_FEATURES_FILE)

    print("\n===== SAVED FILES =====")
    print(FIGURES_DIR / "linear_regression_actual_vs_predicted.png")
    print(FIGURES_DIR / "random_forest_actual_vs_predicted.png")
    print(FIGURES_DIR / "random_forest_feature_importance.png")
    print(RANDOM_FOREST_MODEL_FILE)
    print(REGRESSION_FEATURES_FILE)


if __name__ == "__main__":
    main()
