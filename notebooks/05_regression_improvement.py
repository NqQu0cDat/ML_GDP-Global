"""
05_regression_improvement.py

Improve GDP per capita regression using valid preprocessing:
- remove high target outliers with 95% quantile
- apply log transform to the target
- compare several tree-based regression models

Input file:
- data/processed/country_economic_clustered.csv

Output files:
- reports/figures/improved_regression_actual_vs_predicted.png
- reports/figures/improved_regression_feature_importance.png
- models/improved_gdp_model.pkl
- models/improved_regression_features.pkl
"""

from pathlib import Path
import os
import warnings

# Avoid a macOS/joblib physical-core warning when tree models run.
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="joblib.externals.loky.backend.context",
)

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.compose import TransformedTargetRegressor


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "country_economic_clustered.csv"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
MODELS_DIR = BASE_DIR / "models"
IMPROVED_MODEL_FILE = MODELS_DIR / "improved_gdp_model.pkl"
IMPROVED_FEATURES_FILE = MODELS_DIR / "improved_regression_features.pkl"
MATPLOTLIB_CACHE_DIR = BASE_DIR / "reports" / ".matplotlib_cache"

MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import seaborn as sns


TARGET = "gdp_per_capita"

FEATURE_CANDIDATES = [
    "annual_income",
    "monthly_income",
    "purchasing_power_index",
    "cost_index",
    "corruption_index",
    "unemployment_rate",
    "cluster",
]


def get_available_features(df):
    """Use only features that exist in the dataset."""
    available_features = [column for column in FEATURE_CANDIDATES if column in df.columns]
    missing_features = [column for column in FEATURE_CANDIDATES if column not in df.columns]

    print("\n===== FEATURE SELECTION =====")
    print("Features used:")
    print(available_features)

    if missing_features:
        print("\nFeatures skipped because they are not in the dataset:")
        print(missing_features)

    if not available_features:
        raise ValueError("No valid features found for improved regression.")

    return available_features


def remove_target_outliers(df):
    """Keep rows where target is less than or equal to the 95th percentile."""
    percentile_95 = df[TARGET].quantile(0.95)
    filtered_df = df[df[TARGET] <= percentile_95].copy()

    removed_rows = len(df) - len(filtered_df)

    print("\n===== OUTLIER HANDLING =====")
    print(f"95th percentile of {TARGET}: {percentile_95:,.2f}")
    print(f"Rows before filtering: {len(df)}")
    print(f"Rows after filtering: {len(filtered_df)}")
    print(f"Rows removed: {removed_rows}")

    return filtered_df


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics on the original GDP scale."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2,
        "MAPE (%)": mape,
    }


def get_models():
    """Create model candidates for comparison."""
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_depth=8,
            min_samples_leaf=3,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            max_depth=8,
            min_samples_leaf=3,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            random_state=42,
            learning_rate=0.05,
            max_depth=3,
        ),
    }


def train_and_compare_models(models, X_train, X_test, y_train_log, y_test):
    """Train all models on log target and compare predictions on original scale."""
    results = {}
    fitted_models = {}
    predictions = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train_log)

        # Predict log target, then convert back to the original GDP per capita scale.
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_pred = np.maximum(y_pred, 0)

        results[model_name] = calculate_metrics(y_test, y_pred)
        fitted_models[model_name] = model
        predictions[model_name] = y_pred

    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df[["MAE", "RMSE", "R2 Score", "MAPE (%)"]]

    print("\n===== IMPROVED MODEL COMPARISON =====")
    print(comparison_df.round(4))

    best_model_name = comparison_df["R2 Score"].idxmax()

    print("\n===== BEST MODEL =====")
    print(f"Best model based on test R2 Score: {best_model_name}")

    return best_model_name, fitted_models[best_model_name], predictions[best_model_name], comparison_df


def run_cross_validation(model, X, y):
    """Run 5-Fold Cross Validation and evaluate predictions on original scale."""
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold_number, (train_index, test_index) in enumerate(kfold.split(X), start=1):
        X_train_fold = X.iloc[train_index]
        X_test_fold = X.iloc[test_index]
        y_train_fold = y.iloc[train_index]
        y_test_fold = y.iloc[test_index]

        y_train_fold_log = np.log1p(y_train_fold)

        fold_model = clone(model)
        fold_model.fit(X_train_fold, y_train_fold_log)

        y_pred_fold_log = fold_model.predict(X_test_fold)
        y_pred_fold = np.expm1(y_pred_fold_log)
        y_pred_fold = np.maximum(y_pred_fold, 0)

        metrics = calculate_metrics(y_test_fold, y_pred_fold)
        metrics["Fold"] = fold_number
        fold_results.append(metrics)

    cv_df = pd.DataFrame(fold_results).set_index("Fold")

    print("\n===== 5-FOLD CROSS VALIDATION =====")
    print(cv_df.round(4))

    print("\n===== 5-FOLD CROSS VALIDATION AVERAGE =====")
    print(cv_df.mean().round(4))

    return cv_df


def plot_actual_vs_predicted(y_true, y_pred, model_name):
    """Save actual vs predicted plot for the best model."""
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

    plt.title(f"{model_name}: Actual vs Predicted GDP per Capita")
    plt.xlabel("Actual GDP per Capita")
    plt.ylabel("Predicted GDP per Capita")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "improved_regression_actual_vs_predicted.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_feature_importance(model, features, model_name):
    """Save feature importance plot if the model supports feature_importances_."""
    if not hasattr(model, "feature_importances_"):
        print("\nBest model does not provide feature_importances_. Skipping plot.")
        return

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
    plt.title(f"{model_name}: Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "improved_regression_feature_importance.png",
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
    filtered_df = remove_target_outliers(df)

    X = filtered_df[features]
    y = filtered_df[TARGET]
    y_log = np.log1p(y)

    X_train, X_test, y_train_log, _, _, y_test = train_test_split(
        X,
        y_log,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("\n===== TRAIN TEST SPLIT =====")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    models = get_models()
    best_model_name, best_model, best_predictions, comparison_df = train_and_compare_models(
        models=models,
        X_train=X_train,
        X_test=X_test,
        y_train_log=y_train_log,
        y_test=y_test,
    )

    run_cross_validation(best_model, X, y)

    plot_actual_vs_predicted(y_test, best_predictions, best_model_name)
    plot_feature_importance(best_model, features, best_model_name)

    # Save a final model that handles log transform internally.
    # This means future predictions from the saved model are already in GDP scale.
    final_model = TransformedTargetRegressor(
        regressor=clone(best_model),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    final_model.fit(X, y)

    joblib.dump(final_model, IMPROVED_MODEL_FILE)
    joblib.dump(features, IMPROVED_FEATURES_FILE)

    print("\n===== SAVED FILES =====")
    print(FIGURES_DIR / "improved_regression_actual_vs_predicted.png")
    print(FIGURES_DIR / "improved_regression_feature_importance.png")
    print(IMPROVED_MODEL_FILE)
    print(IMPROVED_FEATURES_FILE)


if __name__ == "__main__":
    main()
