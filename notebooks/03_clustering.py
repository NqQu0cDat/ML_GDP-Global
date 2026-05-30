"""
03_clustering.py

Cluster countries based on economic indicators using KMeans.

Input file:
- data/processed/country_economic_cleaned.csv

Output files:
- data/processed/country_economic_clustered.csv
- reports/figures/elbow_method.png
- reports/figures/kmeans_clusters_pca.png
- models/kmeans_scaler.pkl
- models/kmeans_model.pkl
"""

from pathlib import Path
import os
import warnings

# Avoid a macOS/joblib physical-core warning when KMeans runs in some terminals.
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="joblib.externals.loky.backend.context",
)

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "country_economic_cleaned.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "country_economic_clustered.csv"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
MODELS_DIR = BASE_DIR / "models"
SCALER_FILE = MODELS_DIR / "kmeans_scaler.pkl"
KMEANS_MODEL_FILE = MODELS_DIR / "kmeans_model.pkl"
MATPLOTLIB_CACHE_DIR = BASE_DIR / "reports" / ".matplotlib_cache"

MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import seaborn as sns


FEATURE_CANDIDATES = [
    "gdp_per_capita",
    "annual_income",
    "corruption_index",
    "cost_index",
    "monthly_income",
    "purchasing_power_index",
    "unemployment_rate",
    "tourists_in_millions",
    "receipts_in_billions",
    "percentage_of_gdp",
]


def get_available_features(df):
    """Use only clustering features that exist in the dataframe."""
    available_features = [column for column in FEATURE_CANDIDATES if column in df.columns]
    missing_features = [column for column in FEATURE_CANDIDATES if column not in df.columns]

    print("\n===== FEATURE SELECTION =====")
    print("Features used for clustering:")
    print(available_features)

    if missing_features:
        print("\nFeatures skipped because they are not in the dataset:")
        print(missing_features)

    if len(available_features) < 2:
        raise ValueError("Need at least 2 numeric features for clustering.")

    return available_features


def scale_features(df, features):
    """Scale numeric features so each feature has similar importance."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    return scaler, scaled_data


def plot_elbow_method(scaled_data):
    """Run KMeans for k=1 to k=10 and save the elbow chart."""
    k_values = range(1, 11)
    inertias = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=list(k_values), y=inertias, marker="o")
    plt.title("Elbow Method for Choosing Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(list(k_values))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "elbow_method.png", dpi=300, bbox_inches="tight")
    plt.close()


def train_kmeans(scaled_data, n_clusters=3):
    """Train KMeans model with the selected number of clusters."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    return kmeans, cluster_labels


def print_cluster_summary(df, features):
    """Print useful information to understand each cluster."""
    print("\n===== CLUSTER COUNTS =====")
    print(df["cluster"].value_counts().sort_index())

    print("\n===== CLUSTER AVERAGES =====")
    print(df.groupby("cluster")[features].mean().round(2))

    print("\n===== SAMPLE COUNTRIES BY CLUSTER =====")
    for cluster_id in sorted(df["cluster"].unique()):
        sample_countries = (
            df[df["cluster"] == cluster_id]["country"]
            .sort_values()
            .head(10)
            .tolist()
        )
        print(f"Cluster {cluster_id}: {', '.join(sample_countries)}")


def plot_pca_clusters(df, scaled_data):
    """Reduce scaled features to 2D with PCA and save the cluster scatter plot."""
    pca = PCA(n_components=2, random_state=42)
    pca_data = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(
        {
            "pca_1": pca_data[:, 0],
            "pca_2": pca_data[:, 1],
            "cluster": df["cluster"].astype(str),
            "country": df["country"],
        }
    )

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=pca_df,
        x="pca_1",
        y="pca_2",
        hue="cluster",
        palette="Set2",
        s=80,
        alpha=0.85,
    )
    plt.title("KMeans Country Clusters Visualized with PCA")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kmeans_clusters_pca.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n===== PCA EXPLAINED VARIANCE =====")
    print(f"PCA Component 1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"PCA Component 2: {pca.explained_variance_ratio_[1]:.2%}")


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing cleaned dataset: {DATA_FILE}\n"
            "Please run notebooks/01_data_cleaning.py first."
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    print("\n===== DATASET LOADED =====")
    print(f"Shape: {df.shape}")

    features = get_available_features(df)
    scaler, scaled_data = scale_features(df, features)

    plot_elbow_method(scaled_data)

    kmeans, cluster_labels = train_kmeans(scaled_data, n_clusters=3)
    df["cluster"] = cluster_labels

    print_cluster_summary(df, features)
    plot_pca_clusters(df, scaled_data)

    df.to_csv(OUTPUT_FILE, index=False)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(kmeans, KMEANS_MODEL_FILE)

    print("\n===== SAVED FILES =====")
    print(OUTPUT_FILE)
    print(FIGURES_DIR / "elbow_method.png")
    print(FIGURES_DIR / "kmeans_clusters_pca.png")
    print(SCALER_FILE)
    print(KMEANS_MODEL_FILE)


if __name__ == "__main__":
    main()
