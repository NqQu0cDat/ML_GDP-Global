"""
01_data_cleaning.py

Clean and merge country economic datasets.

Input files:
- data/raw/corruption.csv
- data/raw/cost_of_living.csv
- data/raw/richest_countries.csv
- data/raw/tourism.csv
- data/raw/unemployment.csv

Output file:
- data/processed/country_economic_cleaned.csv
"""

from pathlib import Path
import re

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "country_economic_cleaned.csv"

DATA_FILES = {
    "corruption": "corruption.csv",
    "cost_of_living": "cost_of_living.csv",
    "richest_countries": "richest_countries.csv",
    "tourism": "tourism.csv",
    "unemployment": "unemployment.csv",
}


def clean_column_name(column_name):
    """Convert column names to lowercase snake_case."""
    column_name = str(column_name).strip().lower()
    column_name = re.sub(r"[^a-z0-9]+", "_", column_name)
    column_name = re.sub(r"_+", "_", column_name)
    return column_name.strip("_")


def normalize_country(country):
    """Standardize country text for easier merging."""
    if pd.isna(country):
        return "unknown"

    country = str(country).strip()
    country = re.sub(r"\s+", " ", country)

    country_replacements = {
        "usa": "United States",
        "u.s.a.": "United States",
        "u.s.": "United States",
        "united states of america": "United States",
        "uk": "United Kingdom",
        "u.k.": "United Kingdom",
        "russia": "Russian Federation",
        "south korea": "South Korea",
        "korea, south": "South Korea",
    }

    return country_replacements.get(country.lower(), country)


def country_key(country):
    """Create a lowercase merge key from the standardized country name."""
    country = normalize_country(country)
    return re.sub(r"[^a-z0-9]+", "", country.lower())


def print_dataset_overview(name, df):
    """Print basic information before cleaning."""
    print(f"\n===== {name.upper()} =====")
    print(f"Shape: {df.shape}")
    print("Columns:")
    print(list(df.columns))
    print("Missing values:")
    print(df.isna().sum())


def load_dataset(name, filename):
    """Load one CSV file and apply basic standardization."""
    file_path = RAW_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    df = pd.read_csv(file_path)
    print_dataset_overview(name, df)

    df.columns = [clean_column_name(column) for column in df.columns]

    if "country" not in df.columns:
        raise ValueError(f"{filename} does not contain a 'country' column.")

    df["country"] = df["country"].apply(normalize_country)
    df["country_key"] = df["country"].apply(country_key)

    # Convert numeric-looking columns from text to numbers when needed.
    for column in df.columns:
        if column in ["country", "country_key"]:
            continue

        if df[column].dtype == "object":
            cleaned_column = (
                df[column]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            df[column] = pd.to_numeric(cleaned_column, errors="ignore")

    return df


def merge_datasets(datasets):
    """Merge all datasets by country using an outer join."""
    merged_df = datasets[0]

    for df in datasets[1:]:
        merged_df = pd.merge(
            merged_df,
            df,
            on="country_key",
            how="outer",
            suffixes=("", "_duplicate"),
        )

        country_columns = [column for column in merged_df.columns if column.startswith("country")]
        duplicate_country_columns = [
            column for column in country_columns if column.endswith("_duplicate")
        ]

        for duplicate_column in duplicate_country_columns:
            merged_df["country"] = merged_df["country"].fillna(merged_df[duplicate_column])
            merged_df = merged_df.drop(columns=duplicate_column)

    return merged_df


def fill_missing_values(df):
    """Fill missing numeric values with median and text values with 'unknown'."""
    numeric_columns = df.select_dtypes(include="number").columns
    text_columns = df.select_dtypes(exclude="number").columns

    for column in numeric_columns:
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)

    for column in text_columns:
        df[column] = df[column].fillna("unknown")

    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    datasets = []
    for name, filename in DATA_FILES.items():
        datasets.append(load_dataset(name, filename))

    print("\n===== MERGING DATASETS =====")
    merged_df = merge_datasets(datasets)
    print(f"Merged shape before missing value handling: {merged_df.shape}")
    print("Missing values before filling:")
    print(merged_df.isna().sum())

    cleaned_df = fill_missing_values(merged_df)

    if "country_key" in cleaned_df.columns:
        cleaned_df = cleaned_df.drop(columns="country_key")

    column_order = ["country"] + [column for column in cleaned_df.columns if column != "country"]
    cleaned_df = cleaned_df[column_order]
    cleaned_df = cleaned_df.sort_values("country").reset_index(drop=True)

    cleaned_df.to_csv(OUTPUT_FILE, index=False)

    print("\n===== CLEANED DATASET =====")
    print(f"Final shape: {cleaned_df.shape}")
    print("Final columns:")
    print(list(cleaned_df.columns))
    print("Missing values after filling:")
    print(cleaned_df.isna().sum())
    print(f"\nSaved cleaned data to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
