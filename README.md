# Country Economic Development Analysis and GDP Prediction

## Project Overview

This project analyzes country-level economic indicators and builds a simple machine learning demo to predict GDP per capita.

The main focus of the project is Data Analysis, including data cleaning, exploratory data analysis, visualization, and country clustering. A regression model is included as a lightweight machine learning component to demonstrate an end-to-end analytics workflow.

Project direction:

- 80% Data Analysis
- 20% Machine Learning

The project is designed as a fresher Data Analyst portfolio project. It focuses on clean workflow, readable code, and honest interpretation of model results.

## Dataset Description

The project combines five CSV datasets stored in `data/raw/`:

| File | Description |
| --- | --- |
| `corruption.csv` | Country corruption index and annual income |
| `cost_of_living.csv` | Cost index, monthly income, and purchasing power index |
| `richest_countries.csv` | GDP per capita by country |
| `tourism.csv` | Tourism arrivals, receipts, and tourism contribution to GDP |
| `unemployment.csv` | Unemployment rate by country |

The datasets are merged by country name after standardizing column names and country names.

Cleaned dataset:

- File: `data/processed/country_economic_cleaned.csv`
- Shape: `(167, 12)`
- Missing values after cleaning: `0`

Clustered dataset:

- File: `data/processed/country_economic_clustered.csv`
- Shape: `(167, 13)`
- Added column: `cluster`

## Project Structure

```text
da_machinelearning/
|
|-- app/
|   |-- streamlit_app.py
|
|-- data/
|   |-- raw/
|   |   |-- corruption.csv
|   |   |-- cost_of_living.csv
|   |   |-- richest_countries.csv
|   |   |-- tourism.csv
|   |   |-- unemployment.csv
|   |
|   |-- processed/
|       |-- country_economic_cleaned.csv
|       |-- country_economic_clustered.csv
|
|-- models/
|   |-- kmeans_model.pkl
|   |-- kmeans_scaler.pkl
|   |-- random_forest_gdp_model.pkl
|   |-- regression_features.pkl
|
|-- notebooks/
|   |-- 01_data_cleaning.py
|   |-- 02_eda_analysis.py
|   |-- 03_clustering.py
|   |-- 04_regression_model.py
|
|-- reports/
|   |-- figures/
|
|-- README.md
|-- requirements.txt
```

## Workflow

### 1. Data Cleaning

Script: `notebooks/01_data_cleaning.py`

Main steps:

- Load five raw CSV files
- Print dataset shape, columns, and missing values
- Standardize column names to `snake_case`
- Standardize country names
- Merge datasets using country as the key
- Fill missing numeric values with median
- Fill missing text values with `"unknown"`
- Save cleaned dataset to `data/processed/country_economic_cleaned.csv`

### 2. Exploratory Data Analysis

Script: `notebooks/02_eda_analysis.py`

Main analysis:

- Top countries by GDP per capita
- Annual income vs GDP per capita
- Corruption index vs GDP per capita
- Unemployment rate vs GDP per capita
- Cost index vs GDP per capita
- Purchasing power index vs GDP per capita
- Correlation heatmap

### 3. Clustering

Script: `notebooks/03_clustering.py`

Main steps:

- Select available numeric features
- Scale data using `StandardScaler`
- Use the Elbow Method to test `k = 1` to `10`
- Train KMeans with `k = 3`
- Add cluster labels to the dataset
- Visualize clusters using PCA
- Save KMeans model and scaler

### 4. Regression Modeling

Script: `notebooks/04_regression_model.py`

Target:

- `gdp_per_capita`

Models trained:

- Linear Regression
- Random Forest Regressor

Evaluation metrics:

- MAE
- RMSE
- R2 Score

### 5. Streamlit Demo App

Script: `app/streamlit_app.py`

The app allows users to enter economic indicators and get a demo prediction for GDP per capita using the saved Random Forest model.

## Technologies Used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- Streamlit
- Jupyter

## EDA Outputs

All EDA charts are saved in `reports/figures/`:

```text
top_10_countries_by_gdp_per_capita.png
annual_income_vs_gdp_per_capita.png
corruption_index_vs_gdp_per_capita.png
unemployment_rate_vs_gdp_per_capita.png
cost_index_vs_gdp_per_capita.png
purchasing_power_index_vs_gdp_per_capita.png
correlation_heatmap.png
```

These charts help explore relationships between GDP per capita and income, corruption, unemployment, cost of living, purchasing power, and tourism indicators.

## Clustering Result

KMeans clustering was trained with `k = 3`.

Cluster counts:

| Cluster | Number of Countries |
| --- | ---: |
| 0 | 138 |
| 1 | 26 |
| 2 | 3 |

Clustering outputs:

```text
reports/figures/elbow_method.png
reports/figures/kmeans_clusters_pca.png
models/kmeans_scaler.pkl
models/kmeans_model.pkl
```

The clustering result is used to group countries with similar economic profiles based on selected indicators.

## Regression Model Result

The regression task predicts `gdp_per_capita`.

Model comparison:

| Model | MAE | RMSE | R2 Score |
| --- | ---: | ---: | ---: |
| Linear Regression | 6322.3094 | 11195.3242 | -3.1758 |
| Random Forest Regressor | 2887.6885 | 5411.3112 | 0.0244 |

Random Forest performed better than Linear Regression on this train/test split. However, the R2 Score is still low, so the model should be treated as a learning/demo model rather than a reliable economic forecasting model.

Regression outputs:

```text
reports/figures/linear_regression_actual_vs_predicted.png
reports/figures/random_forest_actual_vs_predicted.png
reports/figures/random_forest_feature_importance.png
models/random_forest_gdp_model.pkl
models/regression_features.pkl
```

## Streamlit Demo

The Streamlit app loads:

```text
models/random_forest_gdp_model.pkl
models/regression_features.pkl
```

The user can enter values for:

- Annual income
- Corruption index
- Cost index
- Monthly income
- Purchasing power index
- Unemployment rate
- Tourists in millions
- Tourism receipts in billions
- Tourism percentage of GDP
- Cluster

Example prediction from a test input:

```text
Predicted GDP per Capita: 37,097.44 USD
```

Important note: this app is a demo for learning and portfolio presentation. It should not be used for real economic forecasting.

## How to Run Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd da_machinelearning
```

### 2. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run data cleaning

```bash
python notebooks/01_data_cleaning.py
```

### 5. Run EDA

```bash
python notebooks/02_eda_analysis.py
```

### 6. Run clustering

```bash
python notebooks/03_clustering.py
```

### 7. Run regression modeling

```bash
python notebooks/04_regression_model.py
```

### 8. Run Streamlit app

```bash
streamlit run app/streamlit_app.py
```

## Limitations

- The datasets are relatively small and come from multiple sources.
- Country names may not perfectly match across all datasets.
- Missing values were filled using median values, which can reduce real-world variation.
- Some countries may have incomplete original data before cleaning.
- The regression model has a low R2 Score, meaning it does not explain GDP per capita well enough for real forecasting.
- Outliers such as very high-income or very tourism-heavy countries can strongly affect both clustering and regression.
- The Streamlit app is for demonstration only.

## Future Improvements

- Use larger and more consistent datasets from official sources.
- Improve country name matching with ISO country codes.
- Add more economic features such as inflation, education, population, trade, and healthcare indicators.
- Compare more regression models and tune hyperparameters.
- Add cross-validation for more stable model evaluation.
- Separate countries by region or income group before modeling.
- Improve Streamlit UI with charts, feature explanations, and sample country presets.
- Add a `requirements-dev.txt` or testing workflow for better project maintenance.

## CV Summary

Built an end-to-end Data Analyst and Machine Learning portfolio project analyzing country-level economic development indicators. Cleaned and merged five datasets, performed exploratory data analysis with visualizations, applied KMeans clustering to group countries by economic profile, trained regression models to predict GDP per capita, and deployed a Streamlit demo app using a saved Random Forest model.

Key skills demonstrated:

- Data cleaning and preprocessing with pandas
- Exploratory data analysis and visualization
- Feature selection and missing value handling
- KMeans clustering and PCA visualization
- Regression modeling and model evaluation
- Model saving/loading with joblib
- Streamlit app development
- Clear project organization for GitHub and CV presentation
