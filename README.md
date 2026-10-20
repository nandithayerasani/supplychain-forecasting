# Supply Chain Demand Forecasting

A data science capstone project that forecasts weekly retail demand (`Sales Quantity`) per store-product combination. It compares three tree-based models (Random Forest, XGBoost, LightGBM) against an existing baseline forecast, and includes an interactive Streamlit dashboard.

## Dataset

`supply_chain_demand_forecasting_v3.xlsx`, synthetic dataset created using Claude

- **26,250 weekly observations**, January 2022 – December 2023
- 10 stores, 25 products (5 per category), 5 categories, 5 regions
- Columns include unit and competitor price, starting and ending inventory, promotion, holiday and stockout flags, temperature, precipitation, and a `Baseline Forecast`
- No missing values

**Target:** `Sales Quantity`, the weekly units sold per store-product.

> `Stockout Flag` is 0 in every row, so stockout classification isn't feasible on this data. Stockout risk is inferred by comparing forecasted demand with inventory levels instead.

## Project Structure

| File | Description |
|------|-------------|
| `supply_chain_demand_forecasting_v3.xlsx` | Source dataset |
| `randomForest.ipynb` | Random Forest pipeline: regression on sales quantity, plus a stockout classifier with fixes for data leakage |
| `xgBoost.ipynb` | EDA plus XGBoost regression, permutation importance, and `RandomizedSearchCV` tuning with `TimeSeriesSplit` |
| `lightGBM.ipynb` | EDA, feature engineering, LightGBM modeling and tuning, SHAP interpretability, and business interpretation |
| `dashboard.py` | Streamlit dashboard with EDA findings and an interactive LightGBM forecaster |

## Methodology

### Feature Engineering
- **Calendar features:** quarter, year-end and Q1 flags, sine/cosine encodings of month and week of year
- **Lag features:** sales 1, 2, 4 and 52 weeks back
- **Rolling statistics:** 4- and 12-week rolling mean and standard deviation, 4-week rolling max, 4-week exponentially weighted mean
- **Promotion and holiday features:** last week's promotion flag, promotion × holiday interaction
- **Price and inventory features:** price relative to the category average, inventory coverage, last week's stockout flag
- **Categorical features:** category, region, store ID, product ID

All lag and rolling features are computed per store-product and shifted so the model never sees the current week's sales.

### Train / Validation / Test Split (time-based)
| Split | Period |
|-------|--------|
| Train | before 2023-07-01 |
| Validation | 2023-07-01 to 2023-10-31 |
| Test | 2023-11-01 onward |

### Evaluation Metrics
MAE, RMSE, R², and MAPE, each compared against the provided `Baseline Forecast` (baseline MAE ≈ 29.76 units, MAPE ≈ 37.7%).

## Key Findings
1. LightGBM beats the baseline forecast on every metric.
2. Lag and rolling features are the strongest predictors: recent sales history carries the most signal.
3. Promotions (about 54% lift) and holidays (about 39% lift) have real effects on demand.
4. Weather features (temperature, precipitation) add little predictive value.
5. Demand rises in November–December, which the `is_year_end` feature partly captures.

## Dashboard

`dashboard.py` is a Streamlit app with two tabs:

1. **EDA Findings:** sales distribution, weekly trends, category breakdown, seasonality heatmap, promotion and holiday impact, regional and store sales, and a correlation matrix.
2. **Forecasting Model:** LightGBM vs. baseline performance, top-15 feature importance, an interactive forecast explorer filtered by category, region, store and product, and a **Future Demand Predictor** that forecasts upcoming weeks from user-set promotion, holiday and weather inputs.

The model is trained when the app starts, and the result is cached.

## Getting Started

### Requirements
Python 3.9+ and these packages:

```bash
pip install pandas numpy scikit-learn lightgbm xgboost streamlit plotly matplotlib seaborn scipy openpyxl shap joblib
```

### Run the dashboard
Keep `supply_chain_demand_forecasting_v3.xlsx` in the same folder as `dashboard.py`, then run:

```bash
streamlit run dashboard.py
```

### Run the notebooks
Open the notebooks in Jupyter or Google Colab. Each one loads the dataset from a hard-coded path (a local path or a Google Drive mount). Change `file_path` near the top of each notebook to point to your copy of the `.xlsx` file.

## Team
| Model | Owner |
|-------|-------|
| Random Forest | Nandini |
| XGBoost | Nanditha |
| LightGBM | Narain |
