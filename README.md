# Multi-Hospital Admission Forecasting and Decision Support

**Dataset Type: Synthetic**

A time-series and machine-learning study across six hospital locations.

## Important Notice

- All data used in this project is **synthetic** and does not contain real patient information.
- Results are based on synthetic data and should **not** be interpreted as identical to results from original data.
- No hospital names, facility codes, or confidential identifiers appear anywhere in this project.

## Locations

| Label | Description |
|-------|-------------|
| Abu_Dhabi_1 | Location 1 |
| Abu_Dhabi_2 | Location 2 |
| Al_Ain | Location 3 |
| Musaffah | Location 4 |
| Al_Reem | Location 5 |
| Asharej | Location 6 |

## Project Structure

```
Project/
├── data/                    # Cleaned intermediate CSVs
├── outputs/
│   ├── eda/                 # 10 exploratory analysis charts
│   ├── models/              # Saved model artifacts (.pkl)
│   ├── forecasts/           # Forecast CSVs and validation charts
│   ├── reports/             # Excel summary workbook
│   └── dashboard/           # Interactive HTML dashboard
├── src/
│   ├── config.py            # Central configuration
│   ├── stage_1_data_prep.py # Data loading and aggregation
│   ├── stage_2_eda.py       # Exploratory Data Analysis
│   ├── stage_3_time_series.py # SARIMA, ETS, Prophet
│   ├── stage_4_ml_models.py # XGBoost, Random Forest, Ridge
│   ├── stage_5_evaluation.py # Model comparison metrics
│   ├── stage_6_forecasts.py # 6-month future forecasts
│   ├── stage_7_reports.py   # Excel report generation
│   └── stage_8_dashboard.py # Interactive HTML dashboard
├── README.md
└── requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt

python src/stage_1_data_prep.py
python src/stage_2_eda.py
python src/stage_3_time_series.py
python src/stage_4_ml_models.py
python src/stage_5_evaluation.py
python src/stage_6_forecasts.py
python src/stage_7_reports.py
python src/stage_8_dashboard.py
```

## Models Used

### Time-Series Models
- **SARIMA** — Seasonal ARIMA with AIC-based parameter selection
- **ETS** — Exponential Smoothing (Holt-Winters) with additive/multiplicative variants
- **Prophet** — Facebook Prophet with yearly seasonality

### Machine Learning Models
- **XGBoost** — Gradient boosted trees regressor
- **Random Forest** — Ensemble of decision trees
- **Linear Regression** — Ridge regression (baseline)

## Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)
