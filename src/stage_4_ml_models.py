"""
Stage 4 -- Machine Learning Models
Dataset Type: Synthetic
XGBoost, Random Forest, Linear Regression for each location.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (DATA_CLEAN_DIR, MODELS_DIR, FORECASTS_DIR, COLORS,
                         LOCATION_ORDER, PLOT_STYLE, WATERMARK,
                         TRAIN_END, TEST_START, TEST_END)

plt.rcParams.update(PLOT_STYLE)

def stamp(ax):
    ax.text(0.99, 0.01, WATERMARK, transform=ax.transAxes, fontsize=7,
            color="#64748B", ha="right", va="bottom", alpha=0.7)

def load_monthly():
    df = pd.read_csv(os.path.join(DATA_CLEAN_DIR, "monthly_admissions.csv"),
                     parse_dates=["Month"])
    return df

def create_features(df):
    """Create ML features from monthly time series."""
    df = df.copy()
    df["Year"] = df["Month"].dt.year
    df["MonthNum"] = df["Month"].dt.month
    df["Quarter"] = df["Month"].dt.quarter
    
    # Sine/cosine seasonal encoding
    df["Month_sin"] = np.sin(2 * np.pi * df["MonthNum"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["MonthNum"] / 12)
    df["Quarter_sin"] = np.sin(2 * np.pi * df["Quarter"] / 4)
    df["Quarter_cos"] = np.cos(2 * np.pi * df["Quarter"] / 4)
    
    # Time index (months since start)
    df["TimeIndex"] = np.arange(len(df))
    
    # Lag features
    for lag in [1, 2, 3, 6, 12]:
        df[f"Lag_{lag}"] = df["Admissions"].shift(lag)
    
    # Rolling features
    df["Roll_3"] = df["Admissions"].rolling(3).mean()
    df["Roll_6"] = df["Admissions"].rolling(6).mean()
    df["Roll_3_std"] = df["Admissions"].rolling(3).std()
    
    return df

FEATURE_COLS = ["Year", "MonthNum", "Quarter", "Month_sin", "Month_cos",
                "Quarter_sin", "Quarter_cos", "TimeIndex",
                "Lag_1", "Lag_2", "Lag_3", "Lag_6", "Lag_12",
                "Roll_3", "Roll_6", "Roll_3_std"]

def prepare_data(monthly, loc):
    """Prepare train/test splits for a location."""
    d = monthly[monthly["Location"] == loc].sort_values("Month").copy()
    d = d.reset_index(drop=True)
    d = create_features(d)
    d = d.dropna()  # Drop rows with NaN from lags
    
    train = d[d["Month"] <= TRAIN_END]
    test = d[(d["Month"] >= TEST_START) & (d["Month"] <= TEST_END)]
    
    X_train = train[FEATURE_COLS]
    y_train = train["Admissions"]
    X_test = test[FEATURE_COLS]
    y_test = test["Admissions"]
    test_months = test["Month"]
    
    return X_train, y_train, X_test, y_test, test_months

def fit_xgboost(X_train, y_train, X_test, y_test):
    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, pred

def fit_random_forest(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_split=3,
        min_samples_leaf=2, random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, pred

def fit_linear_regression(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, pred

def main():
    print("=" * 60)
    print(f"STAGE 4 -- MACHINE LEARNING MODELS  |  {WATERMARK}")
    print("=" * 60)
    
    monthly = load_monthly()
    all_results = []
    
    for loc in LOCATION_ORDER:
        print(f"\n--- {loc} ---")
        
        X_train, y_train, X_test, y_test, test_months = prepare_data(monthly, loc)
        print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        if len(X_test) == 0:
            print(f"  [SKIP] No test data for {loc}")
            continue
        
        loc_results = {
            "Location": loc,
            "test_months": test_months.values,
            "test_actual": y_test.values,
        }
        
        # XGBoost
        print("  Fitting XGBoost ...")
        try:
            xgb_model, xgb_pred = fit_xgboost(X_train, y_train, X_test, y_test)
            loc_results["xgboost_pred"] = xgb_pred
            with open(os.path.join(MODELS_DIR, f"xgboost_{loc}.pkl"), "wb") as f:
                pickle.dump(xgb_model, f)
            print(f"    XGBoost fitted")
        except Exception as e:
            print(f"  [ERROR] XGBoost failed: {e}")
            loc_results["xgboost_pred"] = np.full(len(y_test), np.nan)
        
        # Random Forest
        print("  Fitting Random Forest ...")
        try:
            rf_model, rf_pred = fit_random_forest(X_train, y_train, X_test, y_test)
            loc_results["rf_pred"] = rf_pred
            with open(os.path.join(MODELS_DIR, f"rf_{loc}.pkl"), "wb") as f:
                pickle.dump(rf_model, f)
            print(f"    Random Forest fitted")
        except Exception as e:
            print(f"  [ERROR] RF failed: {e}")
            loc_results["rf_pred"] = np.full(len(y_test), np.nan)
        
        # Linear Regression (Ridge)
        print("  Fitting Linear Regression ...")
        try:
            lr_model, lr_pred = fit_linear_regression(X_train, y_train, X_test, y_test)
            loc_results["lr_pred"] = lr_pred
            with open(os.path.join(MODELS_DIR, f"lr_{loc}.pkl"), "wb") as f:
                pickle.dump(lr_model, f)
            print(f"    Linear Regression fitted")
        except Exception as e:
            print(f"  [ERROR] LR failed: {e}")
            loc_results["lr_pred"] = np.full(len(y_test), np.nan)
        
        all_results.append(loc_results)
        
        # Plot: Actual vs ML predictions
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(test_months, y_test.values, "o-", color="white", linewidth=2,
                label="Actual", markersize=5)
        
        if not np.all(np.isnan(loc_results.get("xgboost_pred", [np.nan]))):
            ax.plot(test_months, loc_results["xgboost_pred"], "s--", color="#F59E0B",
                    linewidth=1.5, label="XGBoost", markersize=4)
        if not np.all(np.isnan(loc_results.get("rf_pred", [np.nan]))):
            ax.plot(test_months, loc_results["rf_pred"], "^--", color="#06B6D4",
                    linewidth=1.5, label="Random Forest", markersize=4)
        if not np.all(np.isnan(loc_results.get("lr_pred", [np.nan]))):
            ax.plot(test_months, loc_results["lr_pred"], "D--", color="#EC4899",
                    linewidth=1.5, label="Linear Regression", markersize=4)
        
        ax.set_title(f"ML Model Predictions -- {loc} (Jan-Jun 2026)", pad=12)
        ax.set_ylabel("Monthly Admissions")
        ax.legend(framealpha=0.3)
        ax.grid(True, alpha=0.3)
        stamp(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(FORECASTS_DIR, f"ml_validation_{loc}.png"))
        plt.close(fig)
    
    # Save all predictions to CSV
    rows = []
    for r in all_results:
        for i, month in enumerate(r["test_months"]):
            row = {
                "Location": r["Location"],
                "Month": month,
                "Actual": r["test_actual"][i],
                "XGBoost": r.get("xgboost_pred", [np.nan])[i] if i < len(r.get("xgboost_pred", [])) else np.nan,
                "RandomForest": r.get("rf_pred", [np.nan])[i] if i < len(r.get("rf_pred", [])) else np.nan,
                "LinearRegression": r.get("lr_pred", [np.nan])[i] if i < len(r.get("lr_pred", [])) else np.nan,
            }
            rows.append(row)
    
    pd.DataFrame(rows).to_csv(os.path.join(FORECASTS_DIR, "ml_validation_results.csv"), index=False)
    print(f"\n[OK] Stage 4 complete -- results saved to {FORECASTS_DIR}")

if __name__ == "__main__":
    main()
