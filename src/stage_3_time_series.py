"""
Stage 3 -- Classical Time-Series Models
Dataset Type: Synthetic
SARIMA, ETS (Holt-Winters), Prophet for each location.
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

def split_data(df, loc):
    d = df[df["Location"] == loc].sort_values("Month").copy()
    d = d.set_index("Month")
    d.index.freq = "MS"
    train = d[d.index <= TRAIN_END]["Admissions"]
    test  = d[(d.index >= TEST_START) & (d.index <= TEST_END)]["Admissions"]
    return train, test

# ── SARIMA ──
def fit_sarima(train, test, loc):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    best_aic = np.inf
    best_order = (1, 1, 1)
    best_seasonal = (1, 1, 1, 12)
    
    # Grid search over a small set of parameters
    for p in [0, 1, 2]:
        for d in [0, 1]:
            for q in [0, 1, 2]:
                for P in [0, 1]:
                    for D in [0, 1]:
                        for Q in [0, 1]:
                            try:
                                model = SARIMAX(train,
                                                order=(p, d, q),
                                                seasonal_order=(P, D, Q, 12),
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                                result = model.fit(disp=False, maxiter=100)
                                if result.aic < best_aic:
                                    best_aic = result.aic
                                    best_order = (p, d, q)
                                    best_seasonal = (P, D, Q, 12)
                            except:
                                continue
    
    # Fit best model
    model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal,
                    enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False, maxiter=200)
    
    # Predict test period
    pred = result.get_forecast(steps=len(test))
    pred_mean = pred.predicted_mean
    pred_ci = pred.conf_int()
    
    print(f"    SARIMA{best_order}x{best_seasonal} AIC={best_aic:.0f}")
    return result, pred_mean, pred_ci, best_order, best_seasonal

# ── ETS (Holt-Winters) ──
def fit_ets(train, test, loc):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    best_model = None
    best_aic = np.inf
    best_config = ""
    
    for trend in ["add", "mul", None]:
        for seasonal in ["add", "mul"]:
            for damped in [True, False]:
                if trend is None and damped:
                    continue
                try:
                    model = ExponentialSmoothing(
                        train, trend=trend, seasonal=seasonal,
                        seasonal_periods=12, damped_trend=damped,
                        initialization_method="estimated"
                    )
                    result = model.fit(optimized=True, use_brute=False)
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_model = result
                        best_config = f"trend={trend},seasonal={seasonal},damped={damped}"
                except:
                    continue
    
    pred_mean = best_model.forecast(len(test))
    pred_mean.index = test.index
    
    print(f"    ETS [{best_config}] AIC={best_aic:.0f}")
    return best_model, pred_mean, best_config

# ── Prophet ──
def fit_prophet(train, test, loc):
    from prophet import Prophet
    
    # Prepare data
    df_train = train.reset_index()
    df_train.columns = ["ds", "y"]
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode="multiplicative",
                    changepoint_prior_scale=0.05)
    model.fit(df_train)
    
    future = pd.DataFrame({"ds": test.index})
    forecast = model.predict(future)
    pred_mean = pd.Series(forecast["yhat"].values, index=test.index)
    pred_lower = pd.Series(forecast["yhat_lower"].values, index=test.index)
    pred_upper = pd.Series(forecast["yhat_upper"].values, index=test.index)
    
    print(f"    Prophet fitted")
    return model, pred_mean, pred_lower, pred_upper

def main():
    print("=" * 60)
    print(f"STAGE 3 -- CLASSICAL TIME-SERIES MODELS  |  {WATERMARK}")
    print("=" * 60)
    
    monthly = load_monthly()
    all_results = []
    
    for loc in LOCATION_ORDER:
        print(f"\n--- {loc} ---")
        train, test = split_data(monthly, loc)
        print(f"  Train: {len(train)} months, Test: {len(test)} months")
        
        if len(train) < 24 or len(test) == 0:
            print(f"  [SKIP] Not enough data for {loc}")
            continue
        
        loc_results = {"Location": loc, "test_index": test.index, "test_actual": test.values}
        
        # SARIMA
        print("  Fitting SARIMA ...")
        try:
            sarima_model, sarima_pred, sarima_ci, sarima_order, sarima_seasonal = fit_sarima(train, test, loc)
            loc_results["sarima_pred"] = sarima_pred.values
            loc_results["sarima_order"] = str(sarima_order)
            loc_results["sarima_seasonal"] = str(sarima_seasonal)
            # Save model
            with open(os.path.join(MODELS_DIR, f"sarima_{loc}.pkl"), "wb") as f:
                pickle.dump(sarima_model, f)
        except Exception as e:
            print(f"  [ERROR] SARIMA failed: {e}")
            loc_results["sarima_pred"] = np.full(len(test), np.nan)
        
        # ETS
        print("  Fitting ETS ...")
        try:
            ets_model, ets_pred, ets_config = fit_ets(train, test, loc)
            loc_results["ets_pred"] = ets_pred.values
            loc_results["ets_config"] = ets_config
            with open(os.path.join(MODELS_DIR, f"ets_{loc}.pkl"), "wb") as f:
                pickle.dump(ets_model, f)
        except Exception as e:
            print(f"  [ERROR] ETS failed: {e}")
            loc_results["ets_pred"] = np.full(len(test), np.nan)
        
        # Prophet
        print("  Fitting Prophet ...")
        try:
            prophet_model, prophet_pred, prophet_lower, prophet_upper = fit_prophet(train, test, loc)
            loc_results["prophet_pred"] = prophet_pred.values
            with open(os.path.join(MODELS_DIR, f"prophet_{loc}.pkl"), "wb") as f:
                pickle.dump(prophet_model, f)
        except Exception as e:
            print(f"  [ERROR] Prophet failed: {e}")
            loc_results["prophet_pred"] = np.full(len(test), np.nan)
        
        all_results.append(loc_results)
        
        # Plot: Actual vs predictions
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(test.index, test.values, "o-", color="white", linewidth=2, label="Actual", markersize=5)
        
        if not np.all(np.isnan(loc_results.get("sarima_pred", [np.nan]))):
            ax.plot(test.index, loc_results["sarima_pred"], "s--", color="#6366F1",
                    linewidth=1.5, label="SARIMA", markersize=4)
        if not np.all(np.isnan(loc_results.get("ets_pred", [np.nan]))):
            ax.plot(test.index, loc_results["ets_pred"], "^--", color="#EC4899",
                    linewidth=1.5, label="ETS", markersize=4)
        if not np.all(np.isnan(loc_results.get("prophet_pred", [np.nan]))):
            ax.plot(test.index, loc_results["prophet_pred"], "D--", color="#10B981",
                    linewidth=1.5, label="Prophet", markersize=4)
        
        ax.set_title(f"Time-Series Model Predictions -- {loc} (Jan-Jun 2026)", pad=12)
        ax.set_ylabel("Monthly Admissions")
        ax.legend(framealpha=0.3)
        ax.grid(True, alpha=0.3)
        stamp(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(FORECASTS_DIR, f"ts_validation_{loc}.png"))
        plt.close(fig)
    
    # Save all predictions to CSV
    rows = []
    for r in all_results:
        for i, idx in enumerate(r["test_index"]):
            row = {
                "Location": r["Location"],
                "Month": idx,
                "Actual": r["test_actual"][i],
                "SARIMA": r.get("sarima_pred", [np.nan]*6)[i] if i < len(r.get("sarima_pred", [])) else np.nan,
                "ETS": r.get("ets_pred", [np.nan]*6)[i] if i < len(r.get("ets_pred", [])) else np.nan,
                "Prophet": r.get("prophet_pred", [np.nan]*6)[i] if i < len(r.get("prophet_pred", [])) else np.nan,
            }
            rows.append(row)
    
    pd.DataFrame(rows).to_csv(os.path.join(FORECASTS_DIR, "ts_validation_results.csv"), index=False)
    print(f"\n[OK] Stage 3 complete -- results saved to {FORECASTS_DIR}")

if __name__ == "__main__":
    main()
