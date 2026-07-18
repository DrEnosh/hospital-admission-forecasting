"""
Stage 6 -- Future Forecasts (Jul-Dec 2026)
Dataset Type: Synthetic
Uses the best model per location to generate 6-month forecasts.
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
                         LOCATION_ORDER, PLOT_STYLE, WATERMARK, FORECAST_MONTHS)

plt.rcParams.update(PLOT_STYLE)

def stamp(ax):
    ax.text(0.99, 0.01, WATERMARK, transform=ax.transAxes, fontsize=7,
            color="#64748B", ha="right", va="bottom", alpha=0.7)

def load_monthly():
    return pd.read_csv(os.path.join(DATA_CLEAN_DIR, "monthly_admissions.csv"),
                       parse_dates=["Month"])

def forecast_sarima(loc, monthly, n_months=6):
    model_path = os.path.join(MODELS_DIR, f"sarima_{loc}.pkl")
    if not os.path.exists(model_path):
        return None, None, None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Re-fit on all available data
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    d = monthly[monthly["Location"] == loc].sort_values("Month").set_index("Month")
    d.index.freq = "MS"
    
    refit = SARIMAX(d["Admissions"],
                    order=model.specification["order"],
                    seasonal_order=model.specification["seasonal_order"],
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    result = refit.fit(disp=False, maxiter=200)
    
    forecast = result.get_forecast(steps=n_months)
    pred = forecast.predicted_mean
    ci = forecast.conf_int()
    return pred, ci.iloc[:, 0], ci.iloc[:, 1]

def forecast_ets(loc, monthly, n_months=6):
    model_path = os.path.join(MODELS_DIR, f"ets_{loc}.pkl")
    if not os.path.exists(model_path):
        return None, None, None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    d = monthly[monthly["Location"] == loc].sort_values("Month").set_index("Month")
    d.index.freq = "MS"
    
    refit = ExponentialSmoothing(
        d["Admissions"],
        trend=model.model.trend if hasattr(model.model, 'trend') else "add",
        seasonal=model.model.seasonal if hasattr(model.model, 'seasonal') else "add",
        seasonal_periods=12,
        damped_trend=model.model.damped_trend if hasattr(model.model, 'damped_trend') else False,
        initialization_method="estimated"
    )
    result = refit.fit(optimized=True, use_brute=False)
    pred = result.forecast(n_months)
    return pred, None, None

def forecast_prophet(loc, monthly, n_months=6):
    model_path = os.path.join(MODELS_DIR, f"prophet_{loc}.pkl")
    if not os.path.exists(model_path):
        return None, None, None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    future_dates = pd.date_range("2026-07-01", periods=n_months, freq="MS")
    future = pd.DataFrame({"ds": future_dates})
    forecast = model.predict(future)
    pred = pd.Series(forecast["yhat"].values, index=future_dates)
    lower = pd.Series(forecast["yhat_lower"].values, index=future_dates)
    upper = pd.Series(forecast["yhat_upper"].values, index=future_dates)
    return pred, lower, upper

def forecast_ml(loc, monthly, model_type, n_months=6):
    model_path = os.path.join(MODELS_DIR, f"{model_type}_{loc}.pkl")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    d = monthly[monthly["Location"] == loc].sort_values("Month").copy().reset_index(drop=True)
    
    future_dates = pd.date_range("2026-07-01", periods=n_months, freq="MS")
    preds = []
    
    # Iterative forecasting (use own predictions as lags)
    hist = d["Admissions"].values.tolist()
    
    for i, fd in enumerate(future_dates):
        features = {}
        features["Year"] = fd.year
        features["MonthNum"] = fd.month
        features["Quarter"] = (fd.month - 1) // 3 + 1
        features["Month_sin"] = np.sin(2 * np.pi * fd.month / 12)
        features["Month_cos"] = np.cos(2 * np.pi * fd.month / 12)
        features["Quarter_sin"] = np.sin(2 * np.pi * features["Quarter"] / 4)
        features["Quarter_cos"] = np.cos(2 * np.pi * features["Quarter"] / 4)
        features["TimeIndex"] = len(d) + i
        
        n = len(hist)
        features["Lag_1"] = hist[n-1] if n >= 1 else 0
        features["Lag_2"] = hist[n-2] if n >= 2 else 0
        features["Lag_3"] = hist[n-3] if n >= 3 else 0
        features["Lag_6"] = hist[n-6] if n >= 6 else 0
        features["Lag_12"] = hist[n-12] if n >= 12 else 0
        features["Roll_3"] = np.mean(hist[-3:]) if n >= 3 else np.mean(hist)
        features["Roll_6"] = np.mean(hist[-6:]) if n >= 6 else np.mean(hist)
        features["Roll_3_std"] = np.std(hist[-3:]) if n >= 3 else np.std(hist)
        
        X = pd.DataFrame([features])
        pred = model.predict(X)[0]
        preds.append(max(0, pred))  # Ensure non-negative
        hist.append(pred)
    
    return pd.Series(preds, index=future_dates)

def main():
    print("=" * 60)
    print(f"STAGE 6 -- FUTURE FORECASTS (Jul-Dec 2026)  |  {WATERMARK}")
    print("=" * 60)
    
    monthly = load_monthly()
    
    # Load best models
    best_models_path = os.path.join(FORECASTS_DIR, "best_models.csv")
    if os.path.exists(best_models_path):
        best_df = pd.read_csv(best_models_path)
        best_map = dict(zip(best_df["Location"], best_df["BestModel"]))
    else:
        best_map = {loc: "SARIMA" for loc in LOCATION_ORDER}
    
    future_dates = pd.date_range("2026-07-01", periods=FORECAST_MONTHS, freq="MS")
    all_forecasts = []
    all_forecast_data = {}
    
    for loc in LOCATION_ORDER:
        print(f"\n--- {loc} ---")
        best = best_map.get(loc, "SARIMA")
        print(f"  Best model: {best}")
        
        forecasts = {}
        ci_data = {}
        
        # Generate forecasts from all available models
        # SARIMA
        try:
            pred, lower, upper = forecast_sarima(loc, monthly)
            if pred is not None:
                forecasts["SARIMA"] = pred.values
                if lower is not None:
                    ci_data["SARIMA"] = (lower.values, upper.values)
                print(f"  SARIMA forecast: {[f'{v:.0f}' for v in pred.values]}")
        except Exception as e:
            print(f"  SARIMA forecast error: {e}")
        
        # ETS
        try:
            pred, _, _ = forecast_ets(loc, monthly)
            if pred is not None:
                forecasts["ETS"] = pred.values
                print(f"  ETS forecast: {[f'{v:.0f}' for v in pred.values]}")
        except Exception as e:
            print(f"  ETS forecast error: {e}")
        
        # Prophet
        try:
            pred, lower, upper = forecast_prophet(loc, monthly)
            if pred is not None:
                forecasts["Prophet"] = pred.values
                if lower is not None:
                    ci_data["Prophet"] = (lower.values, upper.values)
                print(f"  Prophet forecast: {[f'{v:.0f}' for v in pred.values]}")
        except Exception as e:
            print(f"  Prophet forecast error: {e}")
        
        # ML models
        for ml_name, ml_key in [("XGBoost", "xgboost"), ("RandomForest", "rf"), ("LinearRegression", "lr")]:
            try:
                pred = forecast_ml(loc, monthly, ml_key)
                if pred is not None:
                    forecasts[ml_name] = pred.values
                    print(f"  {ml_name} forecast: {[f'{v:.0f}' for v in pred.values]}")
            except Exception as e:
                print(f"  {ml_name} forecast error: {e}")
        
        all_forecast_data[loc] = {"forecasts": forecasts, "ci": ci_data, "best": best}
        
        # Build forecast rows
        for i, fd in enumerate(future_dates):
            row = {"Location": loc, "Month": fd}
            for model_name, vals in forecasts.items():
                row[model_name] = vals[i] if i < len(vals) else np.nan
            if best in forecasts:
                row["BestModel"] = best
                row["BestForecast"] = forecasts[best][i]
            all_forecasts.append(row)
    
    forecast_df = pd.DataFrame(all_forecasts)
    forecast_df.to_csv(os.path.join(FORECASTS_DIR, "future_forecasts.csv"), index=False)
    
    # ── Forecast visualization ──
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    for i, loc in enumerate(LOCATION_ORDER):
        ax = axes[i]
        
        # Historical
        hist = monthly[monthly["Location"] == loc].sort_values("Month")
        ax.plot(hist["Month"], hist["Admissions"], color=COLORS[loc],
                linewidth=1.5, label="Historical", alpha=0.9)
        
        # Best model forecast
        fd = all_forecast_data.get(loc, {})
        best = fd.get("best", "SARIMA")
        forecasts = fd.get("forecasts", {})
        ci = fd.get("ci", {})
        
        if best in forecasts:
            ax.plot(future_dates, forecasts[best], "o--", color="white",
                    linewidth=2, markersize=5, label=f"Forecast ({best})")
            
            if best in ci:
                lower, upper = ci[best]
                ax.fill_between(future_dates, lower, upper, alpha=0.2, color="white")
        
        # Vertical line at forecast start
        ax.axvline(pd.Timestamp("2026-07-01"), color="#64748B", linestyle=":",
                   linewidth=1, alpha=0.7)
        
        ax.set_title(loc, fontsize=12)
        ax.set_ylabel("Monthly Admissions")
        ax.legend(fontsize=8, framealpha=0.3)
        ax.grid(True, alpha=0.3)
        stamp(ax)
    
    fig.suptitle("Historical Admissions + 6-Month Forecast (Jul-Dec 2026)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FORECASTS_DIR, "forecast_all_locations.png"))
    plt.close(fig)
    
    # ── Combined forecast comparison chart ──
    fig, ax = plt.subplots(figsize=(14, 7))
    for loc in LOCATION_ORDER:
        fd = all_forecast_data.get(loc, {})
        best = fd.get("best", "SARIMA")
        forecasts = fd.get("forecasts", {})
        if best in forecasts:
            ax.plot(future_dates, forecasts[best], "o-", color=COLORS[loc],
                    linewidth=2, markersize=5, label=loc)
    
    ax.set_title("Forecasted Monthly Admissions -- All Locations (Jul-Dec 2026)", pad=15)
    ax.set_ylabel("Monthly Admissions")
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.3)
    stamp(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FORECASTS_DIR, "forecast_comparison.png"))
    plt.close(fig)
    
    print(f"\n[OK] Stage 6 complete -- forecasts saved to {FORECASTS_DIR}")

if __name__ == "__main__":
    main()
