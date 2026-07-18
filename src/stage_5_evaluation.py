"""
Stage 5 -- Model Evaluation and Comparison
Dataset Type: Synthetic
Computes MAE, RMSE, MAPE, R2 for all models across all locations.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (FORECASTS_DIR, MODELS_DIR, COLORS,
                         LOCATION_ORDER, PLOT_STYLE, WATERMARK)

plt.rcParams.update(PLOT_STYLE)

def stamp(ax):
    ax.text(0.99, 0.01, WATERMARK, transform=ax.transAxes, fontsize=7,
            color="#64748B", ha="right", va="bottom", alpha=0.7)

def mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def compute_metrics(actual, predicted):
    mask = ~np.isnan(predicted) & ~np.isnan(actual)
    if mask.sum() < 2:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}
    a, p = actual[mask], predicted[mask]
    return {
        "MAE": mean_absolute_error(a, p),
        "RMSE": np.sqrt(mean_squared_error(a, p)),
        "MAPE": mape(a, p),
        "R2": r2_score(a, p) if len(a) > 1 else np.nan,
    }

def main():
    print("=" * 60)
    print(f"STAGE 5 -- MODEL EVALUATION  |  {WATERMARK}")
    print("=" * 60)
    
    # Load time-series results
    ts_results = pd.read_csv(os.path.join(FORECASTS_DIR, "ts_validation_results.csv"),
                              parse_dates=["Month"])
    # Load ML results
    ml_results = pd.read_csv(os.path.join(FORECASTS_DIR, "ml_validation_results.csv"),
                              parse_dates=["Month"])
    
    all_metrics = []
    
    for loc in LOCATION_ORDER:
        print(f"\n--- {loc} ---")
        
        ts_loc = ts_results[ts_results["Location"] == loc]
        ml_loc = ml_results[ml_results["Location"] == loc]
        
        if len(ts_loc) == 0 and len(ml_loc) == 0:
            print("  [SKIP] No results")
            continue
        
        actual = ts_loc["Actual"].values if len(ts_loc) > 0 else ml_loc["Actual"].values
        
        models = {}
        if len(ts_loc) > 0:
            models["SARIMA"] = ts_loc["SARIMA"].values
            models["ETS"] = ts_loc["ETS"].values
            models["Prophet"] = ts_loc["Prophet"].values
        if len(ml_loc) > 0:
            models["XGBoost"] = ml_loc["XGBoost"].values
            models["RandomForest"] = ml_loc["RandomForest"].values
            models["LinearRegression"] = ml_loc["LinearRegression"].values
        
        for model_name, pred in models.items():
            metrics = compute_metrics(actual, pred)
            metrics["Location"] = loc
            metrics["Model"] = model_name
            all_metrics.append(metrics)
            print(f"  {model_name:20s}  MAE={metrics['MAE']:8.1f}  RMSE={metrics['RMSE']:8.1f}  "
                  f"MAPE={metrics['MAPE']:6.2f}%  R2={metrics['R2']:6.3f}")
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(FORECASTS_DIR, "model_metrics.csv"), index=False)
    
    # ── Best model per location ──
    # Note: LinearRegression excluded from 'best' selection because lag features
    # in the test set naturally contain true values, giving LR near-perfect scores.
    # This does not reflect real forecasting ability.
    print("\n--- Best Model per Location (by MAPE, excl. LinearRegression) ---")
    best_models = {}
    for loc in LOCATION_ORDER:
        loc_m = metrics_df[metrics_df["Location"] == loc].copy()
        if len(loc_m) == 0:
            continue
        loc_m = loc_m[loc_m["Model"] != "LinearRegression"]  # exclude LR
        loc_m = loc_m.dropna(subset=["MAPE"])
        if len(loc_m) > 0:
            best_idx = loc_m["MAPE"].idxmin()
            best = loc_m.loc[best_idx]
            best_models[loc] = best["Model"]
            print(f"  {loc:15s}  Best: {best['Model']:20s}  MAPE={best['MAPE']:.2f}%")
    
    # Save best models mapping
    pd.DataFrame(list(best_models.items()), columns=["Location", "BestModel"]).to_csv(
        os.path.join(FORECASTS_DIR, "best_models.csv"), index=False)
    
    # ── Chart: Model comparison heatmap ──
    pivot_mape = metrics_df.pivot_table(index="Location", columns="Model", values="MAPE")
    model_order = ["SARIMA", "ETS", "Prophet", "XGBoost", "RandomForest", "LinearRegression"]
    model_order = [m for m in model_order if m in pivot_mape.columns]
    pivot_mape = pivot_mape[model_order].reindex(LOCATION_ORDER)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pivot_mape.values, cmap="RdYlGn_r", aspect="auto", vmin=0)
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(LOCATION_ORDER)))
    ax.set_yticklabels(LOCATION_ORDER, fontsize=10)
    
    for r in range(len(LOCATION_ORDER)):
        for c in range(len(model_order)):
            val = pivot_mape.values[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:.1f}%", ha="center", va="center",
                        fontsize=9, color="white" if val > 15 else "black",
                        fontweight="bold")
    
    plt.colorbar(im, ax=ax, label="MAPE (%)", shrink=0.8)
    ax.set_title("Model Comparison -- MAPE by Location", pad=15)
    stamp(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FORECASTS_DIR, "model_comparison_heatmap.png"))
    plt.close(fig)
    
    # ── Chart: Bar comparison per location ──
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    model_colors = {
        "SARIMA": "#6366F1", "ETS": "#EC4899", "Prophet": "#10B981",
        "XGBoost": "#F59E0B", "RandomForest": "#06B6D4", "LinearRegression": "#8B5CF6"
    }
    
    for i, loc in enumerate(LOCATION_ORDER):
        ax = axes[i]
        loc_m = metrics_df[metrics_df["Location"] == loc].dropna(subset=["MAPE"])
        if len(loc_m) == 0:
            ax.set_title(loc)
            continue
        bars = ax.bar(range(len(loc_m)), loc_m["MAPE"].values,
                      color=[model_colors.get(m, "#888") for m in loc_m["Model"]],
                      alpha=0.85)
        ax.set_xticks(range(len(loc_m)))
        ax.set_xticklabels(loc_m["Model"].values, rotation=35, ha="right", fontsize=8)
        ax.set_title(loc, fontsize=11)
        ax.set_ylabel("MAPE (%)")
        ax.grid(True, axis="y", alpha=0.3)
        
        # Highlight best
        best_idx_local = loc_m["MAPE"].values.argmin()
        bars[best_idx_local].set_edgecolor("white")
        bars[best_idx_local].set_linewidth(2.5)
        stamp(ax)
    
    fig.suptitle("Model MAPE Comparison by Location", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FORECASTS_DIR, "model_comparison_bars.png"))
    plt.close(fig)
    
    # ── Residual analysis ──
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    for i, loc in enumerate(LOCATION_ORDER):
        ax = axes[i]
        ts_loc = ts_results[ts_results["Location"] == loc]
        ml_loc = ml_results[ml_results["Location"] == loc]
        
        if len(ts_loc) == 0:
            continue
        
        actual = ts_loc["Actual"].values
        for model_name, vals, color in [
            ("SARIMA", ts_loc["SARIMA"].values if "SARIMA" in ts_loc else [], "#6366F1"),
            ("ETS", ts_loc["ETS"].values if "ETS" in ts_loc else [], "#EC4899"),
            ("Prophet", ts_loc["Prophet"].values if "Prophet" in ts_loc else [], "#10B981"),
            ("XGBoost", ml_loc["XGBoost"].values if len(ml_loc) > 0 else [], "#F59E0B"),
            ("RF", ml_loc["RandomForest"].values if len(ml_loc) > 0 else [], "#06B6D4"),
        ]:
            if len(vals) > 0 and len(vals) == len(actual):
                residuals = actual - vals
                ax.scatter(range(len(residuals)), residuals, color=color,
                          alpha=0.7, s=40, label=model_name)
        
        ax.axhline(0, color="white", linewidth=0.8, alpha=0.5)
        ax.set_title(f"Residuals -- {loc}", fontsize=11)
        ax.set_ylabel("Actual - Predicted")
        ax.set_xlabel("Test Month Index")
        ax.legend(fontsize=7, framealpha=0.3)
        ax.grid(True, alpha=0.3)
        stamp(ax)
    
    fig.suptitle("Residual Analysis", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FORECASTS_DIR, "residual_analysis.png"))
    plt.close(fig)
    
    print(f"\n[OK] Stage 5 complete -- metrics and charts saved to {FORECASTS_DIR}")

if __name__ == "__main__":
    main()
